#pragma once
// ============================================================================
// Shared host utilities for Inkling-2 (NVFP4 + MXFP8 3D-packed MoE) TFQ3
// Metal & CUDA converters:
//   - Arbitrary-rank mmap safetensors reader (multi-shard or single-file)
//   - Exact modelopt NVFP4 outer scale & non-finite checking
//   - Trace-guided & data-free router-norm expert heat ranking
//   - Standalone residual safetensors + config/tokenizer exporter
// ============================================================================
#include "../tfq_constants.hpp"
#include "../tfq_tables_io.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace ink2 {

static constexpr uint32_t MAGIC_TFPW = 0x57504654u; // "TFPW"
static constexpr uint32_t MAGIC_TFPN = 0x4E504654u; // "TFPN"
static constexpr uint32_t FORMAT_VER = 3u;
static constexpr size_t   ALIGN_16K  = 16384;

static inline size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

struct Ink2Params {
    uint32_t R;
    uint32_t C;
    uint32_t row0;
    uint32_t row_step;
    float    outer;
    uint32_t pad0;
};

static inline float bf16_to_f32_host(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f = 0.0f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// Round-to-nearest-even float32 -> BF16 -> float32 (exact match to __float2bfloat16_rn).
static inline float bf16_round_f32(float x) {
    uint32_t u = 0;
    memcpy(&u, &x, sizeof(u));
    if ((u & 0x7F800000u) == 0x7F800000u) return x; // NaN/Inf
    uint32_t r = u + 0x7FFFu + ((u >> 16) & 1u);
    r &= 0xFFFF0000u;
    float out = 0.0f;
    memcpy(&out, &r, sizeof(out));
    return out;
}

static inline float nvfp4_outer_scale(float amax_e, float den, const std::string& ctx) {
    if (!std::isfinite(amax_e) || !std::isfinite(den) || den == 0.0f) {
        fprintf(stderr, "Error: non-finite or zero NVFP4 outer scale in %s (amax=%g, den=%g)\n",
                ctx.c_str(), amax_e, den);
        exit(1);
    }
    float out = bf16_round_f32(amax_e) / bf16_round_f32(den);
    if (!std::isfinite(out)) {
        fprintf(stderr, "Error: non-finite NVFP4 outer quotient in %s (amax=%g, den=%g)\n",
                ctx.c_str(), amax_e, den);
        exit(1);
    }
    return out;
}

static inline void check_finite_e4m3_bytes(const uint8_t* p, size_t n, const std::string& name) {
    for (size_t i = 0; i < n; ++i) {
        if ((p[i] & 0x7Fu) == 0x7Fu) {
            fprintf(stderr, "Error: non-finite F8_E4M3 NaN byte in %s at element %zu\n", name.c_str(), i);
            exit(1);
        }
    }
}

static inline void check_finite_e8m0_bytes(const uint8_t* p, size_t n, const std::string& name) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] == 0xFFu) {
            fprintf(stderr, "Error: non-finite UE8M0 NaN scale byte in %s at element %zu\n", name.c_str(), i);
            exit(1);
        }
    }
}

static inline void write_all_or_exit(FILE* f, const void* p, size_t n, const std::string& path) {
    if (n && fwrite(p, 1, n, f) != n) {
        fprintf(stderr, "Write error on %s\n", path.c_str());
        exit(1);
    }
}

static inline void close_or_exit(FILE* f, const std::string& path) {
    if (fclose(f) != 0) {
        fprintf(stderr, "Close/flush error on %s\n", path.c_str());
        exit(1);
    }
}

static inline void rename_or_exit(const std::string& from, const std::string& to) {
    if (rename(from.c_str(), to.c_str()) != 0) {
        fprintf(stderr, "Cannot rename %s -> %s\n", from.c_str(), to.c_str());
        exit(1);
    }
}

static inline float e4m3_to_f32_host(uint8_t b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return 0.0f;
    float v = (e == 0) ? (std::ldexp((float)m, -9)) : (std::ldexp(1.0f + (float)m / 8.0f, e - 7));
    return s ? -v : v;
}

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t off = 0;
    uint64_t len = 0;
};

struct ShardCache {
    struct Entry {
        int fd = -1;
        uint8_t* ptr = nullptr;
        size_t len = 0;
        uint64_t data_base = 0;
        std::map<std::string, TensorInfo> tensors;
    };
    std::string model_dir;
    std::map<std::string, std::string> weight_map;
    std::map<std::string, Entry> shards;

    static void parse_header(const std::string& hdr, std::map<std::string, TensorInfo>& out) {
        size_t pos = 0;
        while (pos < hdr.size()) {
            size_t q1 = hdr.find('"', pos);
            if (q1 == std::string::npos) break;
            size_t q2 = hdr.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            std::string key = hdr.substr(q1 + 1, q2 - q1 - 1);
            size_t col = hdr.find(':', q2);
            if (col == std::string::npos) break;
            size_t p = col + 1;
            while (p < hdr.size() && (hdr[p] == ' ' || hdr[p] == '\n' || hdr[p] == '\r' || hdr[p] == '\t')) ++p;
            if (p >= hdr.size() || hdr[p] != '{') { pos = q2 + 1; continue; }
            int depth = 0;
            size_t end_obj = p;
            bool in_str = false;
            for (; end_obj < hdr.size(); ++end_obj) {
                char ch = hdr[end_obj];
                if (ch == '"' && (end_obj == 0 || hdr[end_obj - 1] != '\\')) in_str = !in_str;
                else if (!in_str) {
                    if (ch == '{') depth++;
                    else if (ch == '}') { depth--; if (depth == 0) break; }
                }
            }
            if (key != "__metadata__" && end_obj < hdr.size()) {
                std::string obj = hdr.substr(p, end_obj - p + 1);
                TensorInfo ti;
                size_t dt_k = obj.find("\"dtype\"");
                if (dt_k != std::string::npos) {
                    size_t dq1 = obj.find('"', obj.find(':', dt_k) + 1);
                    size_t dq2 = obj.find('"', dq1 + 1);
                    if (dq1 != std::string::npos && dq2 != std::string::npos)
                        ti.dtype = obj.substr(dq1 + 1, dq2 - dq1 - 1);
                }
                size_t sh_k = obj.find("\"shape\"");
                if (sh_k != std::string::npos) {
                    size_t lb = obj.find('[', sh_k);
                    size_t rb = obj.find(']', lb);
                    if (lb != std::string::npos && rb != std::string::npos) {
                        std::string arr = obj.substr(lb + 1, rb - lb - 1);
                        const char* s = arr.c_str();
                        while (*s) {
                            while (*s && (*s == ' ' || *s == ',' || *s == '\n' || *s == '\t')) ++s;
                            if (!*s) break;
                            char* ep = nullptr;
                            long long v = strtoll(s, &ep, 10);
                            if (ep == s) break;
                            ti.shape.push_back((int64_t)v);
                            s = ep;
                        }
                    }
                }
                size_t do_k = obj.find("\"data_offsets\"");
                if (do_k != std::string::npos) {
                    size_t lb = obj.find('[', do_k);
                    uint64_t a = 0, b = 0;
                    if (lb != std::string::npos && sscanf(obj.c_str() + lb, "[ %llu , %llu ]",
                        (unsigned long long*)&a, (unsigned long long*)&b) == 2 && b >= a) {
                        ti.off = a;
                        ti.len = b - a;
                    }
                }
                out[key] = std::move(ti);
            }
            pos = end_obj + 1;
        }
    }

    void init(const std::string& dir) {
        model_dir = dir;
        std::string idx_path = dir + "/model.safetensors.index.json";
        std::ifstream f(idx_path);
        if (f.good()) {
            std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            size_t pos = txt.find("\"weight_map\"");
            if (pos == std::string::npos) { fprintf(stderr, "No weight_map in %s\n", idx_path.c_str()); exit(1); }
            pos = txt.find('{', pos);
            size_t end = txt.find('}', pos);
            std::string body = txt.substr(pos + 1, end - pos - 1);
            size_t i = 0;
            while (i < body.size()) {
                size_t q1 = body.find('"', i); if (q1 == std::string::npos) break;
                size_t q2 = body.find('"', q1 + 1);
                size_t q3 = body.find('"', q2 + 1);
                size_t q4 = body.find('"', q3 + 1);
                if (q4 == std::string::npos) break;
                weight_map[body.substr(q1 + 1, q2 - q1 - 1)] = body.substr(q3 + 1, q4 - q3 - 1);
                i = q4 + 1;
            }
            return;
        }
        // Single-file fallback: model.safetensors
        std::string single = "model.safetensors";
        Entry& sh = ensure_shard(single);
        for (const auto& kv : sh.tensors) weight_map[kv.first] = single;
    }

    Entry& ensure_shard(const std::string& shard) {
        auto sit = shards.find(shard);
        if (sit != shards.end()) return sit->second;
        std::string path = model_dir + "/" + shard;
        Entry e;
        e.fd = open(path.c_str(), O_RDONLY);
        if (e.fd < 0) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
        struct stat st{};
        fstat(e.fd, &st);
        e.len = (size_t)st.st_size;
        if (e.len < 8) { fprintf(stderr, "Truncated safetensors %s\n", path.c_str()); exit(1); }
        e.ptr = (uint8_t*)mmap(nullptr, e.len, PROT_READ, MAP_PRIVATE, e.fd, 0);
        if (e.ptr == MAP_FAILED) { fprintf(stderr, "mmap failed on %s\n", path.c_str()); exit(1); }
        uint64_t hl = 0;
        memcpy(&hl, e.ptr, 8);
        if (hl > e.len - 8) { fprintf(stderr, "Corrupt safetensors header in %s\n", path.c_str()); exit(1); }
        e.data_base = 8 + hl;
        std::string hdr((const char*)e.ptr + 8, (size_t)hl);
        parse_header(hdr, e.tensors);
        auto res = shards.emplace(shard, std::move(e));
        return res.first->second;
    }

    bool has_tensor(const std::string& name) const { return weight_map.count(name) > 0; }

    const uint8_t* get_tensor(const std::string& name, TensorInfo* out_info = nullptr) {
        auto it = weight_map.find(name);
        if (it == weight_map.end()) {
            fprintf(stderr, "Missing tensor in weight_map: %s\n", name.c_str());
            exit(1);
        }
        Entry& sh = ensure_shard(it->second);
        auto tit = sh.tensors.find(name);
        if (tit == sh.tensors.end()) {
            fprintf(stderr, "Missing tensor %s inside shard %s\n", name.c_str(), it->second.c_str());
            exit(1);
        }
        if (out_info) *out_info = tit->second;
        return sh.ptr + sh.data_base + tit->second.off;
    }

    void evict_all() {
        for (auto& kv : shards) {
            if (kv.second.ptr && kv.second.ptr != MAP_FAILED) munmap(kv.second.ptr, kv.second.len);
            if (kv.second.fd >= 0) close(kv.second.fd);
        }
        shards.clear();
    }
    ~ShardCache() { evict_all(); }
};

static inline std::map<int, std::vector<int>> load_heat_ranks(const std::string& trace_path, int n_exp) {
    std::map<int, std::vector<int>> counts;
    std::ifstream f(trace_path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("LAYER ", 0) == 0) {
            int lyr = 0, eid = 0; long long cnt = 0;
            if (sscanf(line.c_str(), "LAYER %d E%d %lld", &lyr, &eid, &cnt) == 3) {
                if (!counts.count(lyr)) counts[lyr].assign(n_exp, 0);
                if (eid >= 0 && eid < n_exp) counts[lyr][eid] += (int)cnt;
            }
        } else {
            std::istringstream iss(line);
            long long tok = 0; int lyr = -1, eid = -1;
            if ((iss >> tok >> lyr) && lyr >= 0) {
                while (iss >> eid) {
                    if (!counts.count(lyr)) counts[lyr].assign(n_exp, 0);
                    if (eid >= 0 && eid < n_exp) counts[lyr][eid] += 1;
                }
            }
        }
    }
    std::map<int, std::vector<int>> ranks;
    for (auto& kv : counts) {
        std::vector<int> order(n_exp);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return (kv.second[a] != kv.second[b]) ? (kv.second[a] > kv.second[b]) : (a < b);
        });
        std::vector<int> r(n_exp);
        for (int idx = 0; idx < n_exp; ++idx) r[order[idx]] = idx;
        ranks[kv.first] = std::move(r);
    }
    return ranks;
}

// Data-free expert ranking for Inkling-2: router BF16 gate.weight row L2 norm + F32 gate.bias.
static inline std::vector<int> rank_experts_datafree(ShardCache& cache, int lyr, int n_exp, int H) {
    std::vector<double> scores(n_exp, 0.0);
    char gw_name[192], gb_name[192];
    snprintf(gw_name, sizeof(gw_name), "model.llm.layers.%d.mlp.gate.weight", lyr);
    snprintf(gb_name, sizeof(gb_name), "model.llm.layers.%d.mlp.gate.bias", lyr);

    if (cache.has_tensor(gw_name)) {
        TensorInfo ti;
        const uint8_t* p = cache.get_tensor(gw_name, &ti);
        if (ti.dtype == "BF16" && ti.shape.size() == 2 && ti.shape[0] == n_exp && ti.shape[1] == H) {
            const uint16_t* w = reinterpret_cast<const uint16_t*>(p);
            for (int e = 0; e < n_exp; ++e) {
                double ss = 0.0;
                for (int c = 0; c < H; ++c) {
                    double v = (double)bf16_to_f32_host(w[(size_t)e * H + c]);
                    ss += v * v;
                }
                scores[e] = std::sqrt(ss);
            }
        } else if (ti.dtype == "F32" && ti.shape.size() == 2 && ti.shape[0] == n_exp && ti.shape[1] == H) {
            const float* w = reinterpret_cast<const float*>(p);
            for (int e = 0; e < n_exp; ++e) {
                double ss = 0.0;
                for (int c = 0; c < H; ++c) {
                    double v = (double)w[(size_t)e * H + c];
                    ss += v * v;
                }
                scores[e] = std::sqrt(ss);
            }
        }
    }
    if (cache.has_tensor(gb_name)) {
        TensorInfo ti;
        const uint8_t* p = cache.get_tensor(gb_name, &ti);
        if (ti.shape.size() == 1 && ti.shape[0] == n_exp && (ti.dtype == "F32" || ti.dtype == "BF16")) {
            double s_mean = 0.0;
            for (int e = 0; e < n_exp; ++e) s_mean += scores[e];
            s_mean = std::max(s_mean / std::max(n_exp, 1), 1e-6);
            for (int e = 0; e < n_exp; ++e) {
                double bval = (ti.dtype == "F32")
                    ? (double)reinterpret_cast<const float*>(p)[e]
                    : (double)bf16_to_f32_host(reinterpret_cast<const uint16_t*>(p)[e]);
                scores[e] = scores[e] / s_mean + 0.5 * bval;
            }
        }
    }
    std::vector<int> order(n_exp);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return (scores[a] != scores[b]) ? (scores[a] > scores[b]) : (a < b);
    });
    std::vector<int> ranks(n_exp);
    for (int idx = 0; idx < n_exp; ++idx) ranks[order[idx]] = idx;
    return ranks;
}

static inline bool is_expert_tensor(const std::string& name) {
    return name.find(".mlp.experts.") != std::string::npos;
}

static inline void copy_small_file(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in.good()) return;
    std::string tmp = dst + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out.good()) { fprintf(stderr, "Cannot write %s\n", tmp.c_str()); exit(1); }
    out << in.rdbuf();
    out.close();
    if (!out.good()) { fprintf(stderr, "Write error on %s\n", tmp.c_str()); exit(1); }
    ink2::rename_or_exit(tmp, dst);
}

// Copy all non-expert (and non-excluded) tensors into clean residual safetensors shards + index + config/tokenizer.
static inline void export_residuals_and_assets(
    ShardCache& cache,
    const std::string& model_dir,
    const std::string& out_dir,
    const std::set<std::string>& exclude_tensors = {}
) {
    std::vector<std::string> keep;
    keep.reserve(cache.weight_map.size());
    for (const auto& kv : cache.weight_map) {
        if (is_expert_tensor(kv.first)) continue;
        if (exclude_tensors.count(kv.first)) continue;
        keep.push_back(kv.first);
    }
    std::sort(keep.begin(), keep.end());

    // Group by source shard so we open each source shard only once while packing into <= 4.5 GiB residual shards.
    constexpr uint64_t MAX_SHARD_BYTES = 4500ULL * 1024ULL * 1024ULL;
    struct Rec { std::string name; std::string dtype; std::vector<int64_t> shape; uint64_t len; };
    std::vector<std::vector<Rec>> out_shards;
    out_shards.emplace_back();
    uint64_t cur_sz = 0, total_res_bytes = 0;

    // Collect metadata shard-by-shard to keep memory low
    std::map<std::string, std::vector<std::string>> by_src;
    for (const auto& k : keep) by_src[cache.weight_map[k]].push_back(k);
    std::map<std::string, TensorInfo> infos;
    for (const auto& kv : by_src) {
        for (const auto& k : kv.second) {
            TensorInfo ti;
            cache.get_tensor(k, &ti);
            infos[k] = std::move(ti);
        }
        cache.evict_all();
    }
    for (const auto& k : keep) {
        const auto& ti = infos[k];
        if (!out_shards.back().empty() && cur_sz + ti.len > MAX_SHARD_BYTES) {
            out_shards.emplace_back();
            cur_sz = 0;
        }
        out_shards.back().push_back({k, ti.dtype, ti.shape, ti.len});
        cur_sz += ti.len;
        total_res_bytes += ti.len;
    }

    int n_sh = (int)out_shards.size();
    std::map<std::string, std::string> new_wm;
    for (int si = 0; si < n_sh; ++si) {
        char fname[64];
        snprintf(fname, sizeof(fname), "residual-%05d-of-%05d.safetensors", si + 1, n_sh);
        std::string fpath = out_dir + "/" + fname;
        std::string tmp = fpath + ".tmp";

        std::ostringstream hdr;
        hdr << "{";
        uint64_t off = 0;
        for (size_t i = 0; i < out_shards[si].size(); ++i) {
            const auto& r = out_shards[si][i];
            if (i > 0) hdr << ",";
            hdr << "\"" << r.name << "\":{\"dtype\":\"" << r.dtype << "\",\"shape\":[";
            for (size_t d = 0; d < r.shape.size(); ++d) {
                if (d > 0) hdr << ",";
                hdr << r.shape[d];
            }
            hdr << "],\"data_offsets\":[" << off << "," << (off + r.len) << "]}";
            off += r.len;
            new_wm[r.name] = fname;
        }
        hdr << "}";
        std::string hstr = hdr.str();
        while ((hstr.size() & 7u) != 0u) hstr.push_back(' ');
        uint64_t hlen = hstr.size();

        FILE* fo = fopen(tmp.c_str(), "wb");
        if (!fo) { fprintf(stderr, "Cannot write %s\n", tmp.c_str()); exit(1); }
        ink2::write_all_or_exit(fo, &hlen, 8, tmp);
        ink2::write_all_or_exit(fo, hstr.data(), hstr.size(), tmp);
        for (const auto& r : out_shards[si]) {
            const uint8_t* ptr = cache.get_tensor(r.name);
            ink2::write_all_or_exit(fo, ptr, (size_t)r.len, tmp);
        }
        ink2::close_or_exit(fo, tmp);
        ink2::rename_or_exit(tmp, fpath);
        cache.evict_all();
    }

    // Write model.safetensors.index.json
    std::string idx_out = out_dir + "/model.safetensors.index.json";
    std::string idx_tmp = idx_out + ".tmp";
    FILE* fi = fopen(idx_tmp.c_str(), "wb");
    if (!fi) { fprintf(stderr, "Cannot write %s\n", idx_tmp.c_str()); exit(1); }
    std::ostringstream j;
    j << "{\n  \"metadata\": {\n    \"total_size\": " << total_res_bytes << "\n  },\n  \"weight_map\": {\n";
    size_t idx = 0;
    for (const auto& kv : new_wm) {
        j << "    \"" << kv.first << "\": \"" << kv.second << "\"" << (++idx < new_wm.size() ? "," : "") << "\n";
    }
    j << "  }\n}\n";
    std::string jstr = j.str();
    ink2::write_all_or_exit(fi, jstr.data(), jstr.size(), idx_tmp);
    ink2::close_or_exit(fi, idx_tmp);
    ink2::rename_or_exit(idx_tmp, idx_out);

    // Copy non-safetensors metadata / tokenizer / config assets
    for (const char* f : {"config.json", "generation_config.json", "chat_template.jinja",
                          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                          "preprocessor_config.json", "processor_config.json", "LICENSE"}) {
        copy_small_file(model_dir + "/" + f, out_dir + "/" + f);
    }
    printf("RESIDUALS EXPORTED: %zu tensors -> %d shard(s) (%.3f GiB)\n",
           keep.size(), n_sh, (double)total_res_bytes / (double)(1ULL << 30));
    fflush(stdout);
}

} // namespace ink2
