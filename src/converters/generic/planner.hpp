#pragma once
// ============================================================================
// Model-Agnostic TFQ3 Planner, Arbitrary-Shape Safetensors Reader,
// Zero-Data Sensitivity & --target-gib / --fit auto Allocator, and Packager.
//
// Shared by both Apple Silicon Metal (convert.mm) and NVIDIA CUDA (convert.cu)
// so both platforms produce identical sharding, bit allocation, padded TFPN v3
// metadata, residual.safetensors, and tfq3_manifest.json.
// ============================================================================
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

namespace tfq_generic {

static constexpr uint32_t TFQN_FLAG_PADDED  = 1u << 0;
static constexpr uint32_t TFQN_FLAG_VISION  = 1u << 1;
static constexpr uint32_t TFQN_FLAG_DRAFTER = 1u << 2;
static constexpr uint32_t TFQN_FLAG_EMBED   = 1u << 3;

static inline uint32_t align128(uint32_t x) { return (x + 127u) & ~127u; }
static inline uint64_t align16k(uint64_t x) { return (x + 16383ULL) & ~16383ULL; }

enum class Role {
    EMBED,
    LM_HEAD,
    VISION,
    DRAFTER,
    ATTN,
    MLP,
    EXPERT,
    OTHER_LINEAR,
    RESIDUAL
};

static inline const char* role_name(Role r) {
    switch (r) {
        case Role::EMBED:        return "embed";
        case Role::LM_HEAD:      return "lm_head";
        case Role::VISION:       return "vision";
        case Role::DRAFTER:      return "drafter";
        case Role::ATTN:         return "attn";
        case Role::MLP:          return "mlp";
        case Role::EXPERT:       return "expert";
        case Role::OTHER_LINEAR: return "linear";
        case Role::RESIDUAL:     return "residual";
    }
    return "residual";
}

struct TensorInfo {
    std::string name;
    std::string shard_file;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t off = 0;
    uint64_t len = 0;
};

struct SafetensorsDir {
    struct MappedFile {
        int fd = -1;
        uint8_t* ptr = nullptr;
        size_t len = 0;
        uint64_t data_base = 0;
    };
    std::string dir;
    std::map<std::string, TensorInfo> tensors;
    std::vector<std::string> tensor_order;
    std::map<std::string, MappedFile> files;

    static void parse_header(const std::string& shard_file, const std::string& hdr,
                             std::map<std::string, TensorInfo>& out,
                             std::vector<std::string>& order) {
        size_t p = 0;
        while (true) {
            size_t dk = hdr.find("\"data_offsets\"", p);
            if (dk == std::string::npos) break;
            size_t ob = hdr.rfind('{', dk);
            size_t cb = hdr.find('}', dk);
            size_t col = hdr.rfind(':', ob);
            size_t q2 = hdr.rfind('"', col);
            size_t q1 = hdr.rfind('"', q2 - 1);
            if (ob == std::string::npos || cb == std::string::npos || q1 == std::string::npos) break;
            std::string tname = hdr.substr(q1 + 1, q2 - q1 - 1);
            std::string obj = hdr.substr(ob, cb - ob + 1);

            size_t lb = hdr.find('[', dk), cm = hdr.find(',', lb), rb = hdr.find(']', cm);
            uint64_t o0 = std::stoull(hdr.substr(lb + 1, cm - lb - 1));
            uint64_t o1 = std::stoull(hdr.substr(cm + 1, rb - cm - 1));

            TensorInfo ti;
            ti.name = tname;
            ti.shard_file = shard_file;
            ti.off = o0;
            ti.len = o1 - o0;

            size_t dp = obj.find("\"dtype\"");
            if (dp != std::string::npos) {
                size_t d1 = obj.find('"', obj.find(':', dp) + 1);
                size_t d2 = obj.find('"', d1 + 1);
                ti.dtype = obj.substr(d1 + 1, d2 - d1 - 1);
            }
            size_t sp = obj.find("\"shape\"");
            if (sp != std::string::npos) {
                size_t slb = obj.find('[', sp), srb = obj.find(']', slb);
                std::string sb = obj.substr(slb + 1, srb - slb - 1);
                size_t si = 0;
                while (si < sb.size()) {
                    while (si < sb.size() && (sb[si] == ' ' || sb[si] == ',' || sb[si] == '\t' || sb[si] == '\n' || sb[si] == '\r')) ++si;
                    if (si >= sb.size()) break;
                    size_t sj = si;
                    while (sj < sb.size() && sb[sj] >= '0' && sb[sj] <= '9') ++sj;
                    if (sj == si) { fprintf(stderr, "Bad shape for %s in %s\n", tname.c_str(), shard_file.c_str()); exit(1); }
                    ti.shape.push_back(std::stoull(sb.substr(si, sj - si)));
                    si = sj;
                }
            }
            if (!out.count(tname)) order.push_back(tname);
            out[tname] = std::move(ti);
            p = cb + 1;
        }
    }

    void map_file(const std::string& fn) {
        if (files.count(fn)) return;
        std::string full = dir + "/" + fn;
        int fd = open(full.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "Cannot open safetensors file: %s\n", full.c_str()); exit(1); }
        struct stat st{};
        fstat(fd, &st);
        size_t len = (size_t)st.st_size;
        uint8_t* ptr = (uint8_t*)mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) { fprintf(stderr, "mmap failed on %s\n", full.c_str()); exit(1); }
        uint64_t hlen = 0;
        memcpy(&hlen, ptr, 8);
        std::string hdr((const char*)ptr + 8, hlen);
        parse_header(fn, hdr, tensors, tensor_order);
        files[fn] = {fd, ptr, len, 8 + hlen};
    }

    void init(const std::string& model_dir) {
        dir = model_dir;
        std::set<std::string> shard_files;
        std::ifstream idx(dir + "/model.safetensors.index.json");
        if (idx.good()) {
            std::string txt((std::istreambuf_iterator<char>(idx)), std::istreambuf_iterator<char>());
            size_t pos = txt.find("\"weight_map\"");
            if (pos != std::string::npos) {
                pos = txt.find('{', pos);
                size_t end = txt.find('}', pos);
                std::string body = txt.substr(pos + 1, end - pos - 1);
                size_t i = 0;
                while (i < body.size()) {
                    size_t q1 = body.find('"', i); if (q1 == std::string::npos) break;
                    size_t q2 = body.find('"', q1 + 1);
                    size_t q3 = body.find('"', q2 + 1);
                    size_t q4 = body.find('"', q3 + 1);
                    shard_files.insert(body.substr(q3 + 1, q4 - q3 - 1));
                    i = q4 + 1;
                }
            }
        }
        if (shard_files.empty()) {
            DIR* d = opendir(dir.c_str());
            if (!d) { fprintf(stderr, "Cannot open directory: %s\n", dir.c_str()); exit(1); }
            while (auto* ent = readdir(d)) {
                std::string fn = ent->d_name;
                if (fn.size() > 12 && fn.substr(fn.size() - 12) == ".safetensors") {
                    shard_files.insert(fn);
                }
            }
            closedir(d);
        }
        if (shard_files.empty()) {
            fprintf(stderr, "No .safetensors files found in %s\n", dir.c_str());
            exit(1);
        }
        for (const auto& fn : shard_files) map_file(fn);
    }

    bool has(const std::string& name) const { return tensors.count(name) > 0; }

    const uint8_t* data_ptr(const TensorInfo& ti) const {
        const auto& mf = files.at(ti.shard_file);
        return mf.ptr + mf.data_base + ti.off;
    }

    ~SafetensorsDir() {
        for (auto& kv : files) {
            if (kv.second.ptr && kv.second.ptr != MAP_FAILED) munmap(kv.second.ptr, kv.second.len);
            if (kv.second.fd >= 0) close(kv.second.fd);
        }
    }
};

struct GenericOptions {
    std::string model_dir;
    std::string out_dir;
    std::string tables_path;
    std::string profile = "4b"; // 4b | 3.5b | 3b | 5b | auto
    uint32_t default_bits = 4;
    uint32_t head_bits = 4;
    uint32_t max_bits = 5;
    double target_gib = 0.0;
    bool fit_auto = false;
    bool quantize_vision = false;
    bool quantize_drafter = false;
    bool quantize_embed = false;
    bool dry_run = false;
    uint64_t max_shard_bytes = (uint64_t)(3.4 * (double)(1ULL << 30));
};

struct QuantJob {
    std::string weight_name;   // source safetensors key (e.g. model.layers.0.mlp.gate_proj.weight)
    std::string tfpn_name;     // stored name without ".weight" (<= 79 chars)
    std::string src_format;    // BF16 | F16 | F32 | FP8 | MLX_Q8
    std::string scales_name;   // optional .scales or .weight_scale_inv
    std::string biases_name;   // optional .biases (for MLX Q8)
    uint32_t orig_R = 0, orig_C = 0;
    uint32_t pad_R = 0, pad_C = 0;
    uint32_t group_size = 0;
    uint32_t bits = 4;
    uint32_t flags = 0;
    int layer_idx = -1;
    Role role = Role::OTHER_LINEAR;
    double priority = 0.0;
    float snr_db = 0.0f;
};

static inline bool contains(const std::string& s, const char* sub) {
    return s.find(sub) != std::string::npos;
}

static inline int extract_layer_idx(const std::string& name) {
    for (const char* pat : {"layers.", "blocks.", ".h.", "layer."}) {
        size_t pos = name.find(pat);
        if (pos == std::string::npos) continue;
        pos += strlen(pat);
        if (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
            return atoi(name.c_str() + pos);
        }
    }
    return -1;
}

static inline bool is_drafter_config(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f.good()) return false;
    std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::string low = txt;
    for (auto& c : low) c = (char)std::tolower((unsigned char)c);
    return contains(low, "\"dflash") || contains(low, "\"dspine") || contains(low, "draft_layers") || contains(low, "target_layer_ids");
}

// Every "bits" value inside config.json's MLX "quantization" object (global and per-module overrides).
static inline std::vector<int> mlx_config_bits(const std::string& model_dir) {
    std::vector<int> bits;
    std::ifstream f(model_dir + "/config.json");
    if (!f.good()) return bits;
    std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t q = txt.find("\"quantization\"");
    if (q == std::string::npos) return bits;
    size_t ob = txt.find('{', q);
    if (ob == std::string::npos) return bits;
    size_t end = ob;
    for (int depth = 0; end < txt.size(); ++end) {
        if (txt[end] == '{') ++depth;
        else if (txt[end] == '}' && --depth == 0) break;
    }
    std::string obj = txt.substr(ob, end - ob);
    for (size_t p = obj.find("\"bits\""); p != std::string::npos; p = obj.find("\"bits\"", p + 6)) {
        size_t c = obj.find(':', p);
        if (c != std::string::npos) bits.push_back(atoi(obj.c_str() + c + 1));
    }
    return bits;
}

static inline float read_elem_f32(const uint8_t* ptr, const std::string& dt, uint64_t idx) {
    if (dt == "BF16") {
        uint16_t u; memcpy(&u, ptr + idx * 2, 2);
        uint32_t f = (uint32_t)u << 16;
        float v; memcpy(&v, &f, 4); return v;
    }
    if (dt == "F32") {
        float v; memcpy(&v, ptr + idx * 4, 4); return v;
    }
    if (dt == "F16") {
        uint16_t h; memcpy(&h, ptr + idx * 2, 2);
        uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u;
        if (e == 31u) {
            if (m != 0u) return std::nanf("");
            return s ? -INFINITY : INFINITY;
        }
        float v = (e == 0u) ? std::ldexp((float)m, -24)
                            : std::ldexp(1.0f + (float)m / 1024.0f, (int)e - 15);
        return s ? -v : v;
    }
    if (dt == "F8_E4M3") {
        uint8_t b = ptr[idx];
        int s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
        if (e == 15 && m == 7) return std::nanf("");
        float v = (e == 0) ? std::ldexp((float)m, -9) : std::ldexp(1.0f + (float)m / 8.0f, e - 7);
        return s ? -v : v;
    }
    return 0.0f;
}

static inline double sample_kurtosis_bonus(const SafetensorsDir& st, const TensorInfo& ti) {
    uint64_t n = 1;
    for (auto d : ti.shape) n *= d;
    if (n == 0 || (ti.dtype != "BF16" && ti.dtype != "F16" && ti.dtype != "F32" && ti.dtype != "F8_E4M3")) return 0.0;
    const uint8_t* ptr = st.data_ptr(ti);
    const uint64_t samples = std::min<uint64_t>(4096ULL, n);
    const uint64_t stride = std::max<uint64_t>(1ULL, n / samples);
    double s2 = 0.0, s4 = 0.0;
    uint64_t cnt = 0;
    for (uint64_t i = 0; i < n && cnt < samples; i += stride, ++cnt) {
        double v = (double)read_elem_f32(ptr, ti.dtype, i);
        if (!std::isfinite(v)) continue;
        double v2 = v * v;
        s2 += v2;
        s4 += v2 * v2;
    }
    if (cnt == 0 || s2 <= 1e-18) return 0.0;
    double m2 = s2 / (double)cnt;
    double m4 = s4 / (double)cnt;
    double kurt = m4 / (m2 * m2 + 1e-18);
    if (kurt <= 3.0) return 0.0;
    return std::min(20.0, std::log2(kurt / 3.0) * 5.0);
}

static inline uint64_t tfpn_tensor_bytes(uint32_t pad_R, uint32_t pad_C, uint32_t bits) {
    uint64_t words = (uint64_t)(pad_R / 16u) * (pad_C / 128u) * (bits * 2u) * 32ULL;
    uint64_t sc = (uint64_t)(pad_R / 128u) * (pad_C / 128u);
    return words * 4ULL + sc * 2ULL;
}

static inline std::string make_tfpn_name(const std::string& weight_name) {
    std::string s = weight_name;
    if (s.size() > 7 && s.substr(s.size() - 7) == ".weight") s.resize(s.size() - 7);
    if (s.size() >= 80) s = s.substr(s.size() - 79);
    return s;
}

struct ConversionPlan {
    std::vector<QuantJob> jobs;
    std::vector<std::string> residual_names;
    std::vector<std::vector<size_t>> shards; // indices into jobs
    uint64_t residual_bytes = 0;
    uint64_t tfq_bytes = 0;
    uint64_t total_params = 0;
    int max_layer = 0;
    bool detected_drafter = false;
};

static inline ConversionPlan build_plan(const SafetensorsDir& st, const GenericOptions& opt) {
    ConversionPlan plan;
    plan.detected_drafter = opt.quantize_drafter || is_drafter_config(opt.model_dir);

    const std::vector<int> mlx_bits = mlx_config_bits(opt.model_dir);
    std::set<std::string> consumed;
    for (const auto& name : st.tensor_order) {
        const auto& ti = st.tensors.at(name);
        int l = extract_layer_idx(name);
        if (l > plan.max_layer) plan.max_layer = l;

        bool is_weight = (name.size() > 7 && name.substr(name.size() - 7) == ".weight");
        bool is_2d = (ti.shape.size() == 2);
        if (!is_weight || !is_2d) continue;

        std::string stem = name.substr(0, name.size() - 7);
        std::string fmt;
        std::string scales_name, biases_name;
        uint32_t orig_R = (uint32_t)ti.shape[0];
        uint32_t orig_C = (uint32_t)ti.shape[1];
        uint32_t group_size = 0;

        if (ti.dtype == "BF16" || ti.dtype == "F16" || ti.dtype == "F32") {
            fmt = ti.dtype;
        } else if (ti.dtype == "F8_E4M3") {
            fmt = "FP8";
            if (st.has(stem + ".weight_scale_inv")) scales_name = stem + ".weight_scale_inv";
            else if (st.has(stem + ".weight_scale")) scales_name = stem + ".weight_scale";
        } else if (ti.dtype == "U32" && st.has(stem + ".scales") && st.has(stem + ".biases")) {
            for (int b : mlx_bits) {
                if (b != 8) {
                    fprintf(stderr, "MLX %d-bit weights (%s) are not supported: the generic converter reads MLX 8-bit only. "
                            "Convert from the BF16/F16/FP8 source or an MLX 8-bit checkpoint.\n", b, name.c_str());
                    exit(1);
                }
            }
            const auto& si = st.tensors.at(stem + ".scales");
            if (si.shape.size() == 2 && si.shape[0] == orig_R) {
                uint32_t full_C = orig_C * 4u; // 8-bit MLX packs 4 bytes per U32
                uint32_t gs = (si.shape[1] > 0) ? (full_C / (uint32_t)si.shape[1]) : 0;
                if (gs == 32u || gs == 64u || gs == 128u) {
                    fmt = "MLX_Q8";
                    orig_C = full_C;
                    group_size = gs;
                    scales_name = stem + ".scales";
                    biases_name = stem + ".biases";
                }
            }
        }
        if (fmt.empty()) continue;

        // Classify role.
        Role role = Role::OTHER_LINEAR;
        bool is_vis = contains(name, "visual.") || contains(name, "vision_tower.") || contains(name, "vision_model.") || contains(name, "multi_modal_projector.");
        if (contains(name, "embed_tokens") || contains(name, "word_embeddings") || contains(name, "wte.")) {
            role = Role::EMBED;
        } else if (name == "lm_head.weight" || contains(name, ".lm_head.weight") || name == "output.weight") {
            role = Role::LM_HEAD;
        } else if (is_vis) {
            role = Role::VISION;
        } else if (plan.detected_drafter) {
            role = Role::DRAFTER;
        } else if (contains(name, "experts.") || contains(name, "switch_mlp.")) {
            role = Role::EXPERT;
        } else if (contains(name, "self_attn") || contains(name, "linear_attn") || contains(name, "attention") ||
                   contains(name, "q_proj") || contains(name, "k_proj") || contains(name, "v_proj") ||
                   contains(name, "o_proj") || contains(name, "qkv") || contains(name, "in_proj") || contains(name, "out_proj")) {
            role = Role::ATTN;
        } else if (contains(name, "mlp.") || contains(name, "feed_forward.") || contains(name, "gate_proj") ||
                   contains(name, "up_proj") || contains(name, "down_proj") || contains(name, "fc1") || contains(name, "fc2")) {
            role = Role::MLP;
        }

        // Skip tiny projections (routers, small gating heads, <128 in either dim).
        if (orig_R < 128u || orig_C < 128u) continue;
        if (role == Role::EMBED && !opt.quantize_embed) continue;
        if (role == Role::VISION && !opt.quantize_vision) continue;
        if (role == Role::LM_HEAD && opt.head_bits == 0u) continue;

        QuantJob job;
        job.weight_name = name;
        job.tfpn_name   = make_tfpn_name(name);
        job.src_format  = fmt;
        job.scales_name = scales_name;
        job.biases_name = biases_name;
        job.orig_R      = orig_R;
        job.orig_C      = orig_C;
        job.pad_R       = align128(orig_R);
        job.pad_C       = align128(orig_C);
        job.group_size  = group_size;
        job.layer_idx   = l;
        job.role        = role;
        job.bits        = (role == Role::LM_HEAD) ? opt.head_bits : opt.default_bits;
        if (job.pad_R != job.orig_R || job.pad_C != job.orig_C) job.flags |= TFQN_FLAG_PADDED;
        if (role == Role::VISION)  job.flags |= TFQN_FLAG_VISION;
        if (role == Role::DRAFTER) job.flags |= TFQN_FLAG_DRAFTER;
        if (role == Role::EMBED)   job.flags |= TFQN_FLAG_EMBED;

        plan.jobs.push_back(job);
        consumed.insert(name);
        if (!scales_name.empty()) consumed.insert(scales_name);
        if (!biases_name.empty()) consumed.insert(biases_name);
    }

    // Collect residuals and exact byte footprint.
    for (const auto& name : st.tensor_order) {
        if (consumed.count(name)) continue;
        plan.residual_names.push_back(name);
        plan.residual_bytes += st.tensors.at(name).len;
    }

    // Compute data-free priority score for each job (role + boundary layers + excess kurtosis).
    //
    // Why role + boundary + kurtosis instead of greedy weight-error-per-byte (see issue #25):
    //   After 2D FWHT-128, per-tier relative weight error is nearly constant across tensors
    //   (3b ~0.0185, 4b ~0.0046, 5b ~0.0012, <10% spread). On Qwen2.5-1.5B at matched byte
    //   budgets (3.07 / 3.25 / 3.50 bpw), greedy weight-SSE-per-byte cuts weight SSE by 4.7-7.1%
    //   but spends budget on large MLPs and degrades WikiText-2 PPL/KL; RMSNorm-gamma-weighted
    //   SSE narrows only part of the gap, and relative-error-per-byte over-upgrades small tensors
    //   to 5b and also loses. Prioritizing attention + boundary layers outperforms all three
    //   data-free weight-error objectives; activation-calibrated output-error scoring
    //   tr(dW H dW^T) is a separate concern requiring calibration data.
    const int n_layers = plan.max_layer + 1;
    for (auto& j : plan.jobs) {
        double base = 50.0;
        switch (j.role) {
            case Role::LM_HEAD:      base = 100.0; break;
            case Role::DRAFTER:      base =  85.0; break;
            case Role::VISION:       base =  80.0; break;
            case Role::EMBED:        base =  75.0; break;
            case Role::ATTN:         base =  70.0; break;
            case Role::OTHER_LINEAR: base =  50.0; break;
            case Role::MLP:          base =  40.0; break;
            case Role::EXPERT:       base =  25.0; break;
            case Role::RESIDUAL:     base =   0.0; break;
        }
        if (j.layer_idx >= 0 && n_layers >= 4) {
            int boundary_near = std::max(2, n_layers / 10);
            if (j.layer_idx < 2 || j.layer_idx >= n_layers - 2) base += 25.0;
            else if (j.layer_idx < boundary_near || j.layer_idx >= n_layers - boundary_near) base += 12.0;
        }
        base += sample_kurtosis_bonus(st, st.tensors.at(j.weight_name));
        j.priority = base;
    }

    // Apply profile or --target-gib / --fit auto allocation.
    if (opt.target_gib > 0.0) {
        uint64_t target_bytes = (uint64_t)(opt.target_gib * (double)(1ULL << 30));
        uint64_t overhead = 65536ULL * 4ULL;
        uint64_t budget = (target_bytes > plan.residual_bytes + overhead) ? (target_bytes - plan.residual_bytes - overhead) : 0ULL;
        uint64_t cur = 0;
        for (auto& j : plan.jobs) {
            j.bits = 3u;
            cur += tfpn_tensor_bytes(j.pad_R, j.pad_C, 3u);
        }
        if (cur > budget) {
            double min_gib = (double)(cur + plan.residual_bytes + overhead) / (double)(1ULL << 30);
            fprintf(stderr, "Budget %.3f GiB is below the minimum for this model: %.3f GiB (all quantized tensors at 3 bits + residual)\n",
                    opt.target_gib, min_gib);
            exit(1);
        }
        std::vector<size_t> idx(plan.jobs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return plan.jobs[a].priority > plan.jobs[b].priority;
        });
        // Pass 1: upgrade 3b -> 4b by descending sensitivity.
        for (size_t i : idx) {
            auto& j = plan.jobs[i];
            uint64_t d = tfpn_tensor_bytes(j.pad_R, j.pad_C, 4u) - tfpn_tensor_bytes(j.pad_R, j.pad_C, 3u);
            if (cur + d <= budget) { j.bits = 4u; cur += d; }
        }
        // Pass 2: if headroom remains and max_bits >= 5, upgrade highest-sensitivity tensors 4b -> 5b.
        if (opt.max_bits >= 5u) {
            for (size_t i : idx) {
                auto& j = plan.jobs[i];
                if (j.bits != 4u) continue;
                uint64_t d = tfpn_tensor_bytes(j.pad_R, j.pad_C, 5u) - tfpn_tensor_bytes(j.pad_R, j.pad_C, 4u);
                if (cur + d <= budget) { j.bits = 5u; cur += d; }
            }
        }
    } else if (opt.profile == "3.5b") {
        int bnd = std::max(2, n_layers / 10);
        for (auto& j : plan.jobs) {
            if (j.role == Role::LM_HEAD) j.bits = opt.head_bits;
            else if (j.role == Role::ATTN || j.role == Role::VISION || j.role == Role::DRAFTER || j.role == Role::EMBED) j.bits = 4u;
            else if (j.layer_idx >= 0 && (j.layer_idx < bnd || j.layer_idx >= n_layers - bnd)) j.bits = 4u;
            else j.bits = 3u;
        }
    } else if (opt.profile == "3b") {
        for (auto& j : plan.jobs) j.bits = (j.role == Role::LM_HEAD) ? opt.head_bits : 3u;
    } else if (opt.profile == "5b") {
        for (auto& j : plan.jobs) j.bits = (j.role == Role::LM_HEAD) ? opt.head_bits : 5u;
    }

    // Partition jobs into <= max_shard_bytes shards (16 GB Mac maxBufferLength safe).
    std::vector<size_t> cur_shard;
    uint64_t cur_bytes = 0;
    for (size_t i = 0; i < plan.jobs.size(); ++i) {
        const auto& j = plan.jobs[i];
        uint64_t tb = tfpn_tensor_bytes(j.pad_R, j.pad_C, j.bits);
        plan.tfq_bytes += tb;
        plan.total_params += (uint64_t)j.orig_R * (uint64_t)j.orig_C;
        if (!cur_shard.empty() && cur_bytes + tb > opt.max_shard_bytes) {
            plan.shards.push_back(std::move(cur_shard));
            cur_shard.clear();
            cur_bytes = 0;
        }
        cur_shard.push_back(i);
        cur_bytes += tb;
    }
    if (!cur_shard.empty()) plan.shards.push_back(std::move(cur_shard));
    return plan;
}

// Decode a row slice [r_start, r_start + r_count) of any supported tensor into padded float32 [r_count, pad_C].
static inline void decode_chunk_to_f32(
    const SafetensorsDir& st, const QuantJob& job,
    uint32_t r_start, uint32_t r_count, std::vector<float>& out
) {
    out.assign((size_t)r_count * job.pad_C, 0.0f);
    const auto& ti = st.tensors.at(job.weight_name);
    const uint8_t* wptr = st.data_ptr(ti);
    const uint32_t r_valid = (r_start < job.orig_R) ? std::min(r_count, job.orig_R - r_start) : 0u;
    if (r_valid == 0u) return;

    if (job.src_format == "BF16") {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(wptr);
        for (uint32_t r = 0; r < r_valid; ++r) {
            const uint16_t* src = p + (size_t)(r_start + r) * job.orig_C;
            float* dst = out.data() + (size_t)r * job.pad_C;
            for (uint32_t c = 0; c < job.orig_C; ++c) {
                uint32_t u = (uint32_t)src[c] << 16;
                memcpy(&dst[c], &u, 4);
            }
        }
    } else if (job.src_format == "F32") {
        const float* p = reinterpret_cast<const float*>(wptr);
        for (uint32_t r = 0; r < r_valid; ++r) {
            memcpy(out.data() + (size_t)r * job.pad_C, p + (size_t)(r_start + r) * job.orig_C, job.orig_C * 4ULL);
        }
    } else if (job.src_format == "F16") {
        for (uint32_t r = 0; r < r_valid; ++r) {
            size_t base = (size_t)(r_start + r) * job.orig_C;
            float* dst = out.data() + (size_t)r * job.pad_C;
            for (uint32_t c = 0; c < job.orig_C; ++c) dst[c] = read_elem_f32(wptr, "F16", base + c);
        }
    } else if (job.src_format == "FP8") {
        const uint8_t* sptr = nullptr;
        std::string sdt;
        uint32_t sR = 0, sC = 0;
        if (!job.scales_name.empty()) {
            const auto& si = st.tensors.at(job.scales_name);
            sptr = st.data_ptr(si);
            sdt = si.dtype;
            sR = si.shape.size() >= 1 ? (uint32_t)si.shape[0] : 1u;
            sC = si.shape.size() >= 2 ? (uint32_t)si.shape[1] : 1u;
        }
        float fp8_lut[256];
        for (int i = 0; i < 256; ++i) {
            uint8_t b = (uint8_t)i;
            fp8_lut[i] = read_elem_f32(&b, "F8_E4M3", 0);
        }
        for (uint32_t r = 0; r < r_valid; ++r) {
            uint32_t gr = r_start + r;
            const uint8_t* src = wptr + (size_t)gr * job.orig_C;
            float* dst = out.data() + (size_t)r * job.pad_C;
            for (uint32_t c = 0; c < job.orig_C; ++c) {
                float sc = 1.0f;
                if (sptr) {
                    if (sR == 1u && sC == 1u) sc = read_elem_f32(sptr, sdt, 0);
                    else if (sR == job.orig_R && sC == 1u) sc = read_elem_f32(sptr, sdt, gr); // per-channel [R,1] / [R]
                    else {
                        uint32_t sr = std::min(sR - 1u, gr >> 7);
                        uint32_t sc_i = std::min(sC - 1u, c >> 7);
                        sc = read_elem_f32(sptr, sdt, (uint64_t)sr * sC + sc_i);
                    }
                }
                dst[c] = fp8_lut[src[c]] * sc;
            }
        }
    } else if (job.src_format == "MLX_Q8") {
        const auto& si = st.tensors.at(job.scales_name);
        const auto& bi = st.tensors.at(job.biases_name);
        const uint8_t* sptr = st.data_ptr(si);
        const uint8_t* bptr = st.data_ptr(bi);
        uint32_t n_grp = (uint32_t)si.shape[1];
        for (uint32_t r = 0; r < r_valid; ++r) {
            uint32_t gr = r_start + r;
            const uint8_t* src = wptr + (size_t)gr * job.orig_C;
            float* dst = out.data() + (size_t)r * job.pad_C;
            for (uint32_t g = 0; g < n_grp; ++g) {
                float sc = read_elem_f32(sptr, si.dtype, (uint64_t)gr * n_grp + g);
                float bs = read_elem_f32(bptr, bi.dtype, (uint64_t)gr * n_grp + g);
                uint32_t c0 = g * job.group_size;
                uint32_t c1 = std::min(job.orig_C, c0 + job.group_size);
                for (uint32_t c = c0; c < c1; ++c) dst[c] = (float)src[c] * sc + bs;
            }
        }
    }

    // Reject NaN/Inf weights at conversion time so a damaged source checkpoint
    // never silently corrupts a 128x128 rotated block and its fp16 scale.
    for (uint32_t r = 0; r < r_valid; ++r) {
        const float* row = out.data() + (size_t)r * job.pad_C;
        for (uint32_t c = 0; c < job.orig_C; ++c) {
            if (!std::isfinite(row[c])) {
                fprintf(stderr, "Non-finite weight (%s) in tensor '%s' at [%u, %u]; refusing to quantize\n",
                        std::isnan(row[c]) ? "NaN" : "Inf", job.weight_name.c_str(), r_start + r, c);
                exit(1);
            }
        }
    }
}

// Output writes: a short write or failed close (e.g. disk full) must stop the conversion with a nonzero exit.
static inline void fwrite_or_die(const void* p, size_t n, FILE* f, const std::string& path) {
    if (n && fwrite(p, 1, n, f) != n) { fprintf(stderr, "Write error on %s\n", path.c_str()); exit(1); }
}

static inline void fclose_or_die(FILE* f, const std::string& path) {
    if (fclose(f) != 0) { fprintf(stderr, "Write error on %s\n", path.c_str()); exit(1); }
}

static inline void close_or_die(std::ofstream& f, const std::string& path) {
    f.close();
    if (f.fail()) { fprintf(stderr, "Write error on %s\n", path.c_str()); exit(1); }
}

static inline void copy_file_if_exists(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in.good()) return;
    std::ofstream out(dst, std::ios::binary);
    if (in.peek() != std::ifstream::traits_type::eof()) out << in.rdbuf();
    close_or_die(out, dst);
}

// Package all residual tensors into residual.safetensors (sharded if > 3.4 GiB), copy config/tokenizer, write manifest.
static inline void package_residuals_and_manifest(
    const SafetensorsDir& st, const GenericOptions& opt, const ConversionPlan& plan,
    const std::vector<std::string>& tfq_shard_names,
    const std::string& format_name = "TFQ3",
    uint32_t format_version = 3u,
    const std::string& tables_file = "pw_v3_tables.bin",
    const std::string& manifest_file = "tfq3_manifest.json"
) {
    // 1. Write residual safetensors (preserving original dtypes and arbitrary shapes).
    std::vector<std::string> res_files;
    if (!plan.residual_names.empty()) {
        std::vector<std::vector<std::string>> groups;
        std::vector<std::string> cur;
        uint64_t cur_sz = 0;
        for (const auto& name : plan.residual_names) {
            uint64_t len = st.tensors.at(name).len;
            if (!cur.empty() && cur_sz + len > opt.max_shard_bytes) {
                groups.push_back(std::move(cur));
                cur.clear();
                cur_sz = 0;
            }
            cur.push_back(name);
            cur_sz += len;
        }
        if (!cur.empty()) groups.push_back(std::move(cur));

        std::map<std::string, std::string> weight_map;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            char fn[96];
            if (groups.size() == 1) snprintf(fn, sizeof(fn), "residual.safetensors");
            else snprintf(fn, sizeof(fn), "residual-%05zu-of-%05zu.safetensors", gi + 1, groups.size());
            res_files.push_back(fn);

            std::ostringstream hdr;
            hdr << "{\"__metadata__\":{\"format\":\"tfq3\",\"source\":\"TFQ3-generic\"}";
            uint64_t off = 0;
            for (const auto& name : groups[gi]) {
                const auto& ti = st.tensors.at(name);
                weight_map[name] = fn;
                hdr << ",\"" << name << "\":{\"dtype\":\"" << ti.dtype << "\",\"shape\":[";
                for (size_t k = 0; k < ti.shape.size(); ++k) {
                    if (k) hdr << ',';
                    hdr << ti.shape[k];
                }
                hdr << "],\"data_offsets\":[" << off << "," << (off + ti.len) << "]}";
                off += ti.len;
            }
            hdr << "}";
            std::string hb = hdr.str();
            while (hb.size() % 8 != 0) hb.push_back(' ');
            uint64_t hlen = hb.size();

            std::string full = opt.out_dir + "/" + fn;
            FILE* fout = fopen(full.c_str(), "wb");
            if (!fout) { fprintf(stderr, "Cannot write %s\n", full.c_str()); exit(1); }
            fwrite_or_die(&hlen, 8, fout, full);
            fwrite_or_die(hb.data(), hb.size(), fout, full);
            for (const auto& name : groups[gi]) {
                const auto& ti = st.tensors.at(name);
                fwrite_or_die(st.data_ptr(ti), ti.len, fout, full);
            }
            fclose_or_die(fout, full);
        }

        if (groups.size() > 1) {
            std::string idx_path = opt.out_dir + "/model.safetensors.index.json";
            std::ofstream idx(idx_path);
            idx << "{\n  \"metadata\": {\"total_size\": " << plan.residual_bytes << "},\n  \"weight_map\": {\n";
            size_t k = 0;
            for (const auto& kv : weight_map) {
                idx << "    \"" << kv.first << "\": \"" << kv.second << "\"" << (++k < weight_map.size() ? "," : "") << "\n";
            }
            idx << "  }\n}\n";
            close_or_die(idx, idx_path);
        }
    }

    // 2. Copy config & tokenizer/preprocessor files.
    for (const char* fn : {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "tokenizer.model", "vocab.json", "merges.txt", "special_tokens_map.json",
        "added_tokens.json", "chat_template.jinja", "chat_template.json",
        "preprocessor_config.json", "video_preprocessor_config.json"
    }) {
        copy_file_if_exists(opt.model_dir + "/" + fn, opt.out_dir + "/" + fn);
    }

    // 3. Write manifest JSON for cross-platform engines and tooling.
    std::string mf_path = opt.out_dir + "/" + manifest_file;
    std::ofstream mf(mf_path);
    mf << "{\n";
    mf << "  \"format\": \"" << format_name << "\",\n";
    mf << "  \"version\": " << format_version << ",\n";
    mf << "  \"tables\": \"" << tables_file << "\",\n";
    mf << "  \"tfq_shards\": [";
    for (size_t i = 0; i < tfq_shard_names.size(); ++i) {
        if (i) mf << ", ";
        mf << "\"" << tfq_shard_names[i] << "\"";
    }
    mf << "],\n";
    mf << "  \"residual_shards\": [";
    for (size_t i = 0; i < res_files.size(); ++i) {
        if (i) mf << ", ";
        mf << "\"" << res_files[i] << "\"";
    }
    mf << "],\n";
    mf << "  \"tfq_bytes\": " << plan.tfq_bytes << ",\n";
    mf << "  \"residual_bytes\": " << plan.residual_bytes << ",\n";
    mf << "  \"num_quantized_tensors\": " << plan.jobs.size() << ",\n";
    mf << "  \"num_residual_tensors\": " << plan.residual_names.size() << ",\n";
    mf << "  \"tensors\": [\n";
    for (size_t i = 0; i < plan.jobs.size(); ++i) {
        const auto& j = plan.jobs[i];
        mf << "    {\"name\":\"" << j.tfpn_name << "\",\"role\":\"" << role_name(j.role)
           << "\",\"orig_shape\":[" << j.orig_R << "," << j.orig_C << "],\"pad_shape\":["
           << j.pad_R << "," << j.pad_C << "],\"bits\":" << j.bits
           << ",\"snr_db\":" << j.snr_db << "}" << (i + 1 < plan.jobs.size() ? "," : "") << "\n";
    }
    mf << "  ]\n}\n";
    close_or_die(mf, mf_path);
}

static inline void print_plan_summary(const GenericOptions& opt, const ConversionPlan& plan) {
    uint64_t cnt[6] = {};
    uint64_t role_cnt[9] = {};
    for (const auto& j : plan.jobs) {
        if (j.bits <= 5) cnt[j.bits]++;
        role_cnt[(int)j.role]++;
    }
    double tot_gib = (double)(plan.tfq_bytes + plan.residual_bytes) / (double)(1ULL << 30);
    double tfq_gib = (double)plan.tfq_bytes / (double)(1ULL << 30);
    double res_gib = (double)plan.residual_bytes / (double)(1ULL << 30);
    double eff_bpw = plan.total_params ? (double)(plan.tfq_bytes * 8ULL) / (double)plan.total_params : 0.0;
    printf("=== TFQ3 Generic Plan ===\n");
    printf("  source:    %s\n", opt.model_dir.c_str());
    printf("  output:    %s\n", opt.out_dir.c_str());
    if (opt.target_gib > 0.0) {
        printf("  budget:    target %.2f GiB (%s)\n", opt.target_gib, opt.fit_auto ? "fit=auto" : "target-gib");
    } else {
        printf("  profile:   %s (head=%ub)\n", opt.profile.c_str(), opt.head_bits);
    }
    printf("  tensors:   %zu quantized (3b:%llu 4b:%llu 5b:%llu) | %zu residual\n",
           plan.jobs.size(), (unsigned long long)cnt[3], (unsigned long long)cnt[4], (unsigned long long)cnt[5],
           plan.residual_names.size());
    printf("  roles:     attn=%llu mlp=%llu expert=%llu lm_head=%llu vision=%llu drafter=%llu embed=%llu other=%llu\n",
           (unsigned long long)role_cnt[(int)Role::ATTN], (unsigned long long)role_cnt[(int)Role::MLP],
           (unsigned long long)role_cnt[(int)Role::EXPERT], (unsigned long long)role_cnt[(int)Role::LM_HEAD],
           (unsigned long long)role_cnt[(int)Role::VISION], (unsigned long long)role_cnt[(int)Role::DRAFTER],
           (unsigned long long)role_cnt[(int)Role::EMBED], (unsigned long long)role_cnt[(int)Role::OTHER_LINEAR]);
    printf("  footprint: %.3f GiB total (TFQ3 %.3f GiB @ %.3f bpw + residual %.3f GiB) across %zu shard(s)\n",
           tot_gib, tfq_gib, eff_bpw, res_gib, plan.shards.size());
    fflush(stdout);
}

static inline bool parse_generic_args(int argc, char** argv, GenericOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") return false;
        else if (a == "--model" && i + 1 < argc) opt.model_dir = argv[++i];
        else if ((a == "--out" || a == "-o") && i + 1 < argc) opt.out_dir = argv[++i];
        else if (a == "--tables" && i + 1 < argc) opt.tables_path = argv[++i];
        else if (a == "--profile" && i + 1 < argc) {
            opt.profile = argv[++i];
            if (opt.profile == "3b") opt.default_bits = 3u;
            else if (opt.profile == "4b") opt.default_bits = 4u;
            else if (opt.profile == "5b") opt.default_bits = 5u;
        } else if (a == "--bits" && i + 1 < argc) {
            opt.default_bits = (uint32_t)atoi(argv[++i]);
            opt.profile = std::to_string(opt.default_bits) + "b";
        } else if (a == "--head-bits" && i + 1 < argc) opt.head_bits = (uint32_t)atoi(argv[++i]);
        else if (a == "--max-bits" && i + 1 < argc) opt.max_bits = (uint32_t)atoi(argv[++i]);
        else if (a == "--target-gib" && i + 1 < argc) { opt.target_gib = atof(argv[++i]); opt.profile = "auto"; }
        else if (a == "--fit" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "auto") { opt.fit_auto = true; opt.profile = "auto"; }
            else { opt.target_gib = atof(v.c_str()); opt.profile = "auto"; }
        }
        else if (a == "--quantize-vision") opt.quantize_vision = true;
        else if (a == "--quantize-drafter") opt.quantize_drafter = true;
        else if (a == "--quantize-embed") opt.quantize_embed = true;
        else if (a == "--dry-run") opt.dry_run = true;
    }
    return !opt.model_dir.empty() && !opt.out_dir.empty();
}

} // namespace tfq_generic
