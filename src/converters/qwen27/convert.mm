#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "../tfq_tables_io.hpp"
#include "../tfq_xxh64.hpp"

// ============================================================================
// TensorFold TFQ3 (2D-AffineCoset/half4) Qwen3.8-27B Dual-Source GPU Converter
//   - Accepts BOTH BF16 safetensors (Qwen/Qwen3.8-27B, model.language_model.*
//     or model.*) AND MLX 8-bit/64 safetensors (TensorFold/Qwen3.8-27B-MLX-8bit)
//   - Generates deterministic 16KB pw_v3_tables.bin + 4 sharded .tffpw files
//     (<= 3.6 GiB/shard, pair-contiguous gate+up and k+v for fused runtime dispatch)
//   - Profiles:
//       --profile 3b   : All 64L dense linears at 3-bit, lm_head at --head-bits (4b)
//       --profile 3.5b : GDN/Attn + boundary MLP (L0..5, L58..63) at 4b,
//                        middle MLP (L6..57) at 3b, lm_head at --head-bits (4b)
//       --profile 4b   : All 64L dense linears at 4-bit, lm_head at --head-bits (4b)
// ============================================================================

static constexpr uint32_t MAGIC_TFPN = 0x4E504654u; // "TFPN"
static constexpr uint32_t FORMAT_VER = 3u;
static constexpr size_t   ALIGN_16K  = 16384;

static inline size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

using tfq_tables::PwV3TablesFile;

struct NonMoeHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_tensors;
    uint32_t pad0;
    uint64_t off_meta;
    uint64_t off_scales;
    uint64_t off_weights;
    uint64_t total_scale_halfs;
    uint64_t total_weight_words;
    uint64_t pad1;
};
static_assert(sizeof(NonMoeHeader) == 64, "NonMoeHeader must be 64 bytes");

struct NonMoeTensorMeta {
    char     name[80];
    uint32_t R;
    uint32_t C;
    uint32_t b_bits;
    uint32_t sc_off;
    uint64_t word_off;
    uint64_t word_count;
    float    snr_db;
    uint32_t pad[3];
};
static_assert(sizeof(NonMoeTensorMeta) == 128, "NonMoeTensorMeta must be 128 bytes");

struct MatParams { uint32_t R, C, b_bits; float shrink; uint32_t r_off, pad0; };
struct EncParams { uint32_t R, C, b_bits, stream_off, n_streams; };

static inline void load_or_generate_tables(const std::string& path, PwV3TablesFile& tbl, bool explicit_tables = false) {
    tfq_tables::load_or_write_pwv3_tables(path, tbl, explicit_tables);
}

static inline std::vector<uint16_t> build_grid(const uint32_t* lut_u32, int lut_size, int N_sym, int C_shift, int G, int KC) {
    return tfq_tables::build_grid(lut_u32, lut_size, N_sym, C_shift, G, KC);
}

struct ShardCache {
    struct Meta { uint64_t off, len; std::string dtype; };
    struct Entry {
        int fd = -1; uint8_t* ptr = nullptr; size_t len = 0; uint64_t data_base = 0;
        std::map<std::string, Meta> tensors;
    };
    std::string model_dir;
    std::map<std::string, std::string> weight_map;
    std::map<std::string, Entry> shards;

    void index_single_shard(const std::string& fn) {
        Entry& e = get_shard(fn);
        for (const auto& kv : e.tensors) weight_map[kv.first] = fn;
    }

    void init(const std::string& dir) {
        model_dir = dir;
        std::ifstream f(dir + "/model.safetensors.index.json");
        if (f.good()) {
            std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            size_t pos = txt.find("\"weight_map\"");
            if (pos == std::string::npos) { fprintf(stderr, "No weight_map in %s\n", dir.c_str()); exit(1); }
            pos = txt.find('{', pos);
            size_t end = txt.find('}', pos);
            std::string body = txt.substr(pos + 1, end - pos - 1);
            size_t i = 0;
            while (i < body.size()) {
                size_t q1 = body.find('"', i); if (q1 == std::string::npos) break;
                size_t q2 = body.find('"', q1 + 1);
                size_t q3 = body.find('"', q2 + 1);
                size_t q4 = body.find('"', q3 + 1);
                weight_map[body.substr(q1 + 1, q2 - q1 - 1)] = body.substr(q3 + 1, q4 - q3 - 1);
                i = q4 + 1;
            }
            return;
        }
        index_single_shard("model.safetensors");
    }

    bool has_tensor(const std::string& name) const { return weight_map.count(name) > 0; }

    Entry& get_shard(const std::string& fn) {
        auto it = shards.find(fn);
        if (it != shards.end()) return it->second;
        Entry e;
        std::string full = model_dir + "/" + fn;
        e.fd = open(full.c_str(), O_RDONLY);
        if (e.fd < 0) { fprintf(stderr, "Cannot open %s\n", full.c_str()); exit(1); }
        struct stat st{}; fstat(e.fd, &st); e.len = st.st_size;
        e.ptr = (uint8_t*)mmap(nullptr, e.len, PROT_READ, MAP_PRIVATE, e.fd, 0);
        uint64_t hlen = 0; memcpy(&hlen, e.ptr, 8);
        e.data_base = 8 + hlen;
        std::string hdr((const char*)e.ptr + 8, hlen);
        size_t p = 0;
        while (true) {
            size_t dk = hdr.find("\"data_offsets\"", p);
            if (dk == std::string::npos) break;
            size_t open_brace = hdr.rfind('{', dk);
            size_t close_brace = hdr.find('}', dk);
            size_t colon = hdr.rfind(':', open_brace);
            size_t q2 = hdr.rfind('"', colon);
            size_t q1 = hdr.rfind('"', q2 - 1);
            std::string tname = hdr.substr(q1 + 1, q2 - q1 - 1);
            std::string obj = hdr.substr(open_brace, close_brace - open_brace + 1);
            size_t lb = hdr.find('[', dk), cm = hdr.find(',', lb), rb = hdr.find(']', cm);
            uint64_t o0 = std::stoull(hdr.substr(lb + 1, cm - lb - 1));
            uint64_t o1 = std::stoull(hdr.substr(cm + 1, rb - cm - 1));
            std::string dtype;
            size_t dp = obj.find("\"dtype\"");
            if (dp != std::string::npos) {
                size_t d1 = obj.find('"', obj.find(':', dp) + 1);
                size_t d2 = obj.find('"', d1 + 1);
                dtype = obj.substr(d1 + 1, d2 - d1 - 1);
            }
            e.tensors[tname] = {o0, o1 - o0, dtype};
            p = close_brace + 1;
        }
        shards[fn] = std::move(e);
        return shards[fn];
    }

    const uint8_t* get_tensor(const std::string& name, std::string* out_dtype = nullptr, size_t* out_len = nullptr) {
        auto it = weight_map.find(name);
        if (it == weight_map.end()) { fprintf(stderr, "Missing tensor %s\n", name.c_str()); exit(1); }
        Entry& sh = get_shard(it->second);
        auto tit = sh.tensors.find(name);
        if (tit == sh.tensors.end()) { fprintf(stderr, "Missing %s in %s\n", name.c_str(), it->second.c_str()); exit(1); }
        if (out_dtype) *out_dtype = tit->second.dtype;
        if (out_len) *out_len = tit->second.len;
        return sh.ptr + sh.data_base + tit->second.off;
    }

    ~ShardCache() {
        for (auto& kv : shards) { if (kv.second.ptr) munmap(kv.second.ptr, kv.second.len); if (kv.second.fd >= 0) close(kv.second.fd); }
    }
};

struct Task {
    std::string src_module;
    std::string dst_module;
    uint32_t R, C, bits;
};

static void write_shard(
    const std::string& out_path,
    const std::vector<NonMoeTensorMeta>& metas,
    const std::vector<uint16_t>& all_scales,
    const std::vector<uint32_t>& all_weights
) {
    uint64_t off_meta    = sizeof(NonMoeHeader);
    uint64_t off_scales  = align_up(off_meta + metas.size() * sizeof(NonMoeTensorMeta), ALIGN_16K);
    uint64_t off_weights = align_up(off_scales + all_scales.size() * 2ULL, ALIGN_16K);
    NonMoeHeader hdr{};
    hdr.magic = MAGIC_TFPN; hdr.version = FORMAT_VER; hdr.num_tensors = (uint32_t)metas.size();
    hdr.pad0 = tfq_xxh64::FLAG_XXH64_PRESENT;
    hdr.off_meta = off_meta; hdr.off_scales = off_scales; hdr.off_weights = off_weights;
    hdr.total_scale_halfs = all_scales.size(); hdr.total_weight_words = all_weights.size();
    hdr.pad1 = tfq_xxh64::hash_sections3(
        metas.data(), metas.size() * sizeof(NonMoeTensorMeta),
        all_scales.data(), all_scales.size() * 2ULL,
        all_weights.data(), all_weights.size() * 4ULL
    );

    auto write_all = [&](FILE* f, const void* p, size_t n) {
        if (n && fwrite(p, 1, n, f) != n) { fprintf(stderr, "Write error on %s\n", out_path.c_str()); exit(1); }
    };
    FILE* fout = fopen(out_path.c_str(), "wb");
    if (!fout) { fprintf(stderr, "Cannot open %s\n", out_path.c_str()); exit(1); }
    write_all(fout, &hdr, sizeof(hdr));
    write_all(fout, metas.data(), metas.size() * sizeof(NonMoeTensorMeta));
    size_t cur = off_meta + metas.size() * sizeof(NonMoeTensorMeta);
    std::vector<uint8_t> zeros(ALIGN_16K, 0);
    if (off_scales > cur) write_all(fout, zeros.data(), off_scales - cur);
    write_all(fout, all_scales.data(), all_scales.size() * 2ULL);
    cur = off_scales + all_scales.size() * 2ULL;
    if (off_weights > cur) write_all(fout, zeros.data(), off_weights - cur);
    write_all(fout, all_weights.data(), all_weights.size() * 4ULL);
    size_t end_pos = off_weights + all_weights.size() * 4ULL;
    size_t pad_end = align_up(end_pos, ALIGN_16K) - end_pos;
    if (pad_end) write_all(fout, zeros.data(), pad_end);
    if (fclose(fout) != 0) { fprintf(stderr, "Write error on %s\n", out_path.c_str()); exit(1); }
}

static std::string default_metallib(const char* name) {
    uint32_t sz = 0;
    _NSGetExecutablePath(nullptr, &sz);
    std::string exe(sz, '\0');
    if (_NSGetExecutablePath(&exe[0], &sz) == 0) {
        size_t sl = exe.find_last_of('/');
        if (sl != std::string::npos) return exe.substr(0, sl + 1) + name;
    }
    return name;
}

int main(int argc, char** argv) {
    std::string model_dir   = "";
    std::string out_dir     = "";
    std::string profile     = "4b"; // 4b (recommended) | 3.5b | 3b
    std::string metallib    = default_metallib("tfq3-qwen27-convert.metallib");
    std::string tables_path = "";
    uint32_t head_bits      = 4u;
    int num_layers          = 64;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { model_dir.clear(); break; }
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--metallib" && i + 1 < argc) metallib = argv[++i];
        else if (a == "--tables" && i + 1 < argc) tables_path = argv[++i];
        else if (a == "--head-bits" && i + 1 < argc) head_bits = (uint32_t)atoi(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) num_layers = atoi(argv[++i]);
    }
    if (model_dir.empty() || out_dir.empty()) {
        fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--profile 4b|3.5b|3b] [--head-bits 0|4|5] [--layers 64] [--tables <path>] [--metallib <path>]\n", argv[0]);
        return 1;
    }
    mkdir(out_dir.c_str(), 0755);
    const bool explicit_tables = !tables_path.empty();
    std::string dst_tables = out_dir + "/pw_v3_tables.bin";
    if (!explicit_tables) tables_path = dst_tables;

    PwV3TablesFile tbl;
    load_or_generate_tables(tables_path, tbl, explicit_tables);
    if (tables_path != dst_tables) {
        FILE* fo = fopen(dst_tables.c_str(), "wb");
        if (!fo) { fprintf(stderr, "Cannot write %s\n", dst_tables.c_str()); return 1; }
        fwrite(&tbl, 1, sizeof(tbl), fo);
        fclose(fo);
    }

    auto grid3 = build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
    auto grid4 = build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
    auto grid5 = build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

    ShardCache cache;
    cache.init(model_dir);

    // Auto-detect source namespace: HF BF16 (model.language_model.), MLX (language_model.model.), or plain (model.)
    std::string layer_prefix, head_src;
    if (cache.has_tensor("model.language_model.layers.0.mlp.gate_proj.weight")) {
        layer_prefix = "model.language_model.";
    } else if (cache.has_tensor("language_model.model.layers.0.mlp.gate_proj.weight")) {
        layer_prefix = "language_model.model.";
    } else if (cache.has_tensor("model.layers.0.mlp.gate_proj.weight")) {
        layer_prefix = "model.";
    } else {
        fprintf(stderr, "Could not detect Qwen3.8-27B layer prefix in %s\n", model_dir.c_str());
        return 1;
    }
    if (cache.has_tensor("language_model.lm_head.weight")) head_src = "language_model.lm_head";
    else if (cache.has_tensor("lm_head.weight")) head_src = "lm_head";
    else { fprintf(stderr, "Could not detect lm_head.weight in %s\n", model_dir.c_str()); return 1; }

    std::string sample_dt;
    cache.get_tensor(layer_prefix + "layers.0.mlp.gate_proj.weight", &sample_dt);
    bool src_is_bf16 = (sample_dt == "BF16") && !cache.has_tensor(layer_prefix + "layers.0.mlp.gate_proj.scales");
    printf("Source: %s | prefix='%s' | head='%s' | format=%s | profile=%s | head_bits=%u\n",
           model_dir.c_str(), layer_prefix.c_str(), head_src.c_str(), src_is_bf16 ? "BF16" : "MLX-8bit/64", profile.c_str(), head_bits);
    fflush(stdout);

    auto bits_for = [&](int l, bool is_mlp) -> uint32_t {
        if (profile == "3b") return 3u;
        if (profile == "4b") return 4u;
        // 3.5b hybrid: 4b GDN/Attn + 4b boundary MLP (L0..5, L58..63), 3b middle MLP (L6..57)
        if (!is_mlp) return 4u;
        if (l < 6 || l >= num_layers - 6) return 4u;
        return 3u;
    };

    // Split into 4 shards of 16 layers each (<= 3.6 GiB/shard, pair-contiguous gate+up and k+v inside each shard).
    const int n_shards = 4;
    const int l_per_shard = (num_layers + n_shards - 1) / n_shards;
    std::vector<std::vector<Task>> shard_tasks(n_shards);
    for (int l = 0; l < num_layers; ++l) {
        int si = std::min(n_shards - 1, l / l_per_shard);
        auto add = [&](const std::string& leaf, uint32_t R, uint32_t C, uint32_t b) {
            char sp[192], dp[192];
            snprintf(sp, sizeof(sp), "%slayers.%d.%s", layer_prefix.c_str(), l, leaf.c_str());
            snprintf(dp, sizeof(dp), "model.layers.%d.%s", l, leaf.c_str());
            shard_tasks[si].push_back({sp, dp, R, C, b});
        };
        uint32_t mb = bits_for(l, false);
        uint32_t fb = bits_for(l, true);
        if ((l + 1) % 4 != 0) {
            add("linear_attn.in_proj_qkv", 10240, 5120, mb);
            add("linear_attn.in_proj_z",   6144,  5120, mb);
            add("linear_attn.out_proj",    5120,  6144, mb);
        } else {
            add("self_attn.q_proj", 12288, 5120, mb);
            add("self_attn.k_proj", 1024,  5120, mb);
            add("self_attn.v_proj", 1024,  5120, mb);
            add("self_attn.o_proj", 5120,  6144, mb);
        }
        add("mlp.gate_proj", 17408, 5120,  fb);
        add("mlp.up_proj",   17408, 5120,  fb);
        add("mlp.down_proj", 5120,  17408, fb);
    }
    if (head_bits >= 3 && head_bits <= 5) {
        shard_tasks[n_shards - 1].push_back({head_src, "lm_head", 248320, 5120, head_bits});
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]] error:&err];
        if (!lib) { fprintf(stderr, "metallib err: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        auto get_pso = [&](const char* name) {
            NSError* e = nil;
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:name]] error:&e];
            if (!p) { fprintf(stderr, "pso %s err: %s\n", name, [[e localizedDescription] UTF8String]); exit(1); }
            return p;
        };

        auto pso_bf16_row = get_pso("bf16_row_fwht");
        auto pso_q8_row   = get_pso("q8_64_dequant_row_fwht");
        auto pso_col_fwht = get_pso("col_fwht_and_ss");
        auto pso_pack_nax = get_pso("finalize_scales_and_pack_nax");
        id<MTLComputePipelineState> pso_enc[6] = {};
        pso_enc[3] = get_pso("prism_encode_v3_b3");
        pso_enc[4] = get_pso("prism_encode_v3_b4");
        pso_enc[5] = get_pso("prism_encode_v3_b5");
        auto pso_snr = get_pso("prism_verify_v3_snr");

        id<MTLBuffer> buf_lut[6] = {}, buf_c64[6] = {}, buf_grid[6] = {};
        buf_lut[3]  = [dev newBufferWithBytes:tbl.lut3_u32 length:sizeof(tbl.lut3_u32) options:MTLResourceStorageModeShared];
        buf_lut[4]  = [dev newBufferWithBytes:tbl.lut4_u32 length:sizeof(tbl.lut4_u32) options:MTLResourceStorageModeShared];
        buf_lut[5]  = [dev newBufferWithBytes:tbl.lut5_u32 length:sizeof(tbl.lut5_u32) options:MTLResourceStorageModeShared];
        buf_c64[3]  = [dev newBufferWithBytes:tbl.c64_3    length:sizeof(tbl.c64_3)    options:MTLResourceStorageModeShared];
        buf_c64[4]  = [dev newBufferWithBytes:tbl.c64_4    length:sizeof(tbl.c64_4)    options:MTLResourceStorageModeShared];
        buf_c64[5]  = [dev newBufferWithBytes:tbl.c64_5    length:sizeof(tbl.c64_5)    options:MTLResourceStorageModeShared];
        buf_grid[3] = [dev newBufferWithBytes:grid3.data() length:grid3.size() * 2 options:MTLResourceStorageModeShared];
        buf_grid[4] = [dev newBufferWithBytes:grid4.data() length:grid4.size() * 2 options:MTLResourceStorageModeShared];
        buf_grid[5] = [dev newBufferWithBytes:grid5.data() length:grid5.size() * 2 options:MTLResourceStorageModeShared];

        const size_t MAX_E = 17408ULL * 5120ULL; // 89.1M elems (fits gate/up/down and 16384x5120 lm_head chunk)
        const size_t HIST_SLOTS = 2097152ULL;    // 2M pair-slots * 256 * 4B = 2.0 GiB
        id<MTLBuffer> buf_in_w   = [dev newBufferWithLength:MAX_E * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_q8_s   = [dev newBufferWithLength:(MAX_E / 64) * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_q8_b   = [dev newBufferWithLength:(MAX_E / 64) * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_w_tmp  = [dev newBufferWithLength:MAX_E * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_w_rot  = [dev newBufferWithLength:MAX_E * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_blk_ss = [dev newBufferWithLength:16384ULL * 128ULL * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_str    = [dev newBufferWithLength:MAX_E * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_hist   = [dev newBufferWithLength:HIST_SLOTS * 256ULL * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_pack   = [dev newBufferWithLength:MAX_E options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sc     = [dev newBufferWithLength:16384 * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_se     = [dev newBufferWithLength:2048ULL * 2ULL * 4ULL options:MTLResourceStorageModeShared];

        auto t0_all = std::chrono::high_resolution_clock::now();
        double snr_sum[6] = {}; int snr_cnt[6] = {};
        uint64_t total_disk_bytes = 16384;

        for (int si = 0; si < n_shards; ++si) {
            const auto& tasks = shard_tasks[si];
            if (tasks.empty()) continue;
            std::vector<NonMoeTensorMeta> metas(tasks.size());
            uint64_t total_sc = 0, total_wd = 0;
            for (size_t i = 0; i < tasks.size(); ++i) {
                memset(&metas[i], 0, sizeof(NonMoeTensorMeta));
                strncpy(metas[i].name, tasks[i].dst_module.c_str(), 79);
                metas[i].R = tasks[i].R; metas[i].C = tasks[i].C; metas[i].b_bits = tasks[i].bits;
                uint64_t sc_cnt = (uint64_t)(tasks[i].R / 128) * (tasks[i].C / 128);
                uint64_t wd_cnt = (uint64_t)(tasks[i].R / 16) * (tasks[i].C / 128) * (tasks[i].bits * 2u) * 32u;
                metas[i].sc_off = (uint32_t)total_sc; total_sc += sc_cnt;
                metas[i].word_off = total_wd; metas[i].word_count = wd_cnt; total_wd += wd_cnt;
            }
            std::vector<uint16_t> all_scales(total_sc);
            std::vector<uint32_t> all_weights(total_wd);

            printf("=== Shard %d/%d | %zu tensors | %.3f GiB ===\n",
                   si + 1, n_shards, tasks.size(), (double)(total_wd * 4ULL + total_sc * 2ULL) / (double)(1ULL << 30));
            fflush(stdout);

            for (size_t ti = 0; ti < tasks.size(); ++ti) {
                const auto& t = tasks[ti];
                std::string dt;
                const uint8_t* w_ptr = cache.get_tensor(t.src_module + ".weight", &dt);
                bool is_bf16 = (dt == "BF16") && !cache.has_tensor(t.src_module + ".scales");
                const uint8_t* s_ptr = is_bf16 ? nullptr : cache.get_tensor(t.src_module + ".scales");
                const uint8_t* b_ptr = is_bf16 ? nullptr : cache.get_tensor(t.src_module + ".biases");
                uint32_t chunk_R = (t.R > 17408u) ? 16384u : t.R;
                double tot_sig = 0.0, tot_err = 0.0;

                for (uint32_t r0 = 0; r0 < t.R; r0 += chunk_R) {
                    uint32_t sr = std::min(chunk_R, t.R - r0);
                    if (is_bf16) {
                        const uint16_t* w_bf16 = reinterpret_cast<const uint16_t*>(w_ptr + (size_t)r0 * t.C * 2);
                        tfq_common::check_finite_bf16(w_bf16, sr, t.C, t.src_module + ".weight", r0);
                        memcpy([buf_in_w contents], w_bf16, (size_t)sr * t.C * 2);
                    } else {
                        const uint16_t* s_bf16 = reinterpret_cast<const uint16_t*>(s_ptr + (size_t)r0 * (t.C / 64) * 2);
                        const uint16_t* b_bf16 = reinterpret_cast<const uint16_t*>(b_ptr + (size_t)r0 * (t.C / 64) * 2);
                        tfq_common::check_finite_bf16(s_bf16, sr, t.C / 64, t.src_module + ".scales", r0);
                        tfq_common::check_finite_bf16(b_bf16, sr, t.C / 64, t.src_module + ".biases", r0);
                        memcpy([buf_in_w contents], w_ptr + (size_t)r0 * t.C, (size_t)sr * t.C);
                        memcpy([buf_q8_s contents], s_bf16, (size_t)sr * (t.C / 64) * 2);
                        memcpy([buf_q8_b contents], b_bf16, (size_t)sr * (t.C / 64) * 2);
                    }
                    size_t n_blks = (size_t)(sr / 128) * (t.C / 128);

                    MatParams mp = {sr, t.C, t.bits, tfq_common::qwen27_shrink(t.bits), r0, 0u};
                    uint32_t total_streams = (sr / 16) * 32;
                    uint32_t n_pairs = t.C / 4;
                    uint32_t batch_sz = (uint32_t)std::max<size_t>(1, HIST_SLOTS / n_pairs);

                    @autoreleasepool {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    if (is_bf16) {
                        [enc setComputePipelineState:pso_bf16_row];
                        [enc setBuffer:buf_in_w offset:0 atIndex:0];
                        [enc setBuffer:buf_w_tmp offset:0 atIndex:1];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                    } else {
                        [enc setComputePipelineState:pso_q8_row];
                        [enc setBuffer:buf_in_w offset:0 atIndex:0];
                        [enc setBuffer:buf_q8_s offset:0 atIndex:1];
                        [enc setBuffer:buf_q8_b offset:0 atIndex:2];
                        [enc setBuffer:buf_w_tmp offset:0 atIndex:3];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                    }
                    [enc dispatchThreadgroups:MTLSizeMake(t.C / 128, sr, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_col_fwht];
                    [enc setBuffer:buf_w_tmp offset:0 atIndex:0]; [enc setBuffer:buf_w_rot offset:0 atIndex:1];
                    [enc setBuffer:buf_blk_ss offset:0 atIndex:2]; [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(t.C, sr / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_pack_nax];
                    [enc setBuffer:buf_w_rot offset:0 atIndex:0]; [enc setBuffer:buf_blk_ss offset:0 atIndex:1];
                    [enc setBuffer:buf_sc offset:0 atIndex:2]; [enc setBuffer:buf_str offset:0 atIndex:3];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(sr / 16, t.C / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_enc[t.bits]];
                    for (uint32_t off = 0; off < total_streams; off += batch_sz) {
                        uint32_t cnt = std::min(batch_sz, total_streams - off);
                        EncParams ep = {sr, t.C, t.bits, off, cnt};
                        [enc setBuffer:buf_str offset:0 atIndex:0];
                        [enc setBuffer:buf_lut[t.bits] offset:0 atIndex:1];
                        [enc setBuffer:buf_c64[t.bits] offset:0 atIndex:2];
                        [enc setBuffer:buf_grid[t.bits] offset:0 atIndex:3];
                        [enc setBuffer:buf_hist offset:0 atIndex:4];
                        [enc setBuffer:buf_pack offset:0 atIndex:5];
                        [enc setBytes:&ep length:sizeof(ep) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }

                    [enc setComputePipelineState:pso_snr];
                    [enc setBuffer:buf_pack offset:0 atIndex:0]; [enc setBuffer:buf_sc offset:0 atIndex:1];
                    [enc setBuffer:buf_lut[t.bits] offset:0 atIndex:2]; [enc setBuffer:buf_c64[t.bits] offset:0 atIndex:3];
                    [enc setBuffer:buf_w_rot offset:0 atIndex:4]; [enc setBuffer:buf_se offset:0 atIndex:5];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(sr / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }
                    const float* se = (const float*)[buf_se contents];
                    for (uint32_t si_s = 0; si_s < sr / 16u; ++si_s) {
                        tot_sig += (double)se[si_s * 2u + 0u];
                        tot_err += (double)se[si_s * 2u + 1u];
                    }
                    size_t sc_off_chunk = (size_t)(r0 / 128) * (t.C / 128);
                    size_t wd_off_chunk = (size_t)(r0 / 16) * (t.C / 128) * (t.bits * 2u) * 32u;
                    size_t wd_cnt_chunk = (size_t)(sr / 16) * (t.C / 128) * (t.bits * 2u) * 32u;
                    memcpy(all_scales.data() + metas[ti].sc_off + sc_off_chunk, [buf_sc contents], n_blks * 2);
                    memcpy(all_weights.data() + metas[ti].word_off + wd_off_chunk, [buf_pack contents], wd_cnt_chunk * 4);
                }
                float snr = (float)(10.0 * log10(tot_sig / std::max(1e-18, tot_err)));
                metas[ti].snr_db = snr;
                snr_sum[t.bits] += snr; snr_cnt[t.bits]++;
                if ((ti + 1) % 30 == 0 || ti + 1 == tasks.size()) {
                    double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0_all).count();
                    printf("  [S%d %3zu/%zu] %-48s [%6u,%5u] %ub SNR=%5.2f dB (%.1fs)\n",
                           si + 1, ti + 1, tasks.size(), t.dst_module.c_str(), t.R, t.C, t.bits, snr, el);
                    fflush(stdout);
                }
            }

            char fn[128];
            snprintf(fn, sizeof(fn), "%s/qwen27_tfq3-%05d-of-%05d.tffpw", out_dir.c_str(), si + 1, n_shards);
            write_shard(fn, metas, all_scales, all_weights);
            struct stat st{}; stat(fn, &st);
            total_disk_bytes += (uint64_t)st.st_size;
            printf("Wrote %s (%.3f GiB)\n", fn, (double)st.st_size / (double)(1ULL << 30));
            fflush(stdout);
        }

        double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0_all).count();
        printf("DONE profile=%s in %.1fs | TFQ3 bytes=%.3f GiB | Mean SNR: 3b=%.2f dB (n=%d), 4b=%.2f dB (n=%d), 5b=%.2f dB (n=%d)\n",
               profile.c_str(), el, (double)total_disk_bytes / (double)(1ULL << 30),
               snr_cnt[3] ? snr_sum[3] / snr_cnt[3] : 0.0, snr_cnt[3],
               snr_cnt[4] ? snr_sum[4] / snr_cnt[4] : 0.0, snr_cnt[4],
               snr_cnt[5] ? snr_sum[5] / snr_cnt[5] : 0.0, snr_cnt[5]);
    }
    return 0;
}
