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
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "../moe_heat_rank.hpp"
#include "../tfq_tables_io.hpp"
#include "../tfq_xxh64.hpp"

// ============================================================================
// TFF-PrismWave v3 (2D-AffineCoset/half4) Production GPU Converter (Apple M5)
//   - Generates deterministic 16KB pw_v3_tables.bin (exact fp16 LUTs & c64)
//   - --mode heat543 / heat43 / heat443 / uniform4 / uniform3 : FP8 or BF16 MoE -> layer_XX.tffpw (version=3)
//     (the routed experts' dtype is read from the checkpoint: F8_E4M3 with weight_scale_inv, or BF16)
//   - --mode nonmoe : Converts big 8-bit/64 non-MoE + L07-44 shared -> nonmoe.tffpw ("TFPN", v3)
// ============================================================================

static constexpr uint32_t MAGIC_TFPW = 0x57504654u; // "TFPW"
static constexpr uint32_t MAGIC_TFPN = 0x4E504654u; // "TFPN"
static constexpr uint32_t FORMAT_VER = 3u;
static constexpr size_t   ALIGN_16K  = 16384;

static inline size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

using tfq_tables::PwV3TablesFile;

struct PrismLayerHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_idx;
    uint32_t num_experts;
    uint32_t num_tensors;
    uint32_t hidden_size;
    uint32_t moe_inter;
    uint32_t reserved0;
    uint64_t raw_fp8_bytes;
    uint64_t stream_words;
    uint64_t scale_halfs;
    uint64_t off_meta;
    uint64_t off_scales;
    uint64_t off_weights;
    uint64_t reserved1[4];
    uint8_t  meta_tag[32];
};
static_assert(sizeof(PrismLayerHeader) == 144, "Header must be 144 bytes");

struct PrismExpertMeta {
    uint32_t b_bits;
    uint32_t heat_rank;
    uint64_t gate_word_off;
    uint64_t up_word_off;
    uint64_t down_word_off;
    uint32_t gate_sc_off;
    uint32_t up_sc_off;
    uint32_t down_sc_off;
    float    gate_snr_db;
    float    up_snr_db;
    float    down_snr_db;
    uint32_t reserved[2];
};
static_assert(sizeof(PrismExpertMeta) == 64, "ExpertMeta must be 64 bytes");

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

struct MatParams { uint32_t R, C, b_bits; float shrink; uint32_t r_off, pad0; };  // r_off: the chunk's first absolute row (TFQ3 row signs are absolute)
struct EncParams { uint32_t R, C, b_bits, stream_off, n_streams; };

static inline void load_or_generate_tables(const std::string& path, PwV3TablesFile& tbl, bool explicit_tables = false) {
    tfq_tables::load_or_write_pwv3_tables(path, tbl, explicit_tables);
}

static inline std::vector<uint16_t> build_grid(const uint32_t* lut_u32, int lut_size, int N_sym, int C_shift, int G, int KC) {
    return tfq_tables::build_grid(lut_u32, lut_size, N_sym, C_shift, G, KC);
}

// Fast mmap safetensors reader with cached open shards (supports multi-shard index and single model.safetensors)
struct ShardCache {
    struct Meta { uint64_t off, len; std::string dtype; int R, C; };
    struct Entry {
        int fd = -1; uint8_t* ptr = nullptr; size_t len = 0; uint64_t data_base = 0;
        std::map<std::string, Meta> tensors;
    };
    std::string model_dir;
    std::map<std::string, std::string> weight_map;
    std::map<std::string, Entry> shards;

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
        Entry& e = get_shard("model.safetensors");
        for (const auto& kv : e.tensors) weight_map[kv.first] = "model.safetensors";
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
                size_t d1 = obj.find('"', obj.find(':', dp) + 1), d2 = obj.find('"', d1 + 1);
                dtype = obj.substr(d1 + 1, d2 - d1 - 1);
            }
            int R = 0, C = 0;
            size_t sp = obj.find("\"shape\"");
            if (sp != std::string::npos) {
                size_t slb = obj.find('[', sp), srb = obj.find(']', slb);
                std::string sbody = obj.substr(slb + 1, srb - slb - 1);
                size_t scm = sbody.find(',');
                if (scm != std::string::npos) { R = std::stoi(sbody.substr(0, scm)); C = std::stoi(sbody.substr(scm + 1)); }
            }
            e.tensors[tname] = {o0, o1 - o0, dtype, R, C};
            p = close_brace + 1;
        }
        shards[fn] = std::move(e);
        return shards[fn];
    }

    const uint8_t* get_tensor(const std::string& name, std::string* out_dtype = nullptr, size_t* out_len = nullptr, int* out_R = nullptr, int* out_C = nullptr) {
        auto it = weight_map.find(name);
        if (it == weight_map.end()) { fprintf(stderr, "Missing tensor %s\n", name.c_str()); exit(1); }
        Entry& sh = get_shard(it->second);
        auto tit = sh.tensors.find(name);
        if (tit == sh.tensors.end()) { fprintf(stderr, "Missing %s in %s\n", name.c_str(), it->second.c_str()); exit(1); }
        if (out_dtype) *out_dtype = tit->second.dtype;
        if (out_len) *out_len = tit->second.len;
        if (out_R) *out_R = tit->second.R;
        if (out_C) *out_C = tit->second.C;
        return sh.ptr + sh.data_base + tit->second.off;
    }

    void evict_all() {
        for (auto& kv : shards) { if (kv.second.ptr) munmap(kv.second.ptr, kv.second.len); if (kv.second.fd >= 0) close(kv.second.fd); }
        shards.clear();
    }
    ~ShardCache() { evict_all(); }
};

static std::map<int, std::vector<int>> load_heat_ranks(const std::string& trace_path, int n_exp) {
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
            // Also accept raw routing trace rows: "<tok_idx> <layer_idx> <e0> <e1> ... <ek>"
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
        std::vector<int> order(n_exp); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return (kv.second[a] != kv.second[b]) ? (kv.second[a] > kv.second[b]) : (a < b);
        });
        std::vector<int> r(n_exp);
        for (int idx = 0; idx < n_exp; ++idx) r[order[idx]] = idx;
        ranks[kv.first] = std::move(r);
    }
    return ranks;
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
    std::string trace_path  = "";
    std::string mode        = "heat43"; // heat543 | heat43 | heat443 | nonmoe
    std::string metallib    = default_metallib("tfq3-glm-convert.metallib");
    std::string tables_path = "";
    int l_start = 3, l_end = 44;
    bool skip_existing = false;
    bool shared_pw4 = true;
    bool fast4b = false;
    bool head_pw5 = false;
    bool all_pw5 = false;
    bool skip_head = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { model_dir.clear(); break; }
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--trace" && i + 1 < argc) trace_path = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--metallib" && i + 1 < argc) metallib = argv[++i];
        else if (a == "--tables" && i + 1 < argc) tables_path = argv[++i];
        else if (a == "--layers" && i + 2 < argc) { l_start = atoi(argv[++i]); l_end = atoi(argv[++i]); }
        else if (a == "--skip-existing") skip_existing = true;
        else if (a == "--shared-pw5") shared_pw4 = false;
        else if (a == "--fast4b") fast4b = true;
        else if (a == "--head-pw5") head_pw5 = true;
        else if (a == "--all-pw5") { all_pw5 = true; shared_pw4 = false; head_pw5 = true; }
        else if (a == "--skip-head") skip_head = true;
    }
    if (model_dir.empty() || out_dir.empty()) {
        fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--mode heat543|heat43|heat443|uniform4|uniform3|nonmoe] [--layers S E] [--trace <file>] [--fast4b] [--shared-pw5] [--head-pw5] [--all-pw5] [--skip-head] [--skip-existing] [--tables <path>] [--metallib <path>]\n", argv[0]);
        return 1;
    }
    mkdir(out_dir.c_str(), 0755);
    const bool explicit_tables = !tables_path.empty();
    if (!explicit_tables) tables_path = out_dir + "/pw_v3_tables.bin";

    PwV3TablesFile tbl;
    load_or_generate_tables(tables_path, tbl, explicit_tables);

    auto grid3 = build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
    auto grid4 = build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
    auto grid5 = build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

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

        auto pso_fp8_row  = get_pso("fp8_dequant_row_fwht");
        auto pso_bf16_row = get_pso("bf16_row_fwht");
        auto pso_q8_row   = get_pso("q8_64_dequant_row_fwht");
        auto pso_col_fwht = get_pso("col_fwht_and_ss");
        auto pso_pack_nax = get_pso("finalize_scales_and_pack_nax");
        id<MTLComputePipelineState> pso_enc[6] = {};
        pso_enc[3] = get_pso("prism_encode_v3_b3");
        pso_enc[4] = get_pso("prism_encode_v3_b4");
        pso_enc[5] = get_pso("prism_encode_v3_b5");
        auto pso_snr = get_pso("prism_verify_v3_snr");

        float fp8_lut[256];
        for (int i = 0; i < 256; ++i) {
            int s = (i >> 7) & 1, e = (i >> 3) & 15, m = i & 7;
            float v = (e == 0) ? (m / 8.0f) * powf(2.0f, -6.0f) : (e == 15 && m == 7) ? 0.0f : (1.0f + m / 8.0f) * powf(2.0f, (float)(e - 7));
            fp8_lut[i] = s ? -v : v;
        }
        id<MTLBuffer> buf_fp8_lut = [dev newBufferWithBytes:fp8_lut length:1024 options:MTLResourceStorageModeShared];
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

        ShardCache cache;
        cache.init(model_dir);

        // ====================================================================
        // MODE A: NON-MOE CONVERSION -> nonmoe.tffpw ("TFPN", v3, ~4.21 GB)
        // ====================================================================
        if (mode == "nonmoe") {
            struct Task { std::string name; uint32_t R, C, bits; };
            std::vector<Task> tasks;
            const uint32_t nb_qkv  = all_pw5 ? 5u : 4u;
            const uint32_t nb_proj = (fast4b && !all_pw5) ? 4u : 5u;
            const uint32_t nb_head = (fast4b && !head_pw5 && !all_pw5) ? 4u : 5u;
            for (int l = 0; l < 45; ++l) {
                char p[160];
                snprintf(p, sizeof(p), "model.language_model.layers.%d.self_attn.q_proj", l);
                if (cache.has_tensor(std::string(p) + ".weight")) {
                    // KDA: q_proj, k_proj, v_proj CONTIGUOUS at nb_qkv (5b in --all-pw5, else 4b), o_proj at nb_proj!
                    for (const char* s : {"q_proj", "k_proj", "v_proj"}) {
                        snprintf(p, sizeof(p), "model.language_model.layers.%d.self_attn.%s", l, s);
                        tasks.push_back({p, 8192, 4096, nb_qkv});
                    }
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.self_attn.o_proj", l);
                    tasks.push_back({p, 4096, 8192, nb_proj});
                } else {
                    // MLA: q_b_proj [16384, 1536] + indexer.wq_b [4096, 1536] CONTIGUOUS (merged 20480x1536!) + o_proj!
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.self_attn.q_b_proj", l);
                    if (cache.has_tensor(std::string(p) + ".weight")) tasks.push_back({p, 16384, 1536, nb_proj});
                    if (fast4b || all_pw5) {
                        snprintf(p, sizeof(p), "model.language_model.layers.%d.self_attn.indexer.wq_b", l);
                        if (cache.has_tensor(std::string(p) + ".weight")) tasks.push_back({p, 4096, 1536, nb_proj});
                    }
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.self_attn.o_proj", l);
                    if (cache.has_tensor(std::string(p) + ".weight")) tasks.push_back({p, 4096, 16384, nb_proj});
                }
                if (l < 3) {
                    // Dense MLP: gate_proj, up_proj CONTIGUOUS, down_proj!
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.gate_proj", l); tasks.push_back({p, 12288, 4096, nb_proj});
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.up_proj",   l); tasks.push_back({p, 12288, 4096, nb_proj});
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.down_proj", l); tasks.push_back({p, 4096, 12288, nb_proj});
                } else if ((fast4b || all_pw5) && l >= 3 && l < 7) {
                    // Boundary Shared experts L03-L06 at PW5 (29.35 dB, fused in dec_*_543!)
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.gate_proj", l); tasks.push_back({p, 2048, 4096, 5});
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.up_proj",   l); tasks.push_back({p, 2048, 4096, 5});
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.down_proj", l); tasks.push_back({p, 4096, 2048, 5});
                } else if (l >= 7) {
                    // Deep Shared experts L07-L44: PW4 (23.35 dB -> ZERO 5b code in *_43!) or PW5
                    uint32_t sb = shared_pw4 ? 4u : 5u;
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.gate_proj", l); tasks.push_back({p, 2048, 4096, sb});
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.up_proj",   l); tasks.push_back({p, 2048, 4096, sb});
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.down_proj", l); tasks.push_back({p, 4096, 2048, sb});
                }
            }
            if (!skip_head) tasks.push_back({"lm_head", 154880, 4096, nb_head});

            // Max chunk: 16384 x 4096 = 67.1M elems (lm_head chunked by 8192 rows!)
            const size_t MAX_E = 16384ULL * 4096ULL;
            const size_t HIST_SLOTS = 2097152ULL; // 2M pair-slots * 512 * 4B = 4.0 GB!
            id<MTLBuffer> buf_q8_w   = [dev newBufferWithLength:MAX_E options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_q8_s   = [dev newBufferWithLength:(MAX_E / 64) * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_q8_b   = [dev newBufferWithLength:(MAX_E / 64) * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_w_tmp  = [dev newBufferWithLength:MAX_E * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> buf_w_rot  = [dev newBufferWithLength:MAX_E * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> buf_blk_ss = [dev newBufferWithLength:8192ULL * 128ULL * 4ULL options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_str    = [dev newBufferWithLength:MAX_E * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> buf_hist   = [dev newBufferWithLength:HIST_SLOTS * 512ULL * 4ULL options:MTLResourceStorageModePrivate];
            id<MTLBuffer> buf_pack   = [dev newBufferWithLength:MAX_E options:MTLResourceStorageModeShared]; // max 5b < 1B/elem
            id<MTLBuffer> buf_sc     = [dev newBufferWithLength:8192 * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_se     = [dev newBufferWithLength:2048ULL * 2ULL * 4ULL options:MTLResourceStorageModeShared];

            std::vector<NonMoeTensorMeta> metas(tasks.size());
            uint64_t total_sc = 0, total_wd = 0;
            for (size_t i = 0; i < tasks.size(); ++i) {
                memset(&metas[i], 0, sizeof(NonMoeTensorMeta));
                strncpy(metas[i].name, tasks[i].name.c_str(), 79);
                metas[i].R = tasks[i].R; metas[i].C = tasks[i].C; metas[i].b_bits = tasks[i].bits;
                uint64_t sc_cnt = (uint64_t)(tasks[i].R / 128) * (tasks[i].C / 128);
                uint64_t wd_cnt = (uint64_t)(tasks[i].R / 16) * (tasks[i].C / 128) * (tasks[i].bits * 2u) * 32u;
                metas[i].sc_off = (uint32_t)total_sc; total_sc += sc_cnt;
                metas[i].word_off = total_wd; metas[i].word_count = wd_cnt; total_wd += wd_cnt;
            }
            std::vector<uint16_t> all_scales(total_sc);
            std::vector<uint32_t> all_weights(total_wd);

            printf("====================================================================================================\n");
            printf("TFF-PrismWave v3 NON-MOE CONVERTER | %zu tensors | total=%.3f GB\n",
                   tasks.size(), (double)(total_wd * 4ULL + total_sc * 2ULL) / 1e9);
            printf("====================================================================================================\n");
            fflush(stdout);

            auto t0 = std::chrono::high_resolution_clock::now();
            double snr_sum[6] = {}; int snr_cnt[6] = {};
            for (size_t ti = 0; ti < tasks.size(); ++ti) {
                const auto& t = tasks[ti];
                const uint8_t* w_ptr = cache.get_tensor(t.name + ".weight");
                const uint8_t* s_ptr = cache.get_tensor(t.name + ".scales");
                const uint8_t* b_ptr = cache.get_tensor(t.name + ".biases");
                uint32_t chunk_R = (t.R > 16384u) ? 8192u : t.R;
                double tot_sig = 0.0, tot_err = 0.0;

                for (uint32_t r0 = 0; r0 < t.R; r0 += chunk_R) {
                    uint32_t sr = std::min(chunk_R, t.R - r0);
                    memcpy([buf_q8_w contents], w_ptr + (size_t)r0 * t.C, (size_t)sr * t.C);
                    memcpy([buf_q8_s contents], s_ptr + (size_t)r0 * (t.C / 64) * 2, (size_t)sr * (t.C / 64) * 2);
                    memcpy([buf_q8_b contents], b_ptr + (size_t)r0 * (t.C / 64) * 2, (size_t)sr * (t.C / 64) * 2);
                    size_t n_blks = (size_t)(sr / 128) * (t.C / 128);

                    MatParams mp = {sr, t.C, t.bits, 1.0f, r0, 0u};
                    uint32_t total_streams = (sr / 16) * 32;
                    uint32_t n_pairs = t.C / 4;
                    uint32_t batch_sz = (uint32_t)std::max<size_t>(1, HIST_SLOTS / n_pairs);

                    @autoreleasepool {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_q8_row];
                    [enc setBuffer:buf_q8_w offset:0 atIndex:0]; [enc setBuffer:buf_q8_s offset:0 atIndex:1];
                    [enc setBuffer:buf_q8_b offset:0 atIndex:2]; [enc setBuffer:buf_w_tmp offset:0 atIndex:3];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:4];
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
                if ((ti + 1) % 25 == 0 || ti + 1 == tasks.size() || snr < 22.0f) {
                    double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
                    printf("  [%3zu/%zu] %-56s [%6u,%5u] %ub SNR=%5.2f dB (%.1fs)\n",
                           ti + 1, tasks.size(), t.name.c_str(), t.R, t.C, t.bits, snr, el);
                    fflush(stdout);
                }
            }

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

            std::string out_path = out_dir + "/nonmoe.tffpw";
            auto write_all = [&](FILE* f, const void* p, size_t n) {
                if (n && fwrite(p, 1, n, f) != n) { fprintf(stderr, "Write error on %s\n", out_path.c_str()); exit(1); }
            };
            FILE* fout = fopen(out_path.c_str(), "wb");
            if (!fout) { fprintf(stderr, "Cannot write %s\n", out_path.c_str()); return 1; }
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
            if (fclose(fout) != 0) { fprintf(stderr, "Write error on %s\n", out_path.c_str()); return 1; }
            double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            printf("NON-MOE DONE in %.1fs -> %s (%.3f GB) | Mean SNR: 4b=%.2f dB (n=%d), 5b=%.2f dB (n=%d)\n",
                   el, out_path.c_str(), (double)(end_pos + pad_end) / 1e9,
                   snr_cnt[4] ? snr_sum[4]/snr_cnt[4] : 0.0, snr_cnt[4],
                   snr_cnt[5] ? snr_sum[5]/snr_cnt[5] : 0.0, snr_cnt[5]);
            return 0;
        }

        // ====================================================================
        // MODE B: MOE LAYER CONVERSION -> layer_XX.tffpw ("TFPW", version=3)
        // ====================================================================
        // Auto-detect GLM-5.3-Flash ("model.language_model.layers", H=4096, I=2048, E=288)
        // vs GLM-5.3 753B glm_moe_dsa ("model.layers", H=6144, I=2048, E=256) from safetensors metadata.
        std::string layer_prefix = "model.language_model.layers";
        {
            char probe[256];
            snprintf(probe, sizeof(probe), "model.layers.%d.mlp.experts.0.gate_proj.weight", l_start);
            if (cache.has_tensor(probe)) layer_prefix = "model.layers";
        }
        int H = 4096, I = 2048, N_EXP = 0;
        std::string src_dtype;
        {
            char probe[256];
            snprintf(probe, sizeof(probe), "%s.%d.mlp.experts.0.gate_proj.weight", layer_prefix.c_str(), l_start);
            size_t dummy = 0;
            cache.get_tensor(probe, &src_dtype, &dummy, &I, &H);
            while (true) {
                snprintf(probe, sizeof(probe), "%s.%d.mlp.experts.%d.gate_proj.weight", layer_prefix.c_str(), l_start, N_EXP);
                if (!cache.has_tensor(probe)) break;
                N_EXP++;
            }
            cache.evict_all();
        }
        const bool src_bf16 = (src_dtype == "BF16");
        if (!src_bf16 && src_dtype != "F8_E4M3") {
            fprintf(stderr, "Unsupported routed expert dtype %s (want F8_E4M3 or BF16)\n", src_dtype.c_str());
            return 1;
        }
        const uint32_t N_SC = (uint32_t)((H / 128) * (I / 128));
        const size_t MAX_ELEMS = (size_t)H * (size_t)I;
        const int BATCH_STREAMS = (H > 4096) ? 2048 : 4096;
        const size_t MAX_PACK_BYTES = (size_t)(I / 16) * (size_t)(H / 128) * 10ULL * 32ULL * 4ULL;
        id<MTLBuffer> buf_fp8_in   = src_bf16 ? nil : [dev newBufferWithLength:MAX_ELEMS options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sc_inv   = src_bf16 ? nil : [dev newBufferWithLength:(size_t)N_SC * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_bf16_in  = src_bf16 ? [dev newBufferWithLength:MAX_ELEMS * 2ULL options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> buf_w_tmp    = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_w_rot    = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_blk_ss   = [dev newBufferWithLength:(size_t)N_SC * 128ULL * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_streams  = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_hist     = [dev newBufferWithLength:(size_t)BATCH_STREAMS * (size_t)(H / 4) * 256ULL * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_mat_pack = [dev newBufferWithLength:MAX_PACK_BYTES options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mat_sc   = [dev newBufferWithLength:(size_t)N_SC * 2ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_se_out   = [dev newBufferWithLength:(size_t)(std::max(H, I) / 16) * 2ULL * 4ULL options:MTLResourceStorageModeShared];

        auto heat_ranks = load_heat_ranks(trace_path, N_EXP);
        const char* proj_names[3] = {"gate_proj", "up_proj", "down_proj"};
        const int proj_R[3] = {I, I, H};
        const int proj_C[3] = {H, H, I};

        printf("============================================================================================================\n");
        printf("TFF-PrismWave v3 MOE CONVERTER | mode=%s | layers=%d..%d | H=%d I=%d E=%d prefix=%s src=%s | out=%s\n",
               mode.c_str(), l_start, l_end, H, I, N_EXP, layer_prefix.c_str(), src_dtype.c_str(), out_dir.c_str());
        printf("============================================================================================================\n");
        fflush(stdout);

        auto t_all = std::chrono::high_resolution_clock::now();
        for (int lyr = l_start; lyr <= l_end; ++lyr) {
            char out_path[512];
            snprintf(out_path, sizeof(out_path), "%s/layer_%02d.tffpw", out_dir.c_str(), lyr);
            if (skip_existing) {
                struct stat cst{};
                if (stat(out_path, &cst) == 0 && cst.st_size > 1000000000LL) {
                    printf("L%02d SKIP (exists: %.3f GB)\n", lyr, (double)cst.st_size / 1e9);
                    fflush(stdout);
                    continue;
                }
            }
            auto t_lyr = std::chrono::high_resolution_clock::now();
            std::vector<int> ranks(N_EXP);
            bool used_trace = false, used_router = false;
            if (heat_ranks.count(lyr)) {
                ranks = heat_ranks[lyr];
                used_trace = true;
            } else {
                ranks = tfq_moe::rank_glm_experts_datafree(cache, lyr, N_EXP, H, &used_router, layer_prefix.c_str());
            }
            if (lyr == l_start) {
                const char* datafree = src_bf16
                    ? (used_router ? "DATA-FREE router sensitivity (BF16 source: no weight_scale_inv for the weight term)"
                                   : "none (BF16 source without a router: index order)")
                    : (used_router ? "DATA-FREE router+weight sensitivity (no calibration data)"
                                   : "DATA-FREE expert weight-scale sensitivity (no calibration data)");
                printf("HeatAlloc: %s\n", used_trace
                    ? ("TRACE-GUIDED (uses calibration routing counts from " + trace_path + ")").c_str()
                    : datafree);
                fflush(stdout);
            }

            std::vector<PrismExpertMeta> metas(N_EXP);
            memset(metas.data(), 0, metas.size() * sizeof(PrismExpertMeta));
            const int thr5 = (N_EXP == 288) ? 64  : (N_EXP / 4);
            const int thr4 = (N_EXP == 288) ? 160 : ((N_EXP * 5) / 8);
            const int thr43 = (N_EXP == 288) ? 96 : (N_EXP / 3);
            uint64_t cur_word = 0; uint32_t cur_sc = 0;
            for (int e = 0; e < N_EXP; ++e) {
                int r = ranks[e];
                uint32_t b = 4;
                if (mode == "heat543")       b = (r < thr5) ? 5u : (r < thr4 ? 4u : 3u);
                else if (mode == "heat443")  b = (r < thr4) ? 4u : 3u;
                else if (mode == "heat43")   b = (r < thr43) ? 4u : 3u;
                else if (mode == "uniform4") b = 4u;
                else if (mode == "uniform3") b = 3u;
                metas[e].b_bits = b; metas[e].heat_rank = (uint32_t)r;
                uint64_t wpm = (uint64_t)(I / 16) * (H / 128) * (b * 2u) * 32u;
                metas[e].gate_word_off = cur_word; cur_word += wpm;
                metas[e].up_word_off   = cur_word; cur_word += wpm;
                metas[e].down_word_off = cur_word; cur_word += wpm;
                metas[e].gate_sc_off = cur_sc; cur_sc += N_SC;
                metas[e].up_sc_off   = cur_sc; cur_sc += N_SC;
                metas[e].down_sc_off = cur_sc; cur_sc += N_SC;
            }
            std::vector<uint16_t> all_scales(cur_sc);
            std::vector<uint32_t> all_weights(cur_word);
            double snr_sum[6] = {}; int snr_cnt[6] = {};

            for (int e = 0; e < N_EXP; ++e) {
                uint32_t b = metas[e].b_bits;
                uint64_t wpm = (uint64_t)(I / 16) * (H / 128) * (b * 2u) * 32u;
                for (int p = 0; p < 3; ++p) {
                    int R = proj_R[p], C = proj_C[p];
                    char base[256];
                    snprintf(base, sizeof(base), "%s.%d.mlp.experts.%d.%s", layer_prefix.c_str(), lyr, e, proj_names[p]);
                    if (src_bf16) {
                        std::string dtype; size_t nw = 0; int tR = 0, tC = 0;
                        const uint8_t* w_ptr = cache.get_tensor(std::string(base) + ".weight", &dtype, &nw, &tR, &tC);
                        if (dtype != "BF16" || tR != R || tC != C || nw != (size_t)R * (size_t)C * 2u) {
                            fprintf(stderr, "Unexpected %s.weight: dtype %s shape [%d, %d] bytes %zu (want BF16 [%d, %d])\n",
                                base, dtype.c_str(), tR, tC, nw, R, C);
                            return 1;
                        }
                        tfq_common::check_finite_bf16(reinterpret_cast<const uint16_t*>(w_ptr), (uint32_t)R, (uint32_t)C, std::string(base) + ".weight");
                        memcpy([buf_bf16_in contents], w_ptr, nw);
                    } else {
                        size_t nw = 0, ns = 0;
                        const uint8_t* w_ptr = cache.get_tensor(std::string(base) + ".weight", nullptr, &nw);
                        const uint8_t* s_ptr = cache.get_tensor(std::string(base) + ".weight_scale_inv", nullptr, &ns);
                        tfq_common::check_finite_fp8_e4m3(w_ptr, (uint32_t)R, (uint32_t)C, std::string(base) + ".weight");
                        tfq_common::check_finite_f32((const float*)s_ptr, (uint32_t)(R / 128), (uint32_t)(C / 128), std::string(base) + ".weight_scale_inv");
                        memcpy([buf_fp8_in contents], w_ptr, nw);
                        memcpy([buf_sc_inv contents], s_ptr, ns);
                    }

                    MatParams mp = {(uint32_t)R, (uint32_t)C, b, 1.0f, 0u, 0u};
                    int total_streams = (R / 16) * 32;
                    int batch_sz = (C < H) ? (BATCH_STREAMS * (H / C)) : BATCH_STREAMS;

                    @autoreleasepool {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

                    if (src_bf16) {
                        [enc setComputePipelineState:pso_bf16_row];
                        [enc setBuffer:buf_bf16_in offset:0 atIndex:0]; [enc setBuffer:buf_w_tmp offset:0 atIndex:1];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(C / 128, R, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    } else {
                        [enc setComputePipelineState:pso_fp8_row];
                        [enc setBuffer:buf_fp8_in offset:0 atIndex:0]; [enc setBuffer:buf_sc_inv offset:0 atIndex:1];
                        [enc setBuffer:buf_fp8_lut offset:0 atIndex:2]; [enc setBuffer:buf_w_tmp offset:0 atIndex:3];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(C / 128, R, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    }

                    [enc setComputePipelineState:pso_col_fwht];
                    [enc setBuffer:buf_w_tmp offset:0 atIndex:0]; [enc setBuffer:buf_w_rot offset:0 atIndex:1];
                    [enc setBuffer:buf_blk_ss offset:0 atIndex:2]; [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(C, R / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_pack_nax];
                    [enc setBuffer:buf_w_rot offset:0 atIndex:0]; [enc setBuffer:buf_blk_ss offset:0 atIndex:1];
                    [enc setBuffer:buf_mat_sc offset:0 atIndex:2]; [enc setBuffer:buf_streams offset:0 atIndex:3];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(R / 16, C / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_enc[b]];
                    for (int off = 0; off < total_streams; off += batch_sz) {
                        int cnt = std::min(batch_sz, total_streams - off);
                        EncParams ep = {(uint32_t)R, (uint32_t)C, b, (uint32_t)off, (uint32_t)cnt};
                        [enc setBuffer:buf_streams offset:0 atIndex:0];
                        [enc setBuffer:buf_lut[b] offset:0 atIndex:1];
                        [enc setBuffer:buf_c64[b] offset:0 atIndex:2];
                        [enc setBuffer:buf_grid[b] offset:0 atIndex:3];
                        [enc setBuffer:buf_hist offset:0 atIndex:4];
                        [enc setBuffer:buf_mat_pack offset:0 atIndex:5];
                        [enc setBytes:&ep length:sizeof(ep) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }

                    [enc setComputePipelineState:pso_snr];
                    [enc setBuffer:buf_mat_pack offset:0 atIndex:0]; [enc setBuffer:buf_mat_sc offset:0 atIndex:1];
                    [enc setBuffer:buf_lut[b] offset:0 atIndex:2]; [enc setBuffer:buf_c64[b] offset:0 atIndex:3];
                    [enc setBuffer:buf_w_rot offset:0 atIndex:4]; [enc setBuffer:buf_se_out offset:0 atIndex:5];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(R / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }

                    const float* se = (const float*)[buf_se_out contents];
                    double s_sig = 0.0, s_err = 0.0;
                    for (int si_s = 0; si_s < R / 16; ++si_s) {
                        s_sig += (double)se[si_s * 2 + 0];
                        s_err += (double)se[si_s * 2 + 1];
                    }
                    float snr = (float)(10.0 * log10(s_sig / std::max(1e-12, s_err)));
                    if (snr < 10.0f || std::isnan(snr)) {
                        fprintf(stderr, "FATAL: Bad SNR %.3f on L%d E%d %s!\n", snr, lyr, e, proj_names[p]);
                        return 2;
                    }
                    snr_sum[b] += snr; snr_cnt[b]++;
                    uint64_t w_off = (p == 0) ? metas[e].gate_word_off : (p == 1 ? metas[e].up_word_off : metas[e].down_word_off);
                    uint32_t s_off = (p == 0) ? metas[e].gate_sc_off   : (p == 1 ? metas[e].up_sc_off   : metas[e].down_sc_off);
                    if (p == 0) metas[e].gate_snr_db = snr; else if (p == 1) metas[e].up_snr_db = snr; else metas[e].down_snr_db = snr;
                    memcpy(all_scales.data() + s_off, [buf_mat_sc contents], (size_t)N_SC * 2ULL);
                    memcpy(all_weights.data() + w_off, [buf_mat_pack contents], wpm * 4);
                }
            }
            cache.evict_all();

            uint64_t off_meta    = sizeof(PrismLayerHeader);
            uint64_t off_scales  = align_up(off_meta + metas.size() * sizeof(PrismExpertMeta), ALIGN_16K);
            uint64_t off_weights = align_up(off_scales + all_scales.size() * 2ULL, ALIGN_16K);
            PrismLayerHeader hdr{};
            hdr.magic = MAGIC_TFPW; hdr.version = FORMAT_VER; hdr.layer_idx = (uint32_t)lyr;
            hdr.num_experts = N_EXP; hdr.num_tensors = N_EXP * 3; hdr.hidden_size = H; hdr.moe_inter = I;
            hdr.reserved0 = tfq_xxh64::FLAG_XXH64_PRESENT;
            hdr.raw_fp8_bytes = (uint64_t)N_EXP * 3ULL * H * I;
            hdr.stream_words = all_weights.size(); hdr.scale_halfs = all_scales.size();
            hdr.off_meta = off_meta; hdr.off_scales = off_scales; hdr.off_weights = off_weights;
            hdr.reserved1[0] = tfq_xxh64::hash(metas.data(), metas.size() * sizeof(PrismExpertMeta));
            hdr.reserved1[1] = tfq_xxh64::hash(all_scales.data(), all_scales.size() * 2ULL);
            hdr.reserved1[2] = tfq_xxh64::hash(all_weights.data(), all_weights.size() * 4ULL);
            hdr.reserved1[3] = 0ULL;
            strncpy((char*)hdr.meta_tag, mode.c_str(), 31);

            std::string tmp_path = std::string(out_path) + ".tmp";
            auto write_all = [&](FILE* f, const void* p, size_t n) {
                if (n && fwrite(p, 1, n, f) != n) { fprintf(stderr, "Write error on %s\n", tmp_path.c_str()); exit(1); }
            };
            FILE* fout = fopen(tmp_path.c_str(), "wb");
            if (!fout) { fprintf(stderr, "Cannot write %s\n", tmp_path.c_str()); return 1; }
            write_all(fout, &hdr, sizeof(hdr));
            write_all(fout, metas.data(), metas.size() * sizeof(PrismExpertMeta));
            std::vector<uint8_t> zeros(ALIGN_16K, 0);
            size_t cur = off_meta + metas.size() * sizeof(PrismExpertMeta);
            if (off_scales > cur) write_all(fout, zeros.data(), off_scales - cur);
            write_all(fout, all_scales.data(), all_scales.size() * 2ULL);
            cur = off_scales + all_scales.size() * 2ULL;
            if (off_weights > cur) write_all(fout, zeros.data(), off_weights - cur);
            write_all(fout, all_weights.data(), all_weights.size() * 4ULL);
            size_t end_pos = off_weights + all_weights.size() * 4ULL;
            size_t pad_end = align_up(end_pos, ALIGN_16K) - end_pos;
            if (pad_end) write_all(fout, zeros.data(), pad_end);
            if (fclose(fout) != 0) { fprintf(stderr, "Write error on %s\n", tmp_path.c_str()); return 1; }
            if (rename(tmp_path.c_str(), out_path) != 0) { fprintf(stderr, "Cannot rename %s to %s\n", tmp_path.c_str(), out_path); return 1; }

            double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_lyr).count();
            double tot_dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_all).count();
            printf("L%02d [%s] %.3f GB in %4.1fs (tot %5.1fs) | SNR:", lyr, mode.c_str(), (double)(end_pos + pad_end) / 1e9, dt, tot_dt);
            for (int b = 3; b <= 5; ++b) if (snr_cnt[b]) printf(" %db=%.2fdB", b, snr_sum[b]/snr_cnt[b]);
            printf("\n"); fflush(stdout);
        }
    }
    return 0;
}
