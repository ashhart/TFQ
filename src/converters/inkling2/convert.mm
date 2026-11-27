// ============================================================================
// Inkling-2 (NVFP4 + MXFP8 3D-packed MoE + BF16 non-MoE) Apple Silicon Metal
// TFQ3 (2D-AffineCoset/half4) GPU Converter
//   - --mode uniform4|heat543|heat443|heat43|uniform3 :
//       Converts NVFP4 (L0..39) & MXFP8 (L40..47) 3D experts -> layer_XX.tffpw ("TFPW" v3)
//   - --mode nonmoe [--bits 5|4] :
//       Converts BF16 attention projections -> nonmoe.tffpw ("TFPN" v3)
//   - --copy-residuals :
//       Exports all non-expert BF16/F32 tensors + config/tokenizer into out_dir
// ============================================================================
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>
#include <chrono>
#include "../tfq_xxh64.hpp"
#include "../tfq4/tfq4_format.hpp"
#include "ink2_common.hpp"

using tfq_tables::PwV3TablesFile;

struct Tfq4MatParams { uint32_t R, C, r_off, pad0; };
struct Tfq4ScaleParams { uint32_t n_blocks; float scale_mult; };

static inline int snr_bucket(uint32_t b) {
    if (b == 2u)   return 0;
    if (b == 225u) return 1;
    if (b == 3u)   return 2;
    if (b == 4u)   return 3;
    if (b == 5u)   return 4;
    return 3;
}

struct PrismLayerHeader {
    uint32_t magic, version, layer_idx, num_experts, num_tensors, hidden_size, moe_inter, reserved0;
    uint64_t raw_fp8_bytes, stream_words, scale_halfs, off_meta, off_scales, off_weights;
    uint64_t reserved1[4];
    uint8_t  meta_tag[32];
};
static_assert(sizeof(PrismLayerHeader) == 144, "PrismLayerHeader must be 144 bytes");

struct PrismExpertMeta {
    uint32_t b_bits, heat_rank;
    uint64_t gate_word_off, up_word_off, down_word_off;
    uint32_t gate_sc_off, up_sc_off, down_sc_off;
    float    gate_snr_db, up_snr_db, down_snr_db;
    uint32_t reserved[2];
};
static_assert(sizeof(PrismExpertMeta) == 64, "PrismExpertMeta must be 64 bytes");

struct TfpnHeader {
    uint32_t magic, version, num_tensors, pad0;
    uint64_t off_meta, off_scales, off_weights, total_scale_halfs, total_weight_words, pad1;
};
static_assert(sizeof(TfpnHeader) == 64, "TfpnHeader must be 64 bytes");

struct TfpnTensorMeta {
    char     name[80];
    uint32_t R, C, b_bits, sc_off;
    uint64_t word_off, word_count;
    float    snr_db;
    uint32_t pad[3];
};
static_assert(sizeof(TfpnTensorMeta) == 128, "TfpnTensorMeta must be 128 bytes");

struct MatParams { uint32_t R, C, b_bits; float shrink; uint32_t r_off, pad0; };
struct EncParams { uint32_t R, C, b_bits, stream_off, n_streams; };

static inline bool tfpw_layer_complete(const std::string& path) {
    PrismLayerHeader h{};
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = fread(&h, sizeof(h), 1, f) == 1;
    fclose(f);
    struct stat st{};
    if (!ok || stat(path.c_str(), &st) != 0 || h.magic != ink2::MAGIC_TFPW || h.version != ink2::FORMAT_VER) return false;
    if (h.stream_words > (UINT64_MAX - h.off_weights - ink2::ALIGN_16K) / 4ULL) return false;
    return (uint64_t)st.st_size == ink2::align_up(h.off_weights + h.stream_words * 4ULL, ink2::ALIGN_16K);
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
    std::string model_dir = "", out_dir = "", trace_path = "", tables_path = "";
    std::string mode = "uniform3"; // uniform4 | heat543 | heat443 | heat43 | uniform3 | nonmoe
    std::string metallib = default_metallib("tfq3-inkling2-convert.metallib");
    int l_start = -1, l_end = -1;
    uint32_t nonmoe_bits = 5u;
    bool skip_existing = false;
    bool copy_residuals = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { model_dir.clear(); break; }
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--trace" && i + 1 < argc) trace_path = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--bits" && i + 1 < argc) nonmoe_bits = (uint32_t)atoi(argv[++i]);
        else if (a == "--metallib" && i + 1 < argc) metallib = argv[++i];
        else if (a == "--tables" && i + 1 < argc) tables_path = argv[++i];
        else if (a == "--layers" && i + 2 < argc) { l_start = atoi(argv[++i]); l_end = atoi(argv[++i]); }
        else if (a == "--skip-existing") skip_existing = true;
        else if (a == "--copy-residuals") copy_residuals = true;
    }
    if (model_dir.empty() || out_dir.empty()) {
        fprintf(stderr,
            "Usage: %s --model <dir> --out <dir> [--mode uniform4|heat543|heat443|heat43|uniform3|heat432|heat42|heat43225|uniform2|uniform225|nonmoe]\n"
            "          [--layers S E] [--trace <file>] [--bits 4|5] [--copy-residuals] [--skip-existing]\n"
            "          [--tables <path>] [--metallib <path>]\n", argv[0]);
        return 1;
    }
    if (mode != "uniform4" && mode != "heat543" && mode != "heat443" && mode != "heat43" &&
        mode != "uniform3" && mode != "heat432" && mode != "heat42" && mode != "heat43225" &&
        mode != "uniform2" && mode != "uniform225" && mode != "nonmoe") {
        fprintf(stderr, "Error: unknown --mode %s\n", mode.c_str());
        return 1;
    }
    if (nonmoe_bits < 3u || nonmoe_bits > 5u) {
        fprintf(stderr, "Error: --bits must be 3, 4, or 5\n");
        return 1;
    }

    mkdir(out_dir.c_str(), 0755);
    const bool mode_uses_2b   = (mode == "heat432" || mode == "heat42" || mode == "uniform2");
    const bool mode_uses_225b = (mode == "heat43225" || mode == "uniform225");
    const bool mode_uses_tfq4 = mode_uses_2b || mode_uses_225b;
    const bool explicit_tables = !tables_path.empty();
    std::string dst_tables = out_dir + "/pw_v3_tables.bin";
    if (!explicit_tables) tables_path = dst_tables;

    PwV3TablesFile tbl;
    tfq_tables::load_or_write_pwv3_tables(tables_path, tbl, explicit_tables);
    if (tables_path != dst_tables) ink2::copy_small_file(tables_path, dst_tables);
    if (mode_uses_tfq4) {
        tfq4::write_hybrid_pwv4_tables(out_dir + "/pw_v4_e8p_tables.bin", mode_uses_225b);
    }

    auto grid3 = tfq_tables::build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
    auto grid4 = tfq_tables::build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
    auto grid5 = tfq_tables::build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

    ink2::ShardCache cache;
    cache.init(model_dir);

    // Auto-detect num_layers, num_experts, hidden, inter from layer 0 experts.
    int num_layers = 0;
    while (true) {
        char k[192];
        snprintf(k, sizeof(k), "model.llm.layers.%d.mlp.experts.w13_weight", num_layers);
        if (!cache.has_tensor(k)) break;
        num_layers++;
    }
    if (num_layers == 0) {
        fprintf(stderr, "Error: no model.llm.layers.0.mlp.experts.w13_weight found in %s\n", model_dir.c_str());
        return 1;
    }
    if (l_start < 0) l_start = 0;
    if (l_end < 0 || l_end >= num_layers) l_end = num_layers - 1;

    ink2::TensorInfo w13_0, w2_0;
    cache.get_tensor("model.llm.layers.0.mlp.experts.w13_weight", &w13_0);
    cache.get_tensor("model.llm.layers.0.mlp.experts.w2_weight", &w2_0);
    cache.evict_all();
    if (w13_0.shape.size() != 3 || w2_0.shape.size() != 3) {
        fprintf(stderr, "Error: expected 3D expert tensors for Inkling-2\n");
        return 1;
    }
    const int N_EXP = (int)w13_0.shape[0];
    const int I     = (int)(w13_0.shape[1] / 2);
    const int H     = (int)w2_0.shape[1];
    if (H % 128 != 0 || I % 128 != 0 || w13_0.shape[1] != 2 * I) {
        fprintf(stderr, "Error: Inkling-2 dimensions H=%d I=%d must be multiples of 128\n", H, I);
        return 1;
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "Error: no Metal device found\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]] error:&err];
        if (!lib) { fprintf(stderr, "metallib err (%s): %s\n", metallib.c_str(), [[err localizedDescription] UTF8String]); return 1; }
        auto get_pso = [&](const char* name) {
            NSError* e = nil;
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:name]] error:&e];
            if (!p) { fprintf(stderr, "pso %s err: %s\n", name, [[e localizedDescription] UTF8String]); exit(1); }
            return p;
        };

        auto pso_nvfp4    = get_pso("nvfp4_dequant_row_fwht");
        auto pso_mxfp8    = get_pso("mxfp8_dequant_row_fwht");
        auto pso_bf16_row = get_pso("bf16_row_fwht");
        auto pso_col_fwht = get_pso("col_fwht_and_ss");
        auto pso_pack_nax = get_pso("finalize_scales_and_pack_nax");
        id<MTLComputePipelineState> pso_enc[6] = {};
        pso_enc[3] = get_pso("prism_encode_v3_b3");
        pso_enc[4] = get_pso("prism_encode_v3_b4");
        pso_enc[5] = get_pso("prism_encode_v3_b5");
        auto pso_snr = get_pso("prism_verify_v3_snr");
        id<MTLComputePipelineState> pso_tfq4_sc  = mode_uses_tfq4 ? get_pso("tfq4_finalize_scales") : nil;
        id<MTLComputePipelineState> pso_tfq4_2b  = mode_uses_2b   ? get_pso("tfq4_true_e8p_encode") : nil;
        id<MTLComputePipelineState> pso_tfq4_225 = mode_uses_225b ? get_pso("tfq4_true_e8p_1024_encode") : nil;

        float fp8_lut[256];
        for (int i = 0; i < 256; ++i) fp8_lut[i] = ink2::e4m3_to_f32_host((uint8_t)i);
        id<MTLBuffer> buf_fp8_lut = [dev newBufferWithBytes:fp8_lut length:1024 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_e8p_lut = nil;
        if (mode_uses_225b) {
            tfq4::PwV4E8Tables1024 t4{};
            tfq4::build_true_e8p_1024_tables(t4);
            buf_e8p_lut = [dev newBufferWithBytes:t4.e8p_lut length:sizeof(t4.e8p_lut) options:MTLResourceStorageModeShared];
        } else if (mode_uses_2b) {
            tfq4::PwV4E8Tables t4{};
            tfq4::build_true_e8p_tables(t4);
            buf_e8p_lut = [dev newBufferWithBytes:t4.e8p_lut length:sizeof(t4.e8p_lut) options:MTLResourceStorageModeShared];
        }
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

        // ====================================================================
        // MODE A: NON-MOE ATTENTION PROJECTIONS -> nonmoe.tffpw ("TFPN" v3)
        // ====================================================================
        if (mode == "nonmoe") {
            struct Task { std::string name; uint32_t R, C, bits; };
            std::vector<Task> tasks;
            std::set<std::string> quant_names;
            const char* attn_projs[] = {"wq_du", "wo_ud", "gate_attn", "wk_dv", "wv_dv", "wr_du"};
            for (int l = 0; l < num_layers; ++l) {
                for (const char* p : attn_projs) {
                    char full[192];
                    snprintf(full, sizeof(full), "model.llm.layers.%d.attn.%s.weight", l, p);
                    if (!cache.has_tensor(full)) continue;
                    ink2::TensorInfo ti;
                    cache.get_tensor(full, &ti);
                    if (ti.dtype == "BF16" && ti.shape.size() == 2 &&
                        (ti.shape[0] % 128 == 0) && (ti.shape[1] % 128 == 0)) {
                        std::string nm(full);
                        nm.resize(nm.size() - 7); // strip ".weight"
                        tasks.push_back({nm, (uint32_t)ti.shape[0], (uint32_t)ti.shape[1], nonmoe_bits});
                        quant_names.insert(full);
                    }
                }
            }
            cache.evict_all();
            if (tasks.empty()) {
                fprintf(stderr, "Error: no BF16 attention projections found for --mode nonmoe\n");
                return 1;
            }

            size_t max_e = 0, max_sc = 0, max_r = 0;
            for (const auto& tk : tasks) {
                max_e  = std::max(max_e,  (size_t)tk.R * tk.C);
                max_sc = std::max(max_sc, (size_t)(tk.R / 128) * (tk.C / 128));
                max_r  = std::max(max_r,  (size_t)tk.R);
            }
            constexpr size_t HIST_SLOTS = 524288ULL;
            id<MTLBuffer> buf_bf16   = [dev newBufferWithLength:max_e * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_w_tmp  = [dev newBufferWithLength:max_e * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> buf_w_rot  = [dev newBufferWithLength:max_e * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> buf_blk_ss = [dev newBufferWithLength:max_sc * 128ULL * 4ULL options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_str    = [dev newBufferWithLength:max_e * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> buf_hist   = [dev newBufferWithLength:HIST_SLOTS * 256ULL * 4ULL options:MTLResourceStorageModePrivate];
            id<MTLBuffer> buf_pack   = [dev newBufferWithLength:max_e options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_sc     = [dev newBufferWithLength:max_sc * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_se     = [dev newBufferWithLength:(max_r / 16) * 2ULL * 4ULL options:MTLResourceStorageModeShared];

            std::vector<TfpnTensorMeta> metas(tasks.size());
            uint64_t total_sc = 0, total_wd = 0;
            for (size_t i = 0; i < tasks.size(); ++i) {
                memset(&metas[i], 0, sizeof(TfpnTensorMeta));
                strncpy(metas[i].name, tasks[i].name.c_str(), 79);
                metas[i].R = tasks[i].R; metas[i].C = tasks[i].C; metas[i].b_bits = tasks[i].bits;
                uint64_t sc_cnt = (uint64_t)(tasks[i].R / 128) * (tasks[i].C / 128);
                uint64_t wd_cnt = (uint64_t)(tasks[i].R / 16) * (tasks[i].C / 128) * (tasks[i].bits * 2u) * 32u;
                metas[i].sc_off = (uint32_t)total_sc; total_sc += sc_cnt;
                metas[i].word_off = total_wd; metas[i].word_count = wd_cnt; total_wd += wd_cnt;
            }
            std::vector<uint16_t> all_scales(total_sc);
            std::vector<uint32_t> all_weights(total_wd);

            printf("Inkling-2 NON-MOE Metal (%ub) | %zu tensors | %.3f GiB\n",
                   nonmoe_bits, tasks.size(), (double)(total_wd * 4ULL + total_sc * 2ULL) / (double)(1ULL << 30));
            fflush(stdout);
            auto t0 = std::chrono::steady_clock::now();
            double snr_sum = 0.0;
            for (size_t ti = 0; ti < tasks.size(); ++ti) {
                const auto& t = tasks[ti];
                ink2::TensorInfo info;
                const uint8_t* ptr = cache.get_tensor(t.name + ".weight", &info);
                tfq_common::check_finite_bf16(reinterpret_cast<const uint16_t*>(ptr), t.R, t.C, t.name + ".weight");
                memcpy([buf_bf16 contents], ptr, (size_t)t.R * t.C * 2);

                MatParams mp = {t.R, t.C, t.bits, 1.0f, 0u, 0u};
                uint32_t total_streams = (t.R / 16) * 32;
                uint32_t batch_sz = (uint32_t)std::max<size_t>(1, HIST_SLOTS / (t.C / 4));
                @autoreleasepool {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_bf16_row];
                    [enc setBuffer:buf_bf16 offset:0 atIndex:0]; [enc setBuffer:buf_w_tmp offset:0 atIndex:1];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(t.C / 128, t.R, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_col_fwht];
                    [enc setBuffer:buf_w_tmp offset:0 atIndex:0]; [enc setBuffer:buf_w_rot offset:0 atIndex:1];
                    [enc setBuffer:buf_blk_ss offset:0 atIndex:2]; [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(t.C, t.R / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_pack_nax];
                    [enc setBuffer:buf_w_rot offset:0 atIndex:0]; [enc setBuffer:buf_blk_ss offset:0 atIndex:1];
                    [enc setBuffer:buf_sc offset:0 atIndex:2]; [enc setBuffer:buf_str offset:0 atIndex:3];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(t.R / 16, t.C / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_enc[t.bits]];
                    for (uint32_t off = 0; off < total_streams; off += batch_sz) {
                        uint32_t cnt = std::min(batch_sz, total_streams - off);
                        EncParams ep = {t.R, t.C, t.bits, off, cnt};
                        [enc setBuffer:buf_str offset:0 atIndex:0]; [enc setBuffer:buf_lut[t.bits] offset:0 atIndex:1];
                        [enc setBuffer:buf_c64[t.bits] offset:0 atIndex:2]; [enc setBuffer:buf_grid[t.bits] offset:0 atIndex:3];
                        [enc setBuffer:buf_hist offset:0 atIndex:4]; [enc setBuffer:buf_pack offset:0 atIndex:5];
                        [enc setBytes:&ep length:sizeof(ep) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(cnt, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }

                    [enc setComputePipelineState:pso_snr];
                    [enc setBuffer:buf_pack offset:0 atIndex:0]; [enc setBuffer:buf_sc offset:0 atIndex:1];
                    [enc setBuffer:buf_lut[t.bits] offset:0 atIndex:2]; [enc setBuffer:buf_c64[t.bits] offset:0 atIndex:3];
                    [enc setBuffer:buf_w_rot offset:0 atIndex:4]; [enc setBuffer:buf_se offset:0 atIndex:5];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(t.R / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                const float* se = (const float*)[buf_se contents];
                double sig = 0.0, err_s = 0.0;
                for (uint32_t s = 0; s < t.R / 16; ++s) { sig += se[2 * s]; err_s += se[2 * s + 1]; }
                metas[ti].snr_db = (float)(10.0 * std::log10(sig / std::max(err_s, 1e-18)));
                snr_sum += metas[ti].snr_db;
                memcpy(all_scales.data() + metas[ti].sc_off, [buf_sc contents], (size_t)(t.R / 128) * (t.C / 128) * 2);
                memcpy(all_weights.data() + metas[ti].word_off, [buf_pack contents], metas[ti].word_count * 4);
                if ((ti + 1) % 48 == 0 || ti + 1 == tasks.size()) {
                    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    printf("  [%3zu/%zu] %-48s SNR=%.2f dB (%.1fs)\n", ti + 1, tasks.size(), t.name.c_str(), metas[ti].snr_db, el);
                    fflush(stdout);
                    cache.evict_all();
                }
            }
            uint64_t off_meta = sizeof(TfpnHeader);
            uint64_t off_sc   = ink2::align_up(off_meta + metas.size() * sizeof(TfpnTensorMeta), ink2::ALIGN_16K);
            uint64_t off_w    = ink2::align_up(off_sc + all_scales.size() * 2ULL, ink2::ALIGN_16K);
            uint64_t xxh = tfq_xxh64::hash_sections3(
                metas.data(), metas.size() * sizeof(TfpnTensorMeta),
                all_scales.data(), all_scales.size() * 2ULL,
                all_weights.data(), all_weights.size() * 4ULL
            );
            TfpnHeader hdr{ink2::MAGIC_TFPN, ink2::FORMAT_VER, (uint32_t)metas.size(), tfq_xxh64::FLAG_XXH64_PRESENT,
                           off_meta, off_sc, off_w, all_scales.size(), all_weights.size(), xxh};
            std::vector<uint8_t> zeros(ink2::ALIGN_16K, 0);
            std::string out_path = out_dir + "/nonmoe.tffpw", tmp = out_path + ".tmp";
            FILE* fo = fopen(tmp.c_str(), "wb");
            if (!fo) { fprintf(stderr, "Cannot write %s\n", tmp.c_str()); return 1; }
            ink2::write_all_or_exit(fo, &hdr, sizeof(hdr), tmp);
            ink2::write_all_or_exit(fo, metas.data(), metas.size() * sizeof(TfpnTensorMeta), tmp);
            ink2::write_all_or_exit(fo, zeros.data(), off_sc - (off_meta + metas.size() * sizeof(TfpnTensorMeta)), tmp);
            ink2::write_all_or_exit(fo, all_scales.data(), all_scales.size() * 2ULL, tmp);
            ink2::write_all_or_exit(fo, zeros.data(), off_w - (off_sc + all_scales.size() * 2ULL), tmp);
            ink2::write_all_or_exit(fo, all_weights.data(), all_weights.size() * 4ULL, tmp);
            size_t endp = off_w + all_weights.size() * 4ULL, pade = ink2::align_up(endp, ink2::ALIGN_16K) - endp;
            ink2::write_all_or_exit(fo, zeros.data(), pade, tmp);
            ink2::close_or_exit(fo, tmp);
            ink2::rename_or_exit(tmp, out_path);
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("NON-MOE DONE in %.1fs -> %s (%.3f GiB) | mean SNR=%.2f dB\n",
                   el, out_path.c_str(), (double)(endp + pade) / (double)(1ULL << 30), snr_sum / std::max<size_t>(1, tasks.size()));
            if (copy_residuals) ink2::export_residuals_and_assets(cache, model_dir, out_dir, quant_names);
            return 0;
        }

        // ====================================================================
        // MODE B: MoE LAYER CONVERSION -> layer_00..47.tffpw ("TFPW" v3)
        // ====================================================================
        const uint32_t N_SC = (uint32_t)((H / 128) * (I / 128));
        const size_t MAX_ELEMS = (size_t)H * (size_t)I;
        const int MAX_C = std::max(H, I);
        const size_t W13_MAX_CODE_BYTES = (size_t)(2 * I) * (size_t)H;       // MXFP8 w13: 2*I*H
        const size_t W13_MAX_SC_BYTES   = (size_t)(2 * I) * (size_t)(H / 16); // NVFP4 w13: 2*I*(H/16)
        const int BATCH_STREAMS = 1024;

        id<MTLBuffer> buf_codes    = [dev newBufferWithLength:W13_MAX_CODE_BYTES options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_scales   = [dev newBufferWithLength:W13_MAX_SC_BYTES   options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_w_tmp    = [dev newBufferWithLength:MAX_ELEMS * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_w_rot    = [dev newBufferWithLength:MAX_ELEMS * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_blk_ss   = [dev newBufferWithLength:(size_t)N_SC * 128ULL * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_streams  = [dev newBufferWithLength:MAX_ELEMS * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_hist     = [dev newBufferWithLength:(size_t)BATCH_STREAMS * (size_t)(MAX_C / 4) * 256ULL * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_mat_pack = [dev newBufferWithLength:(size_t)(I / 16) * (size_t)(H / 128) * 10ULL * 32ULL * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mat_sc   = [dev newBufferWithLength:(size_t)N_SC * 2ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_se_out   = [dev newBufferWithLength:(size_t)(std::max(H, I) / 16) * 2ULL * 4ULL options:MTLResourceStorageModeShared];
        const uint32_t MAX_TFQ4_TG = (uint32_t)((MAX_ELEMS / 16ULL + 255ULL) / 256ULL);
        id<MTLBuffer> buf_tfq4_met = mode_uses_tfq4
            ? [dev newBufferWithLength:(size_t)MAX_TFQ4_TG * 2ULL * 4ULL options:MTLResourceStorageModeShared]
            : nil;

        auto heat_ranks = trace_path.empty() ? std::map<int, std::vector<int>>{} : ink2::load_heat_ranks(trace_path, N_EXP);

        printf("====================================================================================================\n");
        printf("Inkling-2 Metal TFQ3 Converter | L=%d..%d (%d total) | H=%d I=%d E=%d | mode=%s | %s\n",
               l_start, l_end, num_layers, H, I, N_EXP, mode.c_str(),
               trace_path.empty() ? "DATA-FREE router-norm heat" : ("TRACE-GUIDED " + trace_path).c_str());
        printf("====================================================================================================\n");
        fflush(stdout);

        auto t_all = std::chrono::steady_clock::now();
        for (int lyr = l_start; lyr <= l_end; ++lyr) {
            char out_path[512];
            snprintf(out_path, sizeof(out_path), "%s/layer_%02d.tffpw", out_dir.c_str(), lyr);
            if (skip_existing && tfpw_layer_complete(out_path)) {
                printf("  [L%02d] SKIP (complete)\n", lyr);
                fflush(stdout);
                continue;
            }
            auto t_lyr = std::chrono::steady_clock::now();

            std::vector<int> ranks = heat_ranks.count(lyr)
                ? heat_ranks[lyr]
                : ink2::rank_experts_datafree(cache, lyr, N_EXP, H);

            char k_w13[192], k_s13[192], k_a13[192], k_d13[192];
            char k_w2[192],  k_s2[192],  k_a2[192],  k_d2[192];
            snprintf(k_w13, sizeof(k_w13), "model.llm.layers.%d.mlp.experts.w13_weight", lyr);
            snprintf(k_s13, sizeof(k_s13), "model.llm.layers.%d.mlp.experts.w13_weight_scale", lyr);
            snprintf(k_a13, sizeof(k_a13), "model.llm.layers.%d.mlp.experts.w13_weight_amax", lyr);
            snprintf(k_d13, sizeof(k_d13), "model.llm.layers.%d.mlp.experts.w13_weight_outer_scale_denominator", lyr);
            snprintf(k_w2,  sizeof(k_w2),  "model.llm.layers.%d.mlp.experts.w2_weight", lyr);
            snprintf(k_s2,  sizeof(k_s2),  "model.llm.layers.%d.mlp.experts.w2_weight_scale", lyr);
            snprintf(k_a2,  sizeof(k_a2),  "model.llm.layers.%d.mlp.experts.w2_weight_amax", lyr);
            snprintf(k_d2,  sizeof(k_d2),  "model.llm.layers.%d.mlp.experts.w2_weight_outer_scale_denominator", lyr);

            const bool is_nvfp4 = cache.has_tensor(k_a13);
            ink2::TensorInfo ti_w13, ti_s13, ti_w2, ti_s2;
            const uint8_t* p_w13 = cache.get_tensor(k_w13, &ti_w13);
            const uint8_t* p_s13 = cache.get_tensor(k_s13, &ti_s13);
            const uint8_t* p_w2  = cache.get_tensor(k_w2,  &ti_w2);
            const uint8_t* p_s2  = cache.get_tensor(k_s2,  &ti_s2);
            const float *p_a13 = nullptr, *p_d13 = nullptr, *p_a2 = nullptr, *p_d2 = nullptr;
            if (is_nvfp4) {
                p_a13 = reinterpret_cast<const float*>(cache.get_tensor(k_a13));
                p_d13 = reinterpret_cast<const float*>(cache.get_tensor(k_d13));
                p_a2  = reinterpret_cast<const float*>(cache.get_tensor(k_a2));
                p_d2  = reinterpret_cast<const float*>(cache.get_tensor(k_d2));
            }

            std::vector<PrismExpertMeta> metas(N_EXP);
            memset(metas.data(), 0, metas.size() * sizeof(PrismExpertMeta));
            const int thr5    = N_EXP / 4;          // 96 / 384 (25%)
            const int thr4    = (N_EXP * 5) / 9;    // 213 / 384 (~55.5%)
            const int thr43   = N_EXP / 3;          // 128 / 384 (33.3%)
            const int thr4_15 = (N_EXP * 15 + 50) / 100;
            const int thr3_40 = (N_EXP * 40 + 50) / 100;
            const int thr4_20 = (N_EXP * 20 + 50) / 100;
            uint64_t cur_word = 0; uint32_t cur_sc = 0;
            for (int e = 0; e < N_EXP; ++e) {
                int r = ranks[e];
                uint32_t b = 4u;
                if (mode == "heat543")         b = (r < thr5) ? 5u : (r < (N_EXP * 5) / 8 ? 4u : 3u);
                else if (mode == "heat443")    b = (r < thr4) ? 4u : 3u;
                else if (mode == "heat43")     b = (r < thr43) ? 4u : 3u;
                else if (mode == "uniform4")   b = 4u;
                else if (mode == "uniform3")   b = 3u;
                else if (mode == "heat432")    b = (r < thr4_15) ? 4u : ((r < thr3_40) ? 3u : 2u);
                else if (mode == "heat42")     b = (r < thr4_20) ? 4u : 2u;
                else if (mode == "heat43225")  b = (r < thr4_15) ? 4u : ((r < thr3_40) ? 3u : 225u);
                else if (mode == "uniform2")   b = 2u;
                else if (mode == "uniform225") b = 225u;
                metas[e].b_bits = b;
                metas[e].heat_rank = (uint32_t)r;
                uint64_t wpm = tfq4::tfpw_proj_words((uint32_t)I, (uint32_t)H, b);
                metas[e].gate_word_off = cur_word; cur_word += wpm;
                metas[e].up_word_off   = cur_word; cur_word += wpm;
                metas[e].down_word_off = cur_word; cur_word += wpm;
                metas[e].gate_sc_off = cur_sc; cur_sc += N_SC;
                metas[e].up_sc_off   = cur_sc; cur_sc += N_SC;
                metas[e].down_sc_off = cur_sc; cur_sc += N_SC;
            }
            std::vector<uint16_t> all_scales(cur_sc);
            std::vector<uint32_t> all_weights(cur_word);
            double snr_sum[5] = {}; int snr_cnt[5] = {};

            const size_t w13_code_stride = is_nvfp4 ? (size_t)(2 * I) * (H / 2)  : (size_t)(2 * I) * H;
            const size_t w13_sc_stride   = is_nvfp4 ? (size_t)(2 * I) * (H / 16) : (size_t)(2 * I) * (H / 32);
            const size_t w2_code_stride  = is_nvfp4 ? (size_t)H * (I / 2)        : (size_t)H * I;
            const size_t w2_sc_stride    = is_nvfp4 ? (size_t)H * (I / 16)       : (size_t)H * (I / 32);

            auto encode_proj = [&](uint32_t R, uint32_t C, uint32_t b, uint32_t row0, uint32_t row_step,
                                   float outer, uint64_t w_off, uint32_t s_off, float& out_snr) {
                ink2::Ink2Params ip{R, C, row0, row_step, outer, 0u};
                MatParams mp{R, C, b, 1.0f, 0u, 0u};
                const bool is_tfq4 = (b == tfq4::TFQ4_BITS_200 || b == tfq4::TFQ4_BITS_225);
                uint32_t tfq4_ntg = is_tfq4 ? (uint32_t)((((uint64_t)R * (C / 16u)) + 255ull) / 256ull) : 0u;
                int total_streams = (int)(R / 16) * 32;
                int batch_sz = std::max(1, (int)((size_t)BATCH_STREAMS * (size_t)MAX_C / (size_t)C));

                @autoreleasepool {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

                    [enc setComputePipelineState:(is_nvfp4 ? pso_nvfp4 : pso_mxfp8)];
                    [enc setBuffer:buf_codes offset:0 atIndex:0];
                    [enc setBuffer:buf_scales offset:0 atIndex:1];
                    [enc setBuffer:buf_fp8_lut offset:0 atIndex:2];
                    [enc setBuffer:buf_w_tmp offset:0 atIndex:3];
                    [enc setBytes:&ip length:sizeof(ip) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(C / 128, R, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_col_fwht];
                    [enc setBuffer:buf_w_tmp offset:0 atIndex:0];
                    [enc setBuffer:buf_w_rot offset:0 atIndex:1];
                    [enc setBuffer:buf_blk_ss offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(C, R / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    if (is_tfq4) {
                        Tfq4ScaleParams sp{N_SC, (b == tfq4::TFQ4_BITS_225) ? tfq4::TRUE_E8P1024_BASE_SCALE : tfq4::TRUE_E8P_BASE_SCALE};
                        [enc setComputePipelineState:pso_tfq4_sc];
                        [enc setBuffer:buf_blk_ss offset:0 atIndex:0];
                        [enc setBuffer:buf_mat_sc offset:0 atIndex:1];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(N_SC, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        Tfq4MatParams tp{R, C, 0u, 0u};
                        [enc setComputePipelineState:((b == tfq4::TFQ4_BITS_225) ? pso_tfq4_225 : pso_tfq4_2b)];
                        [enc setBuffer:buf_w_rot offset:0 atIndex:0];
                        [enc setBuffer:buf_mat_sc offset:0 atIndex:1];
                        [enc setBuffer:buf_e8p_lut offset:0 atIndex:2];
                        [enc setBuffer:buf_mat_pack offset:0 atIndex:3];
                        [enc setBuffer:buf_tfq4_met offset:0 atIndex:4];
                        [enc setBytes:&tp length:sizeof(tp) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(tfq4_ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    } else {
                        [enc setComputePipelineState:pso_pack_nax];
                        [enc setBuffer:buf_w_rot offset:0 atIndex:0];
                        [enc setBuffer:buf_blk_ss offset:0 atIndex:1];
                        [enc setBuffer:buf_mat_sc offset:0 atIndex:2];
                        [enc setBuffer:buf_streams offset:0 atIndex:3];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(R / 16, C / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:pso_enc[b]];
                        for (int off = 0; off < total_streams; off += batch_sz) {
                            int cnt = std::min(batch_sz, total_streams - off);
                            EncParams ep{R, C, b, (uint32_t)off, (uint32_t)cnt};
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
                        [enc setBuffer:buf_mat_pack offset:0 atIndex:0];
                        [enc setBuffer:buf_mat_sc offset:0 atIndex:1];
                        [enc setBuffer:buf_lut[b] offset:0 atIndex:2];
                        [enc setBuffer:buf_c64[b] offset:0 atIndex:3];
                        [enc setBuffer:buf_w_rot offset:0 atIndex:4];
                        [enc setBuffer:buf_se_out offset:0 atIndex:5];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(R / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    }
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }
                double sig = 0.0, err_s = 0.0;
                if (is_tfq4) {
                    const float* met = (const float*)[buf_tfq4_met contents];
                    for (uint32_t s = 0; s < tfq4_ntg; ++s) { sig += (double)met[2u * s]; err_s += (double)met[2u * s + 1u]; }
                } else {
                    const float* se = (const float*)[buf_se_out contents];
                    for (uint32_t s = 0; s < R / 16; ++s) { sig += se[2 * s]; err_s += se[2 * s + 1]; }
                }
                out_snr = (float)(10.0 * std::log10(sig / std::max(err_s, 1e-18)));
                int sb = snr_bucket(b);
                snr_sum[sb] += out_snr;
                snr_cnt[sb]++;
                uint64_t wpm = tfq4::tfpw_proj_words(R, C, b);
                memcpy(all_scales.data() + s_off, [buf_mat_sc contents], (size_t)N_SC * 2ULL);
                memcpy(all_weights.data() + w_off, [buf_mat_pack contents], wpm * 4ULL);
            };

            for (int e = 0; e < N_EXP; ++e) {
                uint32_t b = metas[e].b_bits;
                // 1) Upload w13 once -> encode gate (even rows) + up (odd rows)
                const uint8_t* w13_c = p_w13 + (size_t)e * w13_code_stride;
                const uint8_t* w13_s = p_s13 + (size_t)e * w13_sc_stride;
                float outer13 = 0.0f;
                if (is_nvfp4) {
                    outer13 = ink2::nvfp4_outer_scale(p_a13[e], p_d13[0], std::string(k_w13) + "[" + std::to_string(e) + "]");
                    ink2::check_finite_e4m3_bytes(w13_s, w13_sc_stride, k_s13);
                } else {
                    ink2::check_finite_e4m3_bytes(w13_c, w13_code_stride, k_w13);
                    ink2::check_finite_e8m0_bytes(w13_s, w13_sc_stride, k_s13);
                }
                memcpy([buf_codes contents],  w13_c, w13_code_stride);
                memcpy([buf_scales contents], w13_s, w13_sc_stride);
                encode_proj((uint32_t)I, (uint32_t)H, b, 0u, 2u, outer13, metas[e].gate_word_off, metas[e].gate_sc_off, metas[e].gate_snr_db);
                encode_proj((uint32_t)I, (uint32_t)H, b, 1u, 2u, outer13, metas[e].up_word_off,   metas[e].up_sc_off,   metas[e].up_snr_db);

                // 2) Upload w2 -> encode down
                const uint8_t* w2_c = p_w2 + (size_t)e * w2_code_stride;
                const uint8_t* w2_s = p_s2 + (size_t)e * w2_sc_stride;
                float outer2 = 0.0f;
                if (is_nvfp4) {
                    outer2 = ink2::nvfp4_outer_scale(p_a2[e], p_d2[0], std::string(k_w2) + "[" + std::to_string(e) + "]");
                    ink2::check_finite_e4m3_bytes(w2_s, w2_sc_stride, k_s2);
                } else {
                    ink2::check_finite_e4m3_bytes(w2_c, w2_code_stride, k_w2);
                    ink2::check_finite_e8m0_bytes(w2_s, w2_sc_stride, k_s2);
                }
                memcpy([buf_codes contents],  w2_c, w2_code_stride);
                memcpy([buf_scales contents], w2_s, w2_sc_stride);
                encode_proj((uint32_t)H, (uint32_t)I, b, 0u, 1u, outer2, metas[e].down_word_off, metas[e].down_sc_off, metas[e].down_snr_db);
            }

            uint64_t off_meta = sizeof(PrismLayerHeader);
            uint64_t off_sc   = ink2::align_up(off_meta + metas.size() * sizeof(PrismExpertMeta), ink2::ALIGN_16K);
            uint64_t off_w    = ink2::align_up(off_sc + all_scales.size() * 2ULL, ink2::ALIGN_16K);
            PrismLayerHeader hdr{};
            hdr.magic = ink2::MAGIC_TFPW; hdr.version = ink2::FORMAT_VER;
            hdr.layer_idx = (uint32_t)lyr; hdr.num_experts = (uint32_t)N_EXP; hdr.num_tensors = (uint32_t)(N_EXP * 3);
            hdr.hidden_size = (uint32_t)H; hdr.moe_inter = (uint32_t)I;
            hdr.reserved0 = tfq_xxh64::FLAG_XXH64_PRESENT;
            hdr.raw_fp8_bytes = (uint64_t)N_EXP * (w13_code_stride + w13_sc_stride + w2_code_stride + w2_sc_stride);
            hdr.stream_words = all_weights.size(); hdr.scale_halfs = all_scales.size();
            hdr.off_meta = off_meta; hdr.off_scales = off_sc; hdr.off_weights = off_w;
            hdr.reserved1[0] = tfq_xxh64::hash(metas.data(), metas.size() * sizeof(PrismExpertMeta));
            hdr.reserved1[1] = tfq_xxh64::hash(all_scales.data(), all_scales.size() * 2ULL);
            hdr.reserved1[2] = tfq_xxh64::hash(all_weights.data(), all_weights.size() * 4ULL);
            hdr.reserved1[3] = 0ULL;
            snprintf((char*)hdr.meta_tag, sizeof(hdr.meta_tag), "ink2_%s_%s", is_nvfp4 ? "nvfp4" : "mxfp8", mode.c_str());

            std::vector<uint8_t> zeros(ink2::ALIGN_16K, 0);
            std::string tmp = std::string(out_path) + ".tmp";
            FILE* fo = fopen(tmp.c_str(), "wb");
            if (!fo) { fprintf(stderr, "Cannot write %s\n", tmp.c_str()); return 1; }
            ink2::write_all_or_exit(fo, &hdr, sizeof(hdr), tmp);
            ink2::write_all_or_exit(fo, metas.data(), metas.size() * sizeof(PrismExpertMeta), tmp);
            ink2::write_all_or_exit(fo, zeros.data(), off_sc - (off_meta + metas.size() * sizeof(PrismExpertMeta)), tmp);
            ink2::write_all_or_exit(fo, all_scales.data(), all_scales.size() * 2ULL, tmp);
            ink2::write_all_or_exit(fo, zeros.data(), off_w - (off_sc + all_scales.size() * 2ULL), tmp);
            ink2::write_all_or_exit(fo, all_weights.data(), all_weights.size() * 4ULL, tmp);
            size_t endp = off_w + all_weights.size() * 4ULL, pade = ink2::align_up(endp, ink2::ALIGN_16K) - endp;
            ink2::write_all_or_exit(fo, zeros.data(), pade, tmp);
            ink2::close_or_exit(fo, tmp);
            ink2::rename_or_exit(tmp, out_path);
            cache.evict_all();

            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_lyr).count();
            double tot = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
            printf("  [L%02d/%02d %s] %.3f GiB |", lyr, l_end, is_nvfp4 ? "NVFP4" : "MXFP8",
                   (double)(endp + pade) / (double)(1ULL << 30));
            const char* lbl[5] = {"2b", "2.25b", "3b", "4b", "5b"};
            for (int sb = 0; sb < 5; ++sb) {
                if (snr_cnt[sb] || sb >= 2) {
                    printf(" %s=%.2f(%d)", lbl[sb], snr_cnt[sb] ? snr_sum[sb] / snr_cnt[sb] : 0.0, snr_cnt[sb] / 3);
                }
            }
            printf(" dB | %.1fs (total %.1fm)\n", dt, tot / 60.0);
            fflush(stdout);
        }

        if (copy_residuals) {
            ink2::export_residuals_and_assets(cache, model_dir, out_dir);
        }
        printf("INKLING-2 METAL DONE -> %s\n", out_dir.c_str());
    }
    return 0;
}
