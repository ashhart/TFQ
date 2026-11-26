// ============================================================================
// TensorFold TFQ3 Model-Agnostic Apple Silicon Metal GPU Converter
//   - Converts ANY Safetensors model (BF16, F16, F32, FP8, MLX-8bit)
//   - Zero-pads arbitrary [R, C] shapes cleanly to 128-multiples
//   - Data-free --target-gib / --fit auto sensitivity-based 3b/4b/5b allocator
//   - Optional --quantize-vision, --quantize-drafter, --quantize-embed
// ============================================================================
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>
#include <chrono>
#include "../tfq_tables_io.hpp"
#include "../tfq_xxh64.hpp"
#include "planner.hpp"

using namespace tfq_generic;
using tfq_tables::PwV3TablesFile;

static constexpr uint32_t MAGIC_TFPN = 0x4E504654u;
static constexpr uint32_t FORMAT_VER = 3u;
static constexpr size_t   ALIGN_16K  = 16384;

struct TfpnHeader {
    uint32_t magic, version, num_tensors, pad0;
    uint64_t off_meta, off_scales, off_weights, total_scale_halfs, total_weight_words, pad1;
};
struct TfpnTensorMeta {
    char name[80];
    uint32_t R, C, b_bits, sc_off;
    uint64_t word_off, word_count;
    float snr_db;
    uint32_t pad[3]; // pad[0]=orig_R, pad[1]=orig_C, pad[2]=flags
};

struct MatParams { uint32_t R, C, b_bits; float shrink; uint32_t r_off, pad0; };
struct EncParams { uint32_t R, C, b_bits, stream_off, n_streams; };

static void write_tfpn_shard(const std::string& path, const std::vector<TfpnTensorMeta>& metas,
                             const std::vector<uint16_t>& scales, const std::vector<uint32_t>& weights) {
    const size_t off_meta = sizeof(TfpnHeader);
    const size_t off_scales = (size_t)align16k(off_meta + metas.size() * sizeof(TfpnTensorMeta));
    const size_t off_weights = (size_t)align16k(off_scales + scales.size() * 2ULL);
    TfpnHeader hdr{};
    hdr.magic = MAGIC_TFPN; hdr.version = FORMAT_VER; hdr.num_tensors = (uint32_t)metas.size();
    hdr.pad0 = tfq_xxh64::FLAG_XXH64_PRESENT;
    hdr.off_meta = off_meta; hdr.off_scales = off_scales; hdr.off_weights = off_weights;
    hdr.total_scale_halfs = scales.size(); hdr.total_weight_words = weights.size();
    hdr.pad1 = tfq_xxh64::hash_sections3(
        metas.data(), metas.size() * sizeof(TfpnTensorMeta),
        scales.data(), scales.size() * 2ULL,
        weights.data(), weights.size() * 4ULL
    );
    std::vector<uint8_t> zeros(ALIGN_16K, 0);
    FILE* fout = fopen(path.c_str(), "wb");
    if (!fout) { fprintf(stderr, "Cannot write %s\n", path.c_str()); exit(1); }
    fwrite_or_die(&hdr, sizeof(hdr), fout, path);
    fwrite_or_die(metas.data(), metas.size() * sizeof(TfpnTensorMeta), fout, path);
    size_t p1 = off_scales - (off_meta + metas.size() * sizeof(TfpnTensorMeta));
    fwrite_or_die(zeros.data(), p1, fout, path);
    fwrite_or_die(scales.data(), scales.size() * 2ULL, fout, path);
    size_t p2 = off_weights - (off_scales + scales.size() * 2ULL);
    fwrite_or_die(zeros.data(), p2, fout, path);
    fwrite_or_die(weights.data(), weights.size() * 4ULL, fout, path);
    size_t end_pos = off_weights + weights.size() * 4ULL;
    size_t pe = (size_t)align16k(end_pos) - end_pos;
    fwrite_or_die(zeros.data(), pe, fout, path);
    fclose_or_die(fout, path);
}

static std::string default_metallib(const char* name) {
    uint32_t sz = 0; _NSGetExecutablePath(nullptr, &sz);
    std::string exe(sz, '\0');
    if (_NSGetExecutablePath(&exe[0], &sz) == 0) {
        size_t sl = exe.find_last_of('/');
        if (sl != std::string::npos) return exe.substr(0, sl + 1) + name;
    }
    return name;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        GenericOptions opt;
        std::string metallib = default_metallib("tfq3-generic-convert.metallib");
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--metallib" && i + 1 < argc) metallib = argv[++i];
        }
        if (!parse_generic_args(argc, argv, opt)) {
            fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--profile 4b|3.5b|3b|5b] [--bits 3|4|5] [--head-bits 0|4|5] [--target-gib GB | --fit auto] [--quantize-vision] [--quantize-drafter] [--quantize-embed] [--dry-run] [--tables <path>] [--metallib <path>]\n", argv[0]);
            return 1;
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "No Apple Metal GPU available\n"); return 1; }

        if (opt.fit_auto && opt.target_gib <= 0.0) {
            double ws_gib = (double)dev.recommendedMaxWorkingSetSize / (double)(1ULL << 30);
            double reserve = std::max(4.0, ws_gib * 0.22);
            opt.target_gib = std::max(2.0, ws_gib - reserve);
            printf("Auto-fit Metal GPU (%s): working set %.1f GiB -> weight budget %.2f GiB\n",
                   [[dev name] UTF8String], ws_gib, opt.target_gib);
        }

        SafetensorsDir st;
        st.init(opt.model_dir);
        ConversionPlan plan = build_plan(st, opt);
        print_plan_summary(opt, plan);
        if (plan.jobs.empty()) {
            fprintf(stderr, "No quantizable 2D >=128x128 tensors found in %s\n", opt.model_dir.c_str());
            return 1;
        }
        if (opt.dry_run) return 0;

        mkdir(opt.out_dir.c_str(), 0755);
        std::string dst_tables = opt.out_dir + "/pw_v3_tables.bin";
        bool explicit_tables = !opt.tables_path.empty();
        if (!explicit_tables) opt.tables_path = dst_tables;
        PwV3TablesFile tbl;
        tfq_tables::load_or_write_pwv3_tables(opt.tables_path, tbl, explicit_tables);
        if (opt.tables_path != dst_tables) copy_file_if_exists(opt.tables_path, dst_tables);

        auto grid3 = tfq_tables::build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
        auto grid4 = tfq_tables::build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
        auto grid5 = tfq_tables::build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]] error:&err];
        if (!lib) { fprintf(stderr, "metallib error (%s): %s\n", metallib.c_str(), [[err localizedDescription] UTF8String]); return 1; }
        auto mk_pso = [&](const char* fn) {
            id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
            if (!f) { fprintf(stderr, "Missing Metal kernel %s\n", fn); exit(1); }
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
            if (!p) { fprintf(stderr, "PSO error %s: %s\n", fn, [[err localizedDescription] UTF8String]); exit(1); }
            return p;
        };
        id<MTLComputePipelineState> pso_row  = mk_pso("f32_row_fwht");
        id<MTLComputePipelineState> pso_col  = mk_pso("col_fwht_and_ss");
        id<MTLComputePipelineState> pso_pack = mk_pso("finalize_scales_and_pack_nax");
        id<MTLComputePipelineState> pso_e[6] = {};
        pso_e[3] = mk_pso("prism_encode_v3_b3");
        pso_e[4] = mk_pso("prism_encode_v3_b4");
        pso_e[5] = mk_pso("prism_encode_v3_b5");
        id<MTLComputePipelineState> pso_ver  = mk_pso("prism_verify_v3_snr");
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        id<MTLBuffer> buf_lut[6] = {}, buf_c64[6] = {}, buf_grid[6] = {};
        buf_lut[3]  = [dev newBufferWithBytes:tbl.lut3_u32  length:sizeof(tbl.lut3_u32) options:MTLResourceStorageModeShared];
        buf_lut[4]  = [dev newBufferWithBytes:tbl.lut4_u32  length:sizeof(tbl.lut4_u32) options:MTLResourceStorageModeShared];
        buf_lut[5]  = [dev newBufferWithBytes:tbl.lut5_u32  length:sizeof(tbl.lut5_u32) options:MTLResourceStorageModeShared];
        buf_c64[3]  = [dev newBufferWithBytes:tbl.c64_3     length:sizeof(tbl.c64_3)    options:MTLResourceStorageModeShared];
        buf_c64[4]  = [dev newBufferWithBytes:tbl.c64_4     length:sizeof(tbl.c64_4)    options:MTLResourceStorageModeShared];
        buf_c64[5]  = [dev newBufferWithBytes:tbl.c64_5     length:sizeof(tbl.c64_5)    options:MTLResourceStorageModeShared];
        buf_grid[3] = [dev newBufferWithBytes:grid3.data()  length:grid3.size() * 2     options:MTLResourceStorageModeShared];
        buf_grid[4] = [dev newBufferWithBytes:grid4.data()  length:grid4.size() * 2     options:MTLResourceStorageModeShared];
        buf_grid[5] = [dev newBufferWithBytes:grid5.data()  length:grid5.size() * 2     options:MTLResourceStorageModeShared];

        uint32_t max_pad_C = 128u;
        for (const auto& j : plan.jobs) max_pad_C = std::max(max_pad_C, j.pad_C);
        const uint32_t MAX_CHUNK_R = 4096u;
        const size_t MAX_ELEMS = (size_t)MAX_CHUNK_R * max_pad_C;
        const uint32_t CHUNK_STR = 512u;
        const size_t MAX_HIST = (size_t)CHUNK_STR * (max_pad_C / 4u) * 256ULL * 4ULL;
        const size_t MAX_OUTW = (size_t)(MAX_CHUNK_R / 16u) * (max_pad_C / 128u) * 10ULL * 32ULL * 4ULL;

        id<MTLBuffer> buf_in   = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_tmp  = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_rot  = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_str  = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_ss   = [dev newBufferWithLength:(size_t)(MAX_CHUNK_R / 128u) * (max_pad_C / 128u) * 128ULL * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sc16 = [dev newBufferWithLength:(size_t)(MAX_CHUNK_R / 128u) * (max_pad_C / 128u) * 2ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_hist = [dev newBufferWithLength:MAX_HIST options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_outw = [dev newBufferWithLength:MAX_OUTW options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_se   = [dev newBufferWithLength:(size_t)(MAX_CHUNK_R / 16u) * 2ULL * 4ULL options:MTLResourceStorageModeShared];

        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> host_chunk;
        std::vector<std::string> shard_names;
        double snr_sum[6] = {};
        uint64_t snr_cnt[6] = {};

        for (size_t si = 0; si < plan.shards.size(); ++si) {
            const auto& idxs = plan.shards[si];
            std::vector<TfpnTensorMeta> metas;
            metas.reserve(idxs.size());
            std::vector<uint16_t> all_sc;
            std::vector<uint32_t> all_w;

            for (size_t ji : idxs) {
                auto& j = plan.jobs[ji];
                const float shrink = tfq_common::generic_shrink(j.bits);
                const uint32_t R = j.pad_R, C = j.pad_C, B = j.bits;
                const uint32_t n_sc_tot = (R / 128u) * (C / 128u);
                const uint64_t n_w_tot  = (uint64_t)(R / 16u) * (C / 128u) * (B * 2u) * 32ULL;

                TfpnTensorMeta m{};
                strncpy(m.name, j.tfpn_name.c_str(), 79);
                m.R = R; m.C = C; m.b_bits = B;
                m.sc_off = (uint32_t)all_sc.size();
                m.word_off = all_w.size();
                m.word_count = n_w_tot;
                m.pad[0] = j.orig_R;
                m.pad[1] = j.orig_C;
                m.pad[2] = j.flags;

                all_sc.resize(all_sc.size() + n_sc_tot);
                all_w.resize(all_w.size() + n_w_tot);

                double tot_sig = 0.0, tot_err = 0.0;
                uint32_t sc_cur = 0;
                uint64_t w_cur = 0;
                for (uint32_t r0 = 0; r0 < R; r0 += MAX_CHUNK_R) {
                    uint32_t rc = std::min(MAX_CHUNK_R, R - r0);
                    decode_chunk_to_f32(st, j, r0, rc, host_chunk);
                    memcpy([buf_in contents], host_chunk.data(), host_chunk.size() * 4ULL);

                    uint32_t n_blk = (rc / 128u) * (C / 128u);
                    MatParams mp{rc, C, B, shrink, r0, 0};

                    @autoreleasepool {
                        id<MTLCommandBuffer> cb = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_row];
                        [enc setBuffer:buf_in offset:0 atIndex:0]; [enc setBuffer:buf_tmp offset:0 atIndex:1];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(C / 128u, rc, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:pso_col];
                        [enc setBuffer:buf_tmp offset:0 atIndex:0]; [enc setBuffer:buf_rot offset:0 atIndex:1];
                        [enc setBuffer:buf_ss offset:0 atIndex:2]; [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(C, rc / 128u, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:pso_pack];
                        [enc setBuffer:buf_rot offset:0 atIndex:0]; [enc setBuffer:buf_ss offset:0 atIndex:1];
                        [enc setBuffer:buf_sc16 offset:0 atIndex:2]; [enc setBuffer:buf_str offset:0 atIndex:3];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(rc / 16u, C / 128u, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        uint32_t tot_str = (rc / 16u) * 32u;
                        for (uint32_t s0 = 0; s0 < tot_str; s0 += CHUNK_STR) {
                            uint32_t ns = std::min(CHUNK_STR, tot_str - s0);
                            EncParams ep{rc, C, B, s0, ns};
                            [enc setComputePipelineState:pso_e[B]];
                            [enc setBuffer:buf_str offset:0 atIndex:0];
                            [enc setBuffer:buf_lut[B] offset:0 atIndex:1];
                            [enc setBuffer:buf_c64[B] offset:0 atIndex:2];
                            [enc setBuffer:buf_grid[B] offset:0 atIndex:3];
                            [enc setBuffer:buf_hist offset:0 atIndex:4];
                            [enc setBuffer:buf_outw offset:0 atIndex:5];
                            [enc setBytes:&ep length:sizeof(ep) atIndex:6];
                            [enc dispatchThreadgroups:MTLSizeMake(ns, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        }

                        [enc setComputePipelineState:pso_ver];
                        [enc setBuffer:buf_outw offset:0 atIndex:0]; [enc setBuffer:buf_sc16 offset:0 atIndex:1];
                        [enc setBuffer:buf_lut[B] offset:0 atIndex:2]; [enc setBuffer:buf_c64[B] offset:0 atIndex:3];
                        [enc setBuffer:buf_rot offset:0 atIndex:4]; [enc setBuffer:buf_se offset:0 atIndex:5];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(rc / 16u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }

                    const float* se = (const float*)[buf_se contents];
                    uint32_t n_strips = rc / 16u;
                    for (uint32_t si_s = 0; si_s < n_strips; ++si_s) {
                        tot_sig += (double)se[si_s * 2u + 0u];
                        tot_err += (double)se[si_s * 2u + 1u];
                    }
                    uint64_t cw = (uint64_t)(rc / 16u) * (C / 128u) * (B * 2u) * 32ULL;
                    memcpy(all_sc.data() + m.sc_off + sc_cur, [buf_sc16 contents], n_blk * 2ULL);
                    memcpy(all_w.data() + m.word_off + w_cur, [buf_outw contents], cw * 4ULL);
                    sc_cur += n_blk; w_cur += cw;
                }
                m.snr_db = (tot_sig > 0.0) ? (float)(10.0 * std::log10(tot_sig / std::max(1e-18, tot_err))) : 0.0f;
                j.snr_db = m.snr_db;
                snr_sum[B] += m.snr_db; snr_cnt[B]++;
                metas.push_back(m);
            }

            char fn[96];
            snprintf(fn, sizeof(fn), "tfq3-%05zu-of-%05zu.tffpw", si + 1, plan.shards.size());
            shard_names.push_back(fn);
            write_tfpn_shard(opt.out_dir + "/" + fn, metas, all_sc, all_w);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("  wrote %s (%zu tensors, %.3f GiB) [%.1fs]\n",
                   fn, metas.size(), (double)(all_w.size() * 4ULL + all_sc.size() * 2ULL) / (double)(1ULL << 30), dt);
            fflush(stdout);
        }

        package_residuals_and_manifest(st, opt, plan, shard_names);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("Done in %.1fs | SNR:", dt);
        for (int b = 3; b <= 5; ++b) if (snr_cnt[b]) printf(" %db=%.2fdB(%llu)", b, snr_sum[b] / snr_cnt[b], (unsigned long long)snr_cnt[b]);
        printf("\n");
        return 0;
    }
}
