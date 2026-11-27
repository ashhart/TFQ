// ============================================================================
// TFQ4-E8P (2.001 & 2.251 bpw) Model-Agnostic Apple Silicon Metal GPU Converter
//   - Converts any Safetensors checkpoint (BF16/F16/F32/FP8/MLX-Q8, padded [R,C])
//   - Supports table_kind=0/1 (trained 2.001 bpw), table_kind=2 (True QuIP# E8P 2.001 bpw),
//     and table_kind=3 (True QuIP# E8P-1024 2.251 bpw, 9 words/128 cols)
//   - Row-chunked GPU pipeline (4096 rows/chunk) for bounded VRAM on 70B+ models
//   - Automatic 3.4 GiB shard splitting (16 KiB-aligned cross-platform TFP4 v4)
// ============================================================================
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "../generic/planner.hpp"
#include "tfq4_format.hpp"

namespace fs = std::filesystem;

struct Tfq4MatParams { uint32_t R, C, r_off, pad0; };
struct Tfq4ScaleParams { uint32_t n_blocks; float scale_mult; };

static std::string default_metallib(const char* name) {
    uint32_t sz = 0; _NSGetExecutablePath(nullptr, &sz);
    std::string exe(sz, '\0');
    if (_NSGetExecutablePath(&exe[0], &sz) == 0) {
        size_t sl = exe.find_last_of('/');
        if (sl != std::string::npos) return exe.substr(0, sl + 1) + name;
    }
    return name;
}

static std::vector<std::vector<size_t>> partition_tfq4_shards(
    const std::vector<tfq_generic::QuantJob>& jobs, uint32_t b_bits, uint64_t max_shard_bytes
) {
    std::vector<std::vector<size_t>> shards;
    std::vector<size_t> cur;
    uint64_t cur_bytes = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        uint64_t b = tfq4::tfp4_tensor_bytes(jobs[i].pad_R, jobs[i].pad_C, b_bits);
        if (!cur.empty() && cur_bytes + b > max_shard_bytes) {
            shards.push_back(std::move(cur));
            cur.clear();
            cur_bytes = 0;
        }
        cur.push_back(i);
        cur_bytes += b;
    }
    if (!cur.empty()) shards.push_back(std::move(cur));
    return shards;
}

static void print_usage(FILE* fp) {
    fprintf(fp,
        "Usage: tfq4-convert --model DIR --out DIR [--bits 2|2.25] [--head-bits 0|2|2.25]\n"
        "                    [--table-kind trained|e8p|e8p1024] [--max-shard-gib 3.4]\n"
        "                    [--quantize-vision] [--quantize-drafter] [--quantize-embed] [--dry-run]\n");
}

int main(int argc, char** argv) {
    @autoreleasepool {
        tfq_generic::GenericOptions opt;
        opt.default_bits = 3;
        opt.head_bits = 0;
        double max_shard_gib = 3.4;
        uint32_t table_kind = tfq4::PWV4_TABLE_LEGACY_TRAINED;
        uint32_t tfq4_bits = tfq4::TFQ4_BITS_200;
        bool table_kind_set = false;
        bool bits_set = false;
        std::string metallib = default_metallib("tfq4-convert.metallib");
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if ((a == "--model" || a == "-m") && i + 1 < argc) opt.model_dir = argv[++i];
            else if ((a == "--out" || a == "-o") && i + 1 < argc) opt.out_dir = argv[++i];
            else if (a == "--metallib" && i + 1 < argc) metallib = argv[++i];
            else if (a == "--quantize-vision") opt.quantize_vision = true;
            else if (a == "--quantize-drafter") opt.quantize_drafter = true;
            else if (a == "--quantize-embed") opt.quantize_embed = true;
            else if (a == "--bits" && i + 1 < argc) {
                std::string bv = argv[++i];
                bits_set = true;
                if (bv == "2" || bv == "2.0" || bv == "2.00") {
                    tfq4_bits = tfq4::TFQ4_BITS_200;
                } else if (bv == "2.25" || bv == "225") {
                    tfq4_bits = tfq4::TFQ4_BITS_225;
                } else {
                    fprintf(stderr, "tfq4-convert: invalid --bits '%s' (expected 2 or 2.25)\n", bv.c_str());
                    return 2;
                }
            }
            else if (a == "--head-bits" && i + 1 < argc) {
                std::string hv = argv[++i];
                if (hv == "0") opt.head_bits = 0u;
                else if (hv == "2" || hv == "2.0" || hv == "2.25" || hv == "225") opt.head_bits = 2u;
                else {
                    fprintf(stderr, "tfq4-convert: invalid --head-bits '%s' (TFQ4 supports 0 [keep lm_head in residual, default] or 2/2.25)\n", hv.c_str());
                    return 2;
                }
            }
            else if (a == "--max-shard-gib" && i + 1 < argc) max_shard_gib = std::stod(argv[++i]);
            else if (a == "--table-kind" && i + 1 < argc) {
                std::string k = argv[++i];
                table_kind_set = true;
                if (k == "trained" || k == "0") table_kind = tfq4::PWV4_TABLE_LEGACY_TRAINED;
                else if (k == "1") table_kind = tfq4::PWV4_TABLE_TRAINED;
                else if (k == "e8p" || k == "true-e8p" || k == "2") table_kind = tfq4::PWV4_TABLE_TRUE_E8P;
                else if (k == "e8p1024" || k == "e8p-1024" || k == "true-e8p-1024" || k == "3") table_kind = tfq4::PWV4_TABLE_TRUE_E8P_1024;
                else {
                    fprintf(stderr, "tfq4-convert: invalid --table-kind '%s' (expected trained|e8p|e8p1024|0|1|2|3)\n", k.c_str());
                    return 2;
                }
            }
            else if (a == "--dry-run") opt.dry_run = true;
            else if (a == "-h" || a == "--help") {
                print_usage(stdout);
                return 0;
            }
        }
        if (opt.model_dir.empty() || opt.out_dir.empty()) {
            print_usage(stderr);
            return 2;
        }
        if (tfq4_bits == tfq4::TFQ4_BITS_225) {
            if (table_kind_set && table_kind != tfq4::PWV4_TABLE_TRUE_E8P_1024) {
                fprintf(stderr, "tfq4-convert: --bits 2.25 requires --table-kind e8p1024 (table_kind=3)\n");
                return 2;
            }
            table_kind = tfq4::PWV4_TABLE_TRUE_E8P_1024;
        } else if (table_kind == tfq4::PWV4_TABLE_TRUE_E8P_1024) {
            if (bits_set && tfq4_bits != tfq4::TFQ4_BITS_225) {
                fprintf(stderr, "tfq4-convert: --table-kind e8p1024 uses 2.25 bpw (--bits 2.25)\n");
                return 2;
            }
            tfq4_bits = tfq4::TFQ4_BITS_225;
        }
        const bool use_e8p_1024 = (table_kind == tfq4::PWV4_TABLE_TRUE_E8P_1024);
        const bool use_true_e8p = (table_kind == tfq4::PWV4_TABLE_TRUE_E8P);
        const char* rate_lbl = use_e8p_1024 ? "2.251 bpw" : "2.001 bpw";
        const char* fmt_lbl  = use_e8p_1024 ? "TFQ4-E8P-1024" : "TFQ4-E8P";

        uint64_t max_shard_bytes = (uint64_t)std::max(1.0 * (double)(1ULL << 20), max_shard_gib * (double)(1ULL << 30));

        tfq_generic::SafetensorsDir st;
        st.init(opt.model_dir);
        auto plan = tfq_generic::build_plan(st, opt);
        if (plan.jobs.empty()) { fprintf(stderr, "No quantizable 2D float tensors in %s\n", opt.model_dir.c_str()); return 1; }

        uint64_t est_tfq4_bytes = 0;
        for (auto& j : plan.jobs) {
            j.bits = tfq4_bits;
            est_tfq4_bytes += tfq4::tfp4_tensor_bytes(j.pad_R, j.pad_C, tfq4_bits);
        }
        plan.tfq_bytes = est_tfq4_bytes;
        auto shards = partition_tfq4_shards(plan.jobs, tfq4_bits, max_shard_bytes);

        printf("%s (%s) Plan: %zu quantized tensors (%.3f GiB across %zu shard(s)) + %zu residual (%.3f GiB) = %.3f GiB total\n",
               fmt_lbl, rate_lbl, plan.jobs.size(), est_tfq4_bytes / (double)(1ULL << 30), shards.size(),
               plan.residual_names.size(), plan.residual_bytes / (double)(1ULL << 30),
               (est_tfq4_bytes + plan.residual_bytes) / (double)(1ULL << 30));
        if (opt.dry_run) return 0;

        auto t_start = std::chrono::steady_clock::now();
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "No Apple Metal device\n"); return 1; }
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]] error:&err];
        if (!lib) { fprintf(stderr, "metallib error (%s): %s\n", metallib.c_str(), [[err localizedDescription] UTF8String]); return 1; }
        auto mk = [&](const char* nm) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:nm]];
            if (!fn) { fprintf(stderr, "Missing Metal kernel %s\n", nm); std::exit(1); }
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { fprintf(stderr, "PSO error %s: %s\n", nm, [[err localizedDescription] UTF8String]); std::exit(1); }
            return pso;
        };
        id<MTLComputePipelineState> p_rfwht = mk("f32_row_fwht");
        id<MTLComputePipelineState> p_cfwht = mk("col_fwht_and_ss");
        id<MTLComputePipelineState> p_fsc   = mk("tfq4_finalize_scales");
        id<MTLComputePipelineState> p_enc   = mk(use_e8p_1024 ? "tfq4_true_e8p_1024_encode"
                                               : (use_true_e8p ? "tfq4_true_e8p_encode" : "tfq4_e8p_encode"));
        id<MTLCommandQueue> q = [dev newCommandQueue];

        fs::create_directories(opt.out_dir);
        id<MTLBuffer> b_lut = nil;
        {
            std::string tf_path = (fs::path(opt.out_dir) / "pw_v4_e8p_tables.bin").string();
            std::ofstream tf(tf_path, std::ios::binary);
            if (use_e8p_1024) {
                tfq4::PwV4E8Tables1024 tbl1024{};
                tfq4::build_true_e8p_1024_tables(tbl1024);
                tf.write(reinterpret_cast<const char*>(&tbl1024), sizeof(tbl1024));
                tfq_generic::close_or_die(tf, tf_path);
                b_lut = [dev newBufferWithBytes:tbl1024.e8p_lut length:sizeof(tbl1024.e8p_lut) options:MTLResourceStorageModeShared];
            } else {
                tfq4::PwV4E8Tables tbl{};
                if (use_true_e8p) {
                    tfq4::build_true_e8p_tables(tbl);
                } else {
                    tfq4::build_e8p_tables(tbl);
                    tbl.table_kind = table_kind;
                }
                tf.write(reinterpret_cast<const char*>(&tbl), sizeof(tbl));
                tfq_generic::close_or_die(tf, tf_path);
                b_lut = [dev newBufferWithBytes:tbl.e8p_lut length:sizeof(tbl.e8p_lut) options:MTLResourceStorageModeShared];
            }
        }

        uint32_t max_pad_C = 128u;
        for (const auto& j : plan.jobs) max_pad_C = std::max(max_pad_C, j.pad_C);
        const uint32_t MAX_CHUNK_R = 4096u;
        const size_t MAX_ELEMS = (size_t)MAX_CHUNK_R * max_pad_C;
        const size_t MAX_BLK   = (size_t)(MAX_CHUNK_R / 128u) * (max_pad_C / 128u);
        const size_t MAX_WORDS = (size_t)MAX_CHUNK_R * (size_t)tfq4::tfp4_words_per_row(max_pad_C, tfq4_bits);
        const size_t MAX_THREADS = (size_t)MAX_CHUNK_R * (max_pad_C / 16u);
        const size_t MAX_MET_BLKS = (MAX_THREADS + 255ULL) / 256ULL;

        id<MTLBuffer> b_met = [dev newBufferWithLength:MAX_MET_BLKS * 2ULL * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_in  = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_tmp = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> b_rot = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> b_ss  = [dev newBufferWithLength:MAX_BLK * 128ULL * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_sc  = [dev newBufferWithLength:MAX_BLK   * 2ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_out = [dev newBufferWithLength:MAX_WORDS * 4ULL options:MTLResourceStorageModeShared];

        const float scale_mult = use_e8p_1024 ? tfq4::TRUE_E8P1024_BASE_SCALE
                                              : (use_true_e8p ? tfq4::TRUE_E8P_BASE_SCALE : 1.0f);

        std::vector<float> host_chunk;
        std::vector<std::string> shard_names;
        double snr_sum = 0.0;
        uint64_t total_written = 0;

        for (size_t si = 0; si < shards.size(); ++si) {
            const auto& idxs = shards[si];
            std::vector<tfq4::Tfp4TensorMeta> metas;
            metas.reserve(idxs.size());
            std::vector<uint16_t> all_scales;
            std::vector<uint32_t> all_words;

            for (size_t ji : idxs) {
                auto& j = plan.jobs[ji];
                const uint32_t R = j.pad_R, C = j.pad_C;
                const uint32_t n_bc = C / 128u;
                const uint32_t n_sc_tot = (R / 128u) * n_bc;
                const uint64_t wpr = tfq4::tfp4_words_per_row(C, tfq4_bits);
                const uint64_t n_w_tot  = (uint64_t)R * wpr;

                tfq4::Tfp4TensorMeta m{};
                std::strncpy(m.name, j.tfpn_name.c_str(), sizeof(m.name) - 1);
                m.R = R; m.C = C; m.b_bits = tfq4_bits;
                m.sc_off = (uint32_t)all_scales.size();
                m.word_off = (uint64_t)all_words.size();
                m.word_count = n_w_tot;
                m.pad[0] = j.orig_R; m.pad[1] = j.orig_C; m.pad[2] = j.flags;

                all_scales.resize(all_scales.size() + n_sc_tot);
                all_words.resize(all_words.size() + n_w_tot);

                double tot_sig = 0.0, tot_err = 0.0;
                uint32_t sc_cur = 0;
                uint64_t w_cur = 0;
                for (uint32_t r0 = 0; r0 < R; r0 += MAX_CHUNK_R) {
                    uint32_t rc = std::min(MAX_CHUNK_R, R - r0);
                    tfq_generic::decode_chunk_to_f32(st, j, r0, rc, host_chunk);
                    memcpy([b_in contents], host_chunk.data(), host_chunk.size() * 4ULL);

                    uint32_t n_br = rc / 128u, n_blk = n_br * n_bc;
                    uint64_t n_words = (uint64_t)rc * wpr;
                    uint64_t n_threads = (uint64_t)rc * (C / 16u);
                    uint32_t n_mblk = (uint32_t)((n_threads + 255ull) / 256ull);
                    Tfq4MatParams mp{rc, C, r0, 0u};

                    @autoreleasepool {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:p_rfwht];
                        [enc setBuffer:b_in offset:0 atIndex:0]; [enc setBuffer:b_tmp offset:0 atIndex:1];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(n_bc, rc, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:p_cfwht];
                        [enc setBuffer:b_tmp offset:0 atIndex:0]; [enc setBuffer:b_rot offset:0 atIndex:1];
                        [enc setBuffer:b_ss offset:0 atIndex:2];  [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(C, n_br, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        Tfq4ScaleParams sp{n_blk, scale_mult};
                        [enc setComputePipelineState:p_fsc];
                        [enc setBuffer:b_ss offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(n_blk, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:p_enc];
                        [enc setBuffer:b_rot offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
                        [enc setBuffer:b_lut offset:0 atIndex:2]; [enc setBuffer:b_out offset:0 atIndex:3];
                        [enc setBuffer:b_met offset:0 atIndex:4]; [enc setBytes:&mp length:sizeof(mp) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(n_mblk, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }

                    const float* met = (const float*)[b_met contents];
                    for (uint32_t mi = 0; mi < n_mblk; ++mi) {
                        tot_sig += (double)met[(size_t)mi * 2u + 0u];
                        tot_err += (double)met[(size_t)mi * 2u + 1u];
                    }
                    memcpy(all_scales.data() + m.sc_off + sc_cur, [b_sc contents], (size_t)n_blk * 2ULL);
                    memcpy(all_words.data() + m.word_off + w_cur, [b_out contents], (size_t)n_words * 4ULL);
                    sc_cur += n_blk;
                    w_cur += n_words;
                }

                float snr = (tot_sig > 0.0) ? (float)(10.0 * std::log10(tot_sig / std::max(tot_err, 1e-18))) : 0.0f;
                j.snr_db = snr;
                j.bits = tfq4_bits;
                m.snr_db = snr;
                snr_sum += snr;
                metas.push_back(m);
            }

            char fn[96];
            snprintf(fn, sizeof(fn), "tfq4-%05zu-of-%05zu.tffp4", si + 1, shards.size());
            shard_names.push_back(fn);
            uint64_t sz = tfq4::write_tfp4_shard((fs::path(opt.out_dir) / fn).string(), metas, all_scales, all_words);
            total_written += sz;
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            printf("  wrote %s (%zu tensors, %.3f GiB) [%.1fs]\n", fn, metas.size(), sz / (double)(1ULL << 30), dt);
            fflush(stdout);
        }

        plan.tfq_bytes = total_written;
        tfq_generic::package_residuals_and_manifest(
            st, opt, plan, shard_names, fmt_lbl, tfq4::FORMAT_VER, "pw_v4_e8p_tables.bin", "tfq4_manifest.json"
        );
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        printf("%s Metal DONE: %zu tensors -> %.3f GiB (%zu shard(s), %s) | mean SNR=%.2f dB | %.1fs -> %s\n",
               fmt_lbl, plan.jobs.size(), total_written / (double)(1ULL << 30), shards.size(), rate_lbl,
               snr_sum / plan.jobs.size(), secs, opt.out_dir.c_str());
    }
    return 0;
}
