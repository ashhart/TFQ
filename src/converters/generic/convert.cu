// ============================================================================
// TensorFold TFQ3 Model-Agnostic NVIDIA CUDA (sm_80+) GPU Converter
//   - Converts ANY Safetensors model (BF16, F16, F32, FP8, MLX-8bit)
//   - Zero-pads arbitrary [R, C] shapes cleanly to 128-multiples
//   - Data-free --target-gib / --fit auto sensitivity-based 3b/4b/5b allocator
//   - Optional --quantize-vision, --quantize-drafter, --quantize-embed
//   - Produces byte-compatible TFPN v3 shards identical to Apple Silicon Metal
// ============================================================================
#include "../cuda_encode_common.cuh"
#include "planner.hpp"

using namespace tfq_generic;

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

int main(int argc, char** argv) {
    GenericOptions opt;
    if (!parse_generic_args(argc, argv, opt)) {
        fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--profile 4b|3.5b|3b|5b] [--bits 3|4|5] [--head-bits 0|4|5] [--target-gib GB | --fit auto] [--quantize-vision] [--quantize-drafter] [--quantize-embed] [--dry-run] [--tables <path>]\n", argv[0]);
        return 1;
    }

    if (opt.fit_auto && opt.target_gib <= 0.0) {
        size_t free_b = 0, total_b = 0;
        TFQ_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        cudaDeviceProp prop{};
        TFQ_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        double tot_gib = (double)total_b / (double)(1ULL << 30);
        double reserve = std::max(3.0, tot_gib * 0.20);
        opt.target_gib = std::max(2.0, tot_gib - reserve);
        printf("Auto-fit CUDA GPU (%s, sm_%d%d): %.1f GiB VRAM -> weight budget %.2f GiB\n",
               prop.name, prop.major, prop.minor, tot_gib, opt.target_gib);
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
    load_or_generate_tables(opt.tables_path, tbl, explicit_tables);
    if (opt.tables_path != dst_tables) copy_file_if_exists(opt.tables_path, dst_tables);

    auto grid3 = build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
    auto grid4 = build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
    auto grid5 = build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

    uint32_t *d_lut[6] = {};
    uint2    *d_c64[6] = {};
    uint16_t *d_grid[6] = {};
    auto up_tbl = [&](int b, const uint32_t* lut, size_t nl, const uint16_t c64[64][4], const std::vector<uint16_t>& gr) {
        TFQ_CUDA_CHECK(cudaMalloc(&d_lut[b], nl * 4));
        TFQ_CUDA_CHECK(cudaMemcpy(d_lut[b], lut, nl * 4, cudaMemcpyHostToDevice));
        TFQ_CUDA_CHECK(cudaMalloc(&d_c64[b], 64 * 8));
        TFQ_CUDA_CHECK(cudaMemcpy(d_c64[b], c64, 64 * 8, cudaMemcpyHostToDevice));
        TFQ_CUDA_CHECK(cudaMalloc(&d_grid[b], gr.size() * 2));
        TFQ_CUDA_CHECK(cudaMemcpy(d_grid[b], gr.data(), gr.size() * 2, cudaMemcpyHostToDevice));
    };
    up_tbl(3, tbl.lut3_u32,  256, tbl.c64_3, grid3);
    up_tbl(4, tbl.lut4_u32, 1024, tbl.c64_4, grid4);
    up_tbl(5, tbl.lut5_u32, 1024, tbl.c64_5, grid5);

    uint32_t max_pad_C = 128u;
    for (const auto& j : plan.jobs) max_pad_C = std::max(max_pad_C, j.pad_C);
    const uint32_t MAX_CHUNK_R = 4096u;
    const size_t MAX_ELEMS = (size_t)MAX_CHUNK_R * max_pad_C;
    const uint32_t CHUNK_STR = 512u;
    const size_t MAX_HIST = (size_t)CHUNK_STR * (max_pad_C / 4u) * 256ULL * 4ULL;
    const size_t MAX_OUTW = (size_t)(MAX_CHUNK_R / 16u) * (max_pad_C / 128u) * 10ULL * 32ULL * 4ULL;
    const size_t MAX_BLK  = (size_t)(MAX_CHUNK_R / 128u) * (max_pad_C / 128u);

    float *d_in = nullptr, *d_tmp = nullptr, *d_rot = nullptr, *d_str = nullptr, *d_ss = nullptr, *d_se = nullptr;
    __half *d_sc16 = nullptr;
    uint32_t *d_hist = nullptr, *d_outw = nullptr;
    TFQ_CUDA_CHECK(cudaMalloc(&d_in,   MAX_ELEMS * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_tmp,  MAX_ELEMS * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_rot,  MAX_ELEMS * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_str,  MAX_ELEMS * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_ss,   MAX_BLK * 128ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_sc16, MAX_BLK * 2ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_hist, MAX_HIST));
    TFQ_CUDA_CHECK(cudaMalloc(&d_outw, MAX_OUTW));
    TFQ_CUDA_CHECK(cudaMalloc(&d_se,   (MAX_CHUNK_R / 16u) * 2ULL * 4ULL));

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
                TFQ_CUDA_CHECK(cudaMemcpy(d_in, host_chunk.data(), host_chunk.size() * 4ULL, cudaMemcpyHostToDevice));

                uint32_t n_blk = (rc / 128u) * (C / 128u);
                MatParams mp{rc, C, B, shrink, r0, 0};

                f32_row_fwht_kernel<<<dim3(C / 128u, rc), 32>>>(d_in, d_tmp, mp);
                col_fwht_and_ss_kernel<<<dim3(C, rc / 128u), 32>>>(d_tmp, d_rot, d_ss, mp);
                finalize_scales_and_pack_nax_kernel<<<dim3(rc / 16u, C / 128u), 32>>>(d_rot, d_ss, d_sc16, d_str, mp);

                uint32_t tot_str = (rc / 16u) * 32u;
                for (uint32_t s0 = 0; s0 < tot_str; s0 += CHUNK_STR) {
                    uint32_t ns = std::min(CHUNK_STR, tot_str - s0);
                    EncParams ep{rc, C, B, s0, ns};
                    launch_prism_encode(B, ns, d_str, d_lut[B], d_c64[B], d_grid[B], d_hist, d_outw, ep);
                }
                prism_verify_v3_snr_kernel<<<rc / 16u, 32>>>(d_outw, d_sc16, d_lut[B], d_c64[B], d_rot, d_se, mp);
                TFQ_CUDA_CHECK(cudaDeviceSynchronize());

                double c_sig = 0.0, c_err = 0.0;
                cuda_read_se_det(d_se, rc / 16u, c_sig, c_err);
                tot_sig += c_sig; tot_err += c_err;
                uint64_t cw = (uint64_t)(rc / 16u) * (C / 128u) * (B * 2u) * 32ULL;
                TFQ_CUDA_CHECK(cudaMemcpy(all_sc.data() + m.sc_off + sc_cur, d_sc16, n_blk * 2ULL, cudaMemcpyDeviceToHost));
                TFQ_CUDA_CHECK(cudaMemcpy(all_w.data() + m.word_off + w_cur, d_outw, cw * 4ULL, cudaMemcpyDeviceToHost));
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
