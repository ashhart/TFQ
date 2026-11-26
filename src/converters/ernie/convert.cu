// ============================================================================
// TensorFold TFQ3 (2D-AffineCoset/half4) ERNIE-5.5-Flash CUDA GPU Converter
//   - Source: baidu/ERNIE-5.5-Flash-Preview-BF16 (BF16 routed experts)
//   - Latent MoE: 512 routed experts per layer in the 2,048-wide latent space
//     (moe_latent_size = moe_intermediate_size = 2048), layers 1..42
//   - Converts block_sparse_moe.experts.E.{w1,w3,w2} -> layer_XX.tffpw ("TFPW" v3)
//     as gate / up / down (the vendor's checkpoint mapping: w1 gate, w3 up, w2 down)
//   - Every other tensor (attention, shared experts, latent projections, router,
//     DSpark head) is left to the caller: copy it from the published checkpoint.
// ============================================================================
#include "../cuda_encode_common.cuh"

int main(int argc, char** argv) {
    std::string model_dir = "", out_dir = "", tables_path = "", tag = "";
    int l_start = 1, l_end = 42, bits = 4;
    bool skip_existing = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { model_dir.clear(); break; }
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--tables" && i + 1 < argc) tables_path = argv[++i];
        else if (a == "--layers" && i + 2 < argc) { l_start = atoi(argv[++i]); l_end = atoi(argv[++i]); }
        else if (a == "--bits" && i + 1 < argc) bits = atoi(argv[++i]);
        else if (a == "--tag" && i + 1 < argc) tag = argv[++i];
        else if (a == "--skip-existing") skip_existing = true;
        else { fprintf(stderr, "Unknown argument: %s\n", a.c_str()); model_dir.clear(); break; }
    }
    if (model_dir.empty() || out_dir.empty() || bits < 3 || bits > 5 || l_start < 1 || l_end > 42 || l_start > l_end) {
        fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--layers S E (1..42)] [--bits 3|4|5] [--tag <label>] [--skip-existing] [--tables <path>]\n", argv[0]);
        return 1;
    }
    if (tag.empty()) tag = "ernie-uniform" + std::to_string(bits);
    mkdir(out_dir.c_str(), 0755);
    const bool explicit_tables = !tables_path.empty();
    if (!explicit_tables) tables_path = out_dir + "/pw_v3_tables.bin";

    PwV3TablesFile tbl;
    load_or_generate_tables(tables_path, tbl, explicit_tables);
    auto grid3 = build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
    auto grid4 = build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
    auto grid5 = build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

    uint32_t *d_lut[6] = {}; uint2 *d_c64[6] = {}; uint16_t *d_grid[6] = {};
    auto up_tbl = [&](int b, const uint32_t* lut, size_t nl, const uint16_t c64[64][4], const std::vector<uint16_t>& gr) {
        TFQ_CUDA_CHECK(cudaMalloc(&d_lut[b], nl * 4)); TFQ_CUDA_CHECK(cudaMemcpy(d_lut[b], lut, nl * 4, cudaMemcpyHostToDevice));
        TFQ_CUDA_CHECK(cudaMalloc(&d_c64[b], 64 * 8)); TFQ_CUDA_CHECK(cudaMemcpy(d_c64[b], c64, 64 * 8, cudaMemcpyHostToDevice));
        TFQ_CUDA_CHECK(cudaMalloc(&d_grid[b], gr.size() * 2)); TFQ_CUDA_CHECK(cudaMemcpy(d_grid[b], gr.data(), gr.size() * 2, cudaMemcpyHostToDevice));
    };
    up_tbl(3, tbl.lut3_u32, 256, tbl.c64_3, grid3);
    up_tbl(4, tbl.lut4_u32, 1024, tbl.c64_4, grid4);
    up_tbl(5, tbl.lut5_u32, 1024, tbl.c64_5, grid5);

    ShardCache cache;
    cache.init(model_dir);

    // MoE Layer Mode ("TFPW" v3, H = latent 2048, I = 2048, E = 512)
    constexpr int H = 2048, I = 2048, N_EXP = 512;
    constexpr size_t MAX_ELEMS = (size_t)H * I;
    constexpr uint32_t SC_PER_MAT = (H / 128) * (I / 128);
    constexpr uint32_t CHUNK_STR = 512u;
    uint16_t* d_bf16; float *d_tmp, *d_rot, *d_ss, *d_str, *d_se;
    __half* d_sc; uint32_t *d_hist, *d_pack;
    TFQ_CUDA_CHECK(cudaMalloc(&d_bf16, MAX_ELEMS * 2));
    TFQ_CUDA_CHECK(cudaMalloc(&d_tmp, MAX_ELEMS * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_rot, MAX_ELEMS * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_ss,  (size_t)SC_PER_MAT * 128ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_str, MAX_ELEMS * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_hist, (size_t)CHUNK_STR * (std::max(H, I) / 4) * 256ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_pack, (size_t)(I / 16) * (H / 128) * 10 * 32 * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_sc,  SC_PER_MAT * 2));
    TFQ_CUDA_CHECK(cudaMalloc(&d_se,  (size_t)(std::max(H, I) / 16) * 2ULL * 4ULL));

    const char* ckpt_names[3] = {"w1", "w3", "w2"};  // gate, up, down
    const int proj_R[3] = {I, I, H}, proj_C[3] = {H, H, I};

    printf("CUDA TFQ3 ERNIE-5.5 MOE CONVERTER | bits=%d | layers %d..%d | tag=%s\n", bits, l_start, l_end, tag.c_str());

    for (int lyr = l_start; lyr <= l_end; ++lyr) {
        char out_path[512]; snprintf(out_path, sizeof(out_path), "%s/layer_%02d.tffpw", out_dir.c_str(), lyr);
        if (skip_existing && tfpw_layer_complete(out_path)) continue;
        auto t_lyr = std::chrono::steady_clock::now();
        std::vector<PrismExpertMeta> metas(N_EXP);
        memset(metas.data(), 0, metas.size() * sizeof(PrismExpertMeta));
        uint64_t cur_word = 0; uint32_t cur_sc = 0;
        const uint32_t b = (uint32_t)bits;
        const uint64_t wpm = (uint64_t)(I / 16) * (H / 128) * (b * 2u) * 32u;
        for (int e = 0; e < N_EXP; ++e) {
            metas[e].b_bits = b; metas[e].heat_rank = (uint32_t)e;
            metas[e].gate_word_off = cur_word; cur_word += wpm;
            metas[e].up_word_off   = cur_word; cur_word += wpm;
            metas[e].down_word_off = cur_word; cur_word += wpm;
            metas[e].gate_sc_off = cur_sc; cur_sc += SC_PER_MAT;
            metas[e].up_sc_off   = cur_sc; cur_sc += SC_PER_MAT;
            metas[e].down_sc_off = cur_sc; cur_sc += SC_PER_MAT;
        }
        std::vector<uint16_t> all_scales(cur_sc);
        std::vector<uint32_t> all_weights(cur_word);
        double snr_sum = 0.0, snr_min = 1e9; int snr_cnt = 0;
        for (int e = 0; e < N_EXP; ++e) {
            for (int p = 0; p < 3; ++p) {
                const uint32_t R = proj_R[p], C = proj_C[p];
                char name[256];
                snprintf(name, sizeof(name), "model.layers.%d.block_sparse_moe.experts.%d.%s.weight", lyr, e, ckpt_names[p]);
                std::string dtype; size_t nbytes = 0; int tR = 0, tC = 0;
                const uint8_t* wp = cache.get_tensor(name, &dtype, &nbytes, &tR, &tC);
                if (dtype != "BF16" || (uint32_t)tR != R || (uint32_t)tC != C || nbytes != (size_t)R * C * 2u) {
                    fprintf(stderr, "Unexpected %s: dtype %s shape [%d, %d] bytes %zu (want BF16 [%u, %u])\n", name, dtype.c_str(), tR, tC, nbytes, R, C);
                    return 1;
                }
                tfq_common::check_finite_bf16(reinterpret_cast<const uint16_t*>(wp), R, C, name);
                TFQ_CUDA_CHECK(cudaMemcpy(d_bf16, wp, nbytes, cudaMemcpyHostToDevice));
                MatParams mp{R, C, b, tfq_common::moe_shrink(b), 0u, 0u};
                bf16_row_fwht_kernel<<<dim3(C / 128u, R, 1), 32>>>(d_bf16, d_tmp, mp);
                col_fwht_and_ss_kernel<<<dim3(C, R / 128u, 1), 32>>>(d_tmp, d_rot, d_ss, mp);
                finalize_scales_and_pack_nax_kernel<<<dim3(R / 16u, C / 128u, 1), 32>>>(d_rot, d_ss, d_sc, d_str, mp);
                const uint32_t n_str = (R / 16u) * 32u;
                for (uint32_t off = 0; off < n_str; off += CHUNK_STR) {
                    uint32_t cnt = std::min(CHUNK_STR, n_str - off);
                    launch_prism_encode(b, cnt, d_str, d_lut[b], d_c64[b], d_grid[b], d_hist, d_pack, EncParams{R, C, b, off, cnt});
                }
                prism_verify_v3_snr_kernel<<<R / 16u, 32>>>(d_pack, d_sc, d_lut[b], d_c64[b], d_rot, d_se, mp);
                TFQ_CUDA_CHECK(cudaGetLastError());
                TFQ_CUDA_CHECK(cudaDeviceSynchronize());
                double c_sig = 0.0, c_err = 0.0;
                cuda_read_se_det(d_se, R / 16u, c_sig, c_err);
                float snr = (float)(10.0 * std::log10(c_sig / std::max(c_err, 1e-12)));
                snr_sum += snr; snr_min = std::min(snr_min, (double)snr); snr_cnt++;
                uint64_t w_off = (p == 0) ? metas[e].gate_word_off : (p == 1 ? metas[e].up_word_off : metas[e].down_word_off);
                uint32_t s_off = (p == 0) ? metas[e].gate_sc_off   : (p == 1 ? metas[e].up_sc_off   : metas[e].down_sc_off);
                if (p == 0) metas[e].gate_snr_db = snr; else if (p == 1) metas[e].up_snr_db = snr; else metas[e].down_snr_db = snr;
                TFQ_CUDA_CHECK(cudaMemcpy(all_weights.data() + w_off, d_pack, wpm * 4, cudaMemcpyDeviceToHost));
                TFQ_CUDA_CHECK(cudaMemcpy(all_scales.data() + s_off, d_sc, SC_PER_MAT * 2, cudaMemcpyDeviceToHost));
            }
        }
        size_t off_meta = sizeof(PrismLayerHeader);
        size_t off_sc   = align_up(off_meta + metas.size() * sizeof(PrismExpertMeta), ALIGN_16K);
        size_t off_w    = align_up(off_sc + all_scales.size() * 2ULL, ALIGN_16K);
        PrismLayerHeader hdr{};
        hdr.magic = MAGIC_TFPW; hdr.version = FORMAT_VER; hdr.layer_idx = (uint32_t)lyr;
        hdr.num_experts = N_EXP; hdr.num_tensors = N_EXP * 3; hdr.hidden_size = H; hdr.moe_inter = I;
        hdr.reserved0 = tfq_xxh64::FLAG_XXH64_PRESENT;
        hdr.raw_fp8_bytes = (uint64_t)N_EXP * 3ULL * MAX_ELEMS;
        hdr.stream_words = all_weights.size(); hdr.scale_halfs = all_scales.size();
        hdr.off_meta = off_meta; hdr.off_scales = off_sc; hdr.off_weights = off_w;
        hdr.reserved1[0] = tfq_xxh64::hash(metas.data(), metas.size() * sizeof(PrismExpertMeta));
        hdr.reserved1[1] = tfq_xxh64::hash(all_scales.data(), all_scales.size() * 2ULL);
        hdr.reserved1[2] = tfq_xxh64::hash(all_weights.data(), all_weights.size() * 4ULL);
        hdr.reserved1[3] = 0ULL;
        strncpy((char*)hdr.meta_tag, tag.c_str(), 31);
        std::vector<uint8_t> zeros(ALIGN_16K, 0);
        std::string tmp = std::string(out_path) + ".tmp";
        FILE* fo = fopen(tmp.c_str(), "wb");
        if (!fo) { fprintf(stderr, "Cannot write %s\n", tmp.c_str()); return 1; }
        size_t wrote = 0;
        wrote += fwrite(&hdr, 1, sizeof(hdr), fo);
        wrote += fwrite(metas.data(), 1, metas.size() * sizeof(PrismExpertMeta), fo);
        if (off_sc > off_meta + metas.size() * sizeof(PrismExpertMeta)) wrote += fwrite(zeros.data(), 1, off_sc - (off_meta + metas.size() * sizeof(PrismExpertMeta)), fo);
        wrote += fwrite(all_scales.data(), 1, all_scales.size() * 2ULL, fo);
        if (off_w > off_sc + all_scales.size() * 2ULL) wrote += fwrite(zeros.data(), 1, off_w - (off_sc + all_scales.size() * 2ULL), fo);
        wrote += fwrite(all_weights.data(), 1, all_weights.size() * 4ULL, fo);
        size_t endp = off_w + all_weights.size() * 4ULL, pade = align_up(endp, ALIGN_16K) - endp;
        if (pade) wrote += fwrite(zeros.data(), 1, pade, fo);
        if (fclose(fo) != 0 || wrote != endp + pade) { fprintf(stderr, "Short write %s (%zu of %zu)\n", tmp.c_str(), wrote, endp + pade); return 1; }
        rename_or_exit(tmp, out_path);
        cache.evict_all();
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_lyr).count();
        printf("L%02d -> %s (%.3f GiB) in %.1fs | SNR %db mean=%.2f min=%.2f dB\n",
               lyr, out_path, (double)(endp + pade) / (double)(1ULL << 30), el, bits,
               snr_cnt ? snr_sum / snr_cnt : 0.0, snr_min);
        fflush(stdout);
    }
    return 0;
}
