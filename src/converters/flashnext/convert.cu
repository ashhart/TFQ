// ============================================================================
// TensorFold TFQ3 (2D-AffineCoset/half4) FlashNext CUDA GPU Converter
//   - Source: Qwen3.8-Flash-Next MLX 8-bit/32 (H=2560, I=640, E=512, 48L + MTP)
//   - --mode moe         : Converts switch_mlp -> layer_XX.tffpw / mtp_00.tffpw ("TFPW" v3)
//   - --mode requant-ple : Fast CUDA GPU 8-bit/32 -> B-bit/32 PLE shard builder
// ============================================================================
#include "../cuda_encode_common.cuh"
#include "../moe_heat_rank.hpp"
#include "flashnext_common.hpp"

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

static void write_safetensors(
    const std::string& path,
    const std::vector<std::string>& names,
    const std::vector<std::string>& dtypes,
    const std::vector<std::vector<size_t>>& shapes,
    const std::vector<const void*>& ptrs,
    const std::vector<size_t>& lens
) {
    std::ostringstream hdr;
    hdr << "{";
    uint64_t off = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) hdr << ",";
        hdr << "\"" << names[i] << "\":{\"dtype\":\"" << dtypes[i] << "\",\"shape\":[";
        for (size_t j = 0; j < shapes[i].size(); ++j) { if (j) hdr << ","; hdr << shapes[i][j]; }
        hdr << "],\"data_offsets\":[" << off << "," << (off + lens[i]) << "]}";
        off += lens[i];
    }
    hdr << "}";
    std::string hstr = hdr.str();
    while (hstr.size() % 8 != 0) hstr.push_back(' ');
    uint64_t hlen = hstr.size();
    std::string tmp = path + ".tmp";
    FILE* fo = fopen(tmp.c_str(), "wb");
    if (!fo) { fprintf(stderr, "Cannot write %s\n", tmp.c_str()); exit(1); }
    write_all_or_exit(fo, &hlen, 8, tmp);
    write_all_or_exit(fo, hstr.data(), hlen, tmp);
    for (size_t i = 0; i < names.size(); ++i) write_all_or_exit(fo, ptrs[i], lens[i], tmp);
    close_or_exit(fo, tmp);
    rename_or_exit(tmp, path);
}

int main(int argc, char** argv) {
    std::string model_dir = "", out_dir = "", trace_path = "", tables_path = "";
    std::string mode = "moe", tag = "heat";
    int l_start = 0, l_end = 47, n5 = 0, n4 = 192, ple_bits = 3;
    bool layers_given = false, do_mtp = false, skip_existing = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { model_dir.clear(); break; }
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--trace" && i + 1 < argc) trace_path = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--tables" && i + 1 < argc) tables_path = argv[++i];
        else if (a == "--layers" && i + 2 < argc) { l_start = atoi(argv[++i]); l_end = atoi(argv[++i]); layers_given = true; }
        else if (a == "--n5" && i + 1 < argc) n5 = atoi(argv[++i]);
        else if (a == "--n4" && i + 1 < argc) n4 = atoi(argv[++i]);
        else if (a == "--ple-bits" && i + 1 < argc) ple_bits = atoi(argv[++i]);
        else if (a == "--tag" && i + 1 < argc) tag = argv[++i];
        else if (a == "--mtp") do_mtp = true;
        else if (a == "--skip-existing") skip_existing = true;
    }
    if (model_dir.empty() || out_dir.empty()) {
        fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--mode moe|requant-ple] [--layers S E] [--n5 N] [--n4 N] [--trace <file>] [--mtp] [--ple-bits 3|4] [--skip-existing] [--tables <path>]\n", argv[0]);
        return 1;
    }
    mkdir(out_dir.c_str(), 0755);

    ShardCache cache;
    cache.init(model_dir);

    if (mode == "requant-ple") {
        const uint32_t ROWS = 2500012u, COLS = 160u, NG = COLS / 32u;
        const size_t src_w_bytes = (size_t)ROWS * COLS;
        const size_t sb_bytes = (size_t)ROWS * NG * 2ULL;
        const size_t dst_w_bytes = (size_t)ROWS * NG * (size_t)ple_bits * 4ULL;
        uint8_t *d_sw = nullptr; uint16_t *d_ss = nullptr, *d_sb = nullptr, *d_ds = nullptr, *d_db = nullptr;
        uint32_t *d_dw = nullptr;
        TFQ_CUDA_CHECK(cudaMalloc(&d_sw, src_w_bytes));
        TFQ_CUDA_CHECK(cudaMalloc(&d_ss, sb_bytes));
        TFQ_CUDA_CHECK(cudaMalloc(&d_sb, sb_bytes));
        TFQ_CUDA_CHECK(cudaMalloc(&d_dw, dst_w_bytes));
        TFQ_CUDA_CHECK(cudaMalloc(&d_ds, sb_bytes));
        TFQ_CUDA_CHECK(cudaMalloc(&d_db, sb_bytes));
        auto t0 = std::chrono::steady_clock::now();
        for (int grp = 0; grp < 8; ++grp) {
            std::vector<std::string> names, dtypes;
            std::vector<std::vector<size_t>> shapes;
            std::vector<std::vector<uint8_t>> storage;
            storage.reserve(48);
            for (int k = 0; k < 16; ++k) {
                int sh = grp * 16 + k;
                std::string stem = tfq_flashnext::canonical_ple_stem(sh);
                std::string src_stem = tfq_flashnext::resolve_ple_src_stem(cache, sh);
                size_t lw = 0, ls = 0, lb = 0;
                const uint8_t* pw = cache.get_tensor(src_stem + ".weight", nullptr, &lw);
                const uint8_t* ps = cache.get_tensor(src_stem + ".scales", nullptr, &ls);
                const uint8_t* pb = cache.get_tensor(src_stem + ".biases", nullptr, &lb);
                if (lw != src_w_bytes || ls != sb_bytes || lb != sb_bytes) {
                    fprintf(stderr, "Unexpected PLE shard_%d sizes: %zu %zu %zu\n", sh, lw, ls, lb);
                    return 2;
                }
                TFQ_CUDA_CHECK(cudaMemcpy(d_sw, pw, src_w_bytes, cudaMemcpyHostToDevice));
                TFQ_CUDA_CHECK(cudaMemcpy(d_ss, ps, sb_bytes, cudaMemcpyHostToDevice));
                TFQ_CUDA_CHECK(cudaMemcpy(d_sb, pb, sb_bytes, cudaMemcpyHostToDevice));
                PleParams pp{ROWS, COLS, (uint32_t)ple_bits, 0u};
                uint32_t total_g = ROWS * NG;
                requant_ple_q8_to_qb_kernel<<<(total_g + 255u) / 256u, 256>>>(d_sw, d_ss, d_sb, d_dw, d_ds, d_db, pp);
                TFQ_CUDA_CHECK(cudaDeviceSynchronize());
                storage.emplace_back(dst_w_bytes);
                TFQ_CUDA_CHECK(cudaMemcpy(storage.back().data(), d_dw, dst_w_bytes, cudaMemcpyDeviceToHost));
                names.push_back(stem + ".weight"); dtypes.push_back("U32"); shapes.push_back({ROWS, NG * (size_t)ple_bits});
                storage.emplace_back(sb_bytes);
                TFQ_CUDA_CHECK(cudaMemcpy(storage.back().data(), d_ds, sb_bytes, cudaMemcpyDeviceToHost));
                names.push_back(stem + ".scales"); dtypes.push_back("BF16"); shapes.push_back({ROWS, NG});
                storage.emplace_back(sb_bytes);
                TFQ_CUDA_CHECK(cudaMemcpy(storage.back().data(), d_db, sb_bytes, cudaMemcpyDeviceToHost));
                names.push_back(stem + ".biases"); dtypes.push_back("BF16"); shapes.push_back({ROWS, NG});
            }
            std::vector<const void*> ptrs(storage.size());
            std::vector<size_t> lens(storage.size());
            for (size_t i = 0; i < storage.size(); ++i) { ptrs[i] = storage[i].data(); lens[i] = storage[i].size(); }
            char out_file[512];
            snprintf(out_file, sizeof(out_file), "%s/ple_b%d_%02d.safetensors", out_dir.c_str(), ple_bits, grp);
            write_safetensors(out_file, names, dtypes, shapes, ptrs, lens);
            cache.evict_all();
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("  wrote %s (%.1fs)\n", out_file, el);
        }
        return 0;
    }

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

    constexpr int H = 2560, I = 640, N_EXP = 512;
    constexpr size_t MAX_ELEMS = (size_t)H * I;
    constexpr uint32_t SC_PER_MAT = (H / 128) * (I / 128);
    constexpr uint32_t CHUNK_STR = 512u;
    uint8_t *d_q8w; uint16_t *d_q8s, *d_q8b;
    float *d_tmp, *d_rot, *d_ss, *d_str, *d_se;
    __half *d_sc; uint32_t *d_hist, *d_pack;
    TFQ_CUDA_CHECK(cudaMalloc(&d_q8w, MAX_ELEMS));
    TFQ_CUDA_CHECK(cudaMalloc(&d_q8s, (MAX_ELEMS / 32) * 2));
    TFQ_CUDA_CHECK(cudaMalloc(&d_q8b, (MAX_ELEMS / 32) * 2));
    TFQ_CUDA_CHECK(cudaMalloc(&d_tmp, MAX_ELEMS * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_rot, MAX_ELEMS * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_ss,  (size_t)SC_PER_MAT * 128ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_str, MAX_ELEMS * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_hist, (size_t)CHUNK_STR * (H / 4) * 256ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_pack, (size_t)(I / 16) * (H / 128) * 10 * 32 * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_sc,  SC_PER_MAT * 2));
    TFQ_CUDA_CHECK(cudaMalloc(&d_se,  (size_t)(std::max(H, I) / 16) * 2ULL * 4ULL));

    auto heat_ranks = trace_path.empty() ? std::map<int, std::vector<int>>{} : load_heat_ranks(trace_path, N_EXP);
    const char* proj_names[3] = {"gate_proj", "up_proj", "down_proj"};
    const int proj_R[3] = {I, I, H}, proj_C[3] = {H, H, I};
    std::vector<int> layer_list = tfq_flashnext::build_layer_list(l_start, l_end, layers_given, do_mtp);

    printf("CUDA TFQ3 FLASHNEXT MOE CONVERTER | n5=%d n4=%d n3=%d (avg %.3f bpw) | layers=%zu\n",
           n5, n4, N_EXP - n5 - n4, (5.0 * n5 + 4.0 * n4 + 3.0 * (N_EXP - n5 - n4)) / N_EXP, layer_list.size());

    for (int lyr : layer_list) {
        bool is_mtp = (lyr == 48);
        char out_path[512];
        if (is_mtp) snprintf(out_path, sizeof(out_path), "%s/mtp_00.tffpw", out_dir.c_str());
        else        snprintf(out_path, sizeof(out_path), "%s/layer_%02d.tffpw", out_dir.c_str(), lyr);
        if (skip_existing && tfpw_layer_complete(out_path)) continue;
        auto t_lyr = std::chrono::steady_clock::now();
        const uint8_t *stacked_w[3], *stacked_s[3], *stacked_b[3];
        for (int p = 0; p < 3; ++p) {
            char base[256];
            if (is_mtp) snprintf(base, sizeof(base), "language_model.mtp.layers.0.mlp.switch_mlp.%s", proj_names[p]);
            else        snprintf(base, sizeof(base), "language_model.model.layers.%d.mlp.switch_mlp.%s", lyr, proj_names[p]);
            stacked_w[p] = cache.get_tensor(std::string(base) + ".weight");
            stacked_s[p] = cache.get_tensor(std::string(base) + ".scales");
            stacked_b[p] = cache.get_tensor(std::string(base) + ".biases");
        }
        const size_t exp_sb = (MAX_ELEMS / 32) * 2;
        std::vector<int> ranks(N_EXP);
        bool used_trace = false, used_router = false;
        if (heat_ranks.count(lyr)) {
            ranks = heat_ranks[lyr];
            used_trace = true;
        } else {
            ranks = tfq_moe::rank_flashnext_experts_datafree(cache, lyr, is_mtp, N_EXP, H, stacked_s, exp_sb, &used_router);
        }
        if (lyr == layer_list.front()) {
            printf("HeatAlloc: %s\n", used_trace
                ? ("TRACE-GUIDED (uses calibration routing counts from " + trace_path + ")").c_str()
                : (used_router ? "DATA-FREE router+weight sensitivity (no calibration data)"
                               : "DATA-FREE expert weight-scale sensitivity (no calibration data)"));
            fflush(stdout);
        }

        std::vector<PrismExpertMeta> metas(N_EXP);
        memset(metas.data(), 0, metas.size() * sizeof(PrismExpertMeta));
        uint64_t cur_word = 0; uint32_t cur_sc = 0;
        for (int e = 0; e < N_EXP; ++e) {
            int r = ranks[e];
            uint32_t b = (r < n5) ? 5u : ((r < n5 + n4) ? 4u : 3u);
            metas[e].b_bits = b; metas[e].heat_rank = (uint32_t)r;
            uint64_t wpm = (uint64_t)(I / 16) * (H / 128) * (b * 2u) * 32u;
            metas[e].gate_word_off = cur_word; cur_word += wpm;
            metas[e].up_word_off   = cur_word; cur_word += wpm;
            metas[e].down_word_off = cur_word; cur_word += wpm;
            metas[e].gate_sc_off = cur_sc; cur_sc += SC_PER_MAT;
            metas[e].up_sc_off   = cur_sc; cur_sc += SC_PER_MAT;
            metas[e].down_sc_off = cur_sc; cur_sc += SC_PER_MAT;
        }
        std::vector<uint16_t> all_scales(cur_sc);
        std::vector<uint32_t> all_weights(cur_word);
        double snr_sum[6] = {}; int snr_cnt[6] = {};
        for (int e = 0; e < N_EXP; ++e) {
            uint32_t b = metas[e].b_bits;
            uint64_t wpm = (uint64_t)(I / 16) * (H / 128) * (b * 2u) * 32u;
            for (int p = 0; p < 3; ++p) {
                uint32_t R = proj_R[p], C = proj_C[p];
                const uint8_t* sp_e = stacked_s[p] + (size_t)e * exp_sb;
                const uint8_t* bp_e = stacked_b[p] + (size_t)e * exp_sb;
                char tlabel[128];
                snprintf(tlabel, sizeof(tlabel), "L%d.expert%d.%s", lyr, e, (p == 0 ? "gate" : (p == 1 ? "up" : "down")));
                tfq_common::check_finite_bf16((const uint16_t*)sp_e, R, C / 32u, std::string(tlabel) + ".scales");
                tfq_common::check_finite_bf16((const uint16_t*)bp_e, R, C / 32u, std::string(tlabel) + ".biases");
                TFQ_CUDA_CHECK(cudaMemcpy(d_q8w, stacked_w[p] + (size_t)e * MAX_ELEMS, MAX_ELEMS, cudaMemcpyHostToDevice));
                TFQ_CUDA_CHECK(cudaMemcpy(d_q8s, sp_e, exp_sb, cudaMemcpyHostToDevice));
                TFQ_CUDA_CHECK(cudaMemcpy(d_q8b, bp_e, exp_sb, cudaMemcpyHostToDevice));
                MatParams mp{R, C, b, 1.0f, 0u, 0u};
                q8_32_dequant_row_fwht_kernel<<<dim3(C / 128u, R, 1), 32>>>(d_q8w, d_q8s, d_q8b, d_tmp, mp);
                col_fwht_and_ss_kernel<<<dim3(C, R / 128u, 1), 32>>>(d_tmp, d_rot, d_ss, mp);
                finalize_scales_and_pack_nax_kernel<<<dim3(R / 16u, C / 128u, 1), 32>>>(d_rot, d_ss, d_sc, d_str, mp);
                uint32_t n_str = (R / 16u) * 32u;
                for (uint32_t off = 0; off < n_str; off += CHUNK_STR) {
                    uint32_t cnt = std::min(CHUNK_STR, n_str - off);
                    launch_prism_encode(b, cnt, d_str, d_lut[b], d_c64[b], d_grid[b], d_hist, d_pack, EncParams{R, C, b, off, cnt});
                }
                prism_verify_v3_snr_kernel<<<R / 16u, 32>>>(d_pack, d_sc, d_lut[b], d_c64[b], d_rot, d_se, mp);
                TFQ_CUDA_CHECK(cudaDeviceSynchronize());
                double c_sig = 0.0, c_err = 0.0;
                cuda_read_se_det(d_se, R / 16u, c_sig, c_err);
                float snr = (float)(10.0 * std::log10(c_sig / std::max(c_err, 1e-12)));
                snr_sum[b] += snr; snr_cnt[b]++;
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
        hdr.raw_fp8_bytes = (uint64_t)N_EXP * 3ULL * MAX_ELEMS;
        hdr.stream_words = all_weights.size(); hdr.scale_halfs = all_scales.size();
        hdr.off_meta = off_meta; hdr.off_scales = off_sc; hdr.off_weights = off_w;
        snprintf((char*)hdr.meta_tag, 32, "fn_v3_%s", tag.c_str());
        std::vector<uint8_t> zeros(ALIGN_16K, 0);
        std::string tmp = std::string(out_path) + ".tmp";
        FILE* fo = fopen(tmp.c_str(), "wb");
        if (!fo) { fprintf(stderr, "Cannot write %s\n", tmp.c_str()); return 1; }
        write_all_or_exit(fo, &hdr, sizeof(hdr), tmp);
        write_all_or_exit(fo, metas.data(), metas.size() * sizeof(PrismExpertMeta), tmp);
        write_all_or_exit(fo, zeros.data(), off_sc - (off_meta + metas.size() * sizeof(PrismExpertMeta)), tmp);
        write_all_or_exit(fo, all_scales.data(), all_scales.size() * 2ULL, tmp);
        write_all_or_exit(fo, zeros.data(), off_w - (off_sc + all_scales.size() * 2ULL), tmp);
        write_all_or_exit(fo, all_weights.data(), all_weights.size() * 4ULL, tmp);
        size_t endp = off_w + all_weights.size() * 4ULL, pade = align_up(endp, ALIGN_16K) - endp;
        write_all_or_exit(fo, zeros.data(), pade, tmp);
        close_or_exit(fo, tmp);
        rename_or_exit(tmp, out_path);
        cache.evict_all();
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_lyr).count();
        printf("L%02d -> %s (%.3f GiB) in %.1fs | SNR 3b=%.2f 4b=%.2f 5b=%.2f dB\n",
               lyr, out_path, (double)(endp + pade) / (double)(1ULL << 30), el,
               snr_cnt[3] ? snr_sum[3] / snr_cnt[3] : 0.0,
               snr_cnt[4] ? snr_sum[4] / snr_cnt[4] : 0.0,
               snr_cnt[5] ? snr_sum[5] / snr_cnt[5] : 0.0);
    }
    return 0;
}
