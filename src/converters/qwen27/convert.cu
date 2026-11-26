// ============================================================================
// TensorFold TFQ3 (2D-AffineCoset/half4) Qwen3.8-27B Dual-Source CUDA Converter
//   - Accepts BOTH BF16 safetensors (Qwen/Qwen3.8-27B) AND MLX 8-bit/64
//   - Emits deterministic 16 KiB pw_v3_tables.bin + 4 sharded .tffpw files
//     bit-format compatible with both Apple Silicon Metal and NVIDIA CUDA!
// ============================================================================
#include "../cuda_encode_common.cuh"

static void write_tfpn_shard(
    const std::string& path,
    const std::vector<TfpnTensorMeta>& metas,
    const std::vector<uint16_t>& all_scales,
    const std::vector<uint32_t>& all_weights
) {
    const size_t off_meta    = sizeof(TfpnHeader);
    const size_t off_scales  = align_up(off_meta + metas.size() * sizeof(TfpnTensorMeta), ALIGN_16K);
    const size_t off_weights = align_up(off_scales + all_scales.size() * 2ULL, ALIGN_16K);

    TfpnHeader hdr{};
    hdr.magic              = MAGIC_TFPN;
    hdr.version            = FORMAT_VER;
    hdr.num_tensors        = (uint32_t)metas.size();
    hdr.pad0               = tfq_xxh64::FLAG_XXH64_PRESENT;
    hdr.off_meta           = off_meta;
    hdr.off_scales         = off_scales;
    hdr.off_weights        = off_weights;
    hdr.total_scale_halfs  = all_scales.size();
    hdr.total_weight_words = all_weights.size();
    hdr.pad1               = tfq_xxh64::hash_sections3(
        metas.data(), metas.size() * sizeof(TfpnTensorMeta),
        all_scales.data(), all_scales.size() * 2ULL,
        all_weights.data(), all_weights.size() * 4ULL
    );

    std::vector<uint8_t> zeros(ALIGN_16K, 0);
    FILE* fout = fopen(path.c_str(), "wb");
    if (!fout) { fprintf(stderr, "Cannot write %s\n", path.c_str()); exit(1); }
    write_all_or_exit(fout, &hdr, sizeof(hdr), path);
    write_all_or_exit(fout, metas.data(), metas.size() * sizeof(TfpnTensorMeta), path);
    size_t pad1 = off_scales - (off_meta + metas.size() * sizeof(TfpnTensorMeta));
    write_all_or_exit(fout, zeros.data(), pad1, path);
    write_all_or_exit(fout, all_scales.data(), all_scales.size() * 2ULL, path);
    size_t pad2 = off_weights - (off_scales + all_scales.size() * 2ULL);
    write_all_or_exit(fout, zeros.data(), pad2, path);
    write_all_or_exit(fout, all_weights.data(), all_weights.size() * 4ULL, path);
    size_t end_pos = off_weights + all_weights.size() * 4ULL;
    size_t pad_end = align_up(end_pos, ALIGN_16K) - end_pos;
    write_all_or_exit(fout, zeros.data(), pad_end, path);
    close_or_exit(fout, path);
}

int main(int argc, char** argv) {
    std::string model_dir   = "";
    std::string out_dir     = "";
    std::string profile     = "4b"; // 4b | 3.5b | 3b
    std::string tables_path = "";
    uint32_t head_bits      = 4u;
    int num_layers          = 64;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { model_dir.clear(); break; }
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--tables" && i + 1 < argc) tables_path = argv[++i];
        else if (a == "--head-bits" && i + 1 < argc) head_bits = (uint32_t)atoi(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) num_layers = atoi(argv[++i]);
    }
    if (model_dir.empty() || out_dir.empty()) {
        fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--profile 4b|3.5b|3b] [--head-bits 0|4|5] [--layers 64] [--tables <path>]\n", argv[0]);
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
        write_all_or_exit(fo, &tbl, sizeof(tbl), dst_tables);
        close_or_exit(fo, dst_tables);
    }

    auto grid3 = build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
    auto grid4 = build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
    auto grid5 = build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

    ShardCache cache;
    cache.init(model_dir);

    std::string layer_prefix, head_src;
    if (cache.has_tensor("model.language_model.layers.0.mlp.gate_proj.weight")) {
        layer_prefix = "model.language_model."; head_src = "lm_head";
    } else if (cache.has_tensor("language_model.model.layers.0.mlp.gate_proj.weight")) {
        layer_prefix = "language_model.model."; head_src = "language_model.lm_head";
    } else if (cache.has_tensor("model.layers.0.mlp.gate_proj.weight")) {
        layer_prefix = "model."; head_src = "lm_head";
    } else {
        fprintf(stderr, "Could not detect Qwen3.8-27B layer prefix in %s\n", model_dir.c_str());
        return 1;
    }
    std::string sample_dt;
    cache.get_tensor(layer_prefix + "layers.0.mlp.gate_proj.weight", &sample_dt);
    bool src_is_bf16 = (sample_dt == "BF16") && !cache.has_tensor(layer_prefix + "layers.0.mlp.gate_proj.scales");
    printf("CUDA Source: %s | prefix='%s' | head='%s' | format=%s | profile=%s | head_bits=%u\n",
           model_dir.c_str(), layer_prefix.c_str(), head_src.c_str(),
           src_is_bf16 ? "BF16" : "MLX-8bit/64", profile.c_str(), head_bits);

    struct Job { std::string src_mod, dst_name; uint32_t R, C, bits; };
    auto pick_bits = [&](int l, bool is_mlp) -> uint32_t {
        if (profile == "3b") return 3u;
        if (profile == "4b") return 4u;
        if (!is_mlp) return 4u;
        if (l < 6 || l >= num_layers - 6) return 4u;
        return 3u;
    };

    const int n_shards = 4;
    const int l_per_shard = (num_layers + n_shards - 1) / n_shards;
    std::vector<std::vector<Job>> shard_jobs(n_shards);
    for (int l = 0; l < num_layers; ++l) {
        int sidx = std::min(n_shards - 1, l / l_per_shard);
        auto add = [&](const std::string& leaf, uint32_t R, uint32_t C, bool is_mlp) {
            char sp[160], dp[160];
            snprintf(sp, sizeof(sp), "%slayers.%d.%s", layer_prefix.c_str(), l, leaf.c_str());
            snprintf(dp, sizeof(dp), "model.layers.%d.%s", l, leaf.c_str());
            shard_jobs[sidx].push_back({sp, dp, R, C, pick_bits(l, is_mlp)});
        };
        if ((l + 1) % 4 != 0) {
            add("linear_attn.in_proj_qkv", 10240u,  5120u, false);
            add("linear_attn.in_proj_z",    6144u,  5120u, false);
            add("linear_attn.out_proj",     5120u,  6144u, false);
        } else {
            add("self_attn.q_proj",        12288u,  5120u, false);
            add("self_attn.k_proj",         1024u,  5120u, false);
            add("self_attn.v_proj",         1024u,  5120u, false);
            add("self_attn.o_proj",         5120u,  6144u, false);
        }
        add("mlp.gate_proj", 17408u,  5120u, true);
        add("mlp.up_proj",   17408u,  5120u, true);
        add("mlp.down_proj",  5120u, 17408u, true);
    }
    if (head_bits == 3u || head_bits == 4u || head_bits == 5u) {
        shard_jobs[n_shards - 1].push_back({head_src, "lm_head", 248320u, 5120u, head_bits});
    }

    // Upload LUTs, c64, and acceleration grids to GPU once.
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

    // Reusable device scratch (handles up to 17408x5120 or 16384x5120 chunk of lm_head).
    const size_t MAX_ELEMS = 17408ULL * 5120ULL;
    const uint32_t CHUNK_STR = 512u;
    const size_t MAX_HIST = (size_t)CHUNK_STR * (17408ULL / 4ULL) * 256ULL * 4ULL;
    const size_t MAX_OUTW = (17408ULL / 16ULL) * (17408ULL / 128ULL) * 10ULL * 32ULL * 4ULL;

    uint8_t  *d_in_w = nullptr;
    uint16_t *d_in_s = nullptr, *d_in_b = nullptr;
    float    *d_tmp = nullptr, *d_rot = nullptr, *d_streams = nullptr, *d_ss = nullptr, *d_se = nullptr;
    __half   *d_sc16 = nullptr;
    uint32_t *d_hist = nullptr, *d_outw = nullptr;

    TFQ_CUDA_CHECK(cudaMalloc(&d_in_w,    MAX_ELEMS * 2ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_in_s,    (MAX_ELEMS / 64ULL) * 2ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_in_b,    (MAX_ELEMS / 64ULL) * 2ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_tmp,     MAX_ELEMS * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_rot,     MAX_ELEMS * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_streams, MAX_ELEMS * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_ss,      256ULL * 256ULL * 128ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_sc16,    256ULL * 256ULL * 2ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_hist,    MAX_HIST));
    TFQ_CUDA_CHECK(cudaMalloc(&d_outw,    MAX_OUTW));
    TFQ_CUDA_CHECK(cudaMalloc(&d_se,      2048ULL * 2ULL * 4ULL));

    auto t_start = std::chrono::steady_clock::now();
    double snr_sum[6] = {};
    uint64_t snr_cnt[6] = {};
    uint64_t total_tfq_bytes = 0;

    for (int sidx = 0; sidx < n_shards; ++sidx) {
        const auto& jobs = shard_jobs[sidx];
        std::vector<TfpnTensorMeta> metas;
        metas.reserve(jobs.size());
        size_t tot_sc = 0, tot_w = 0;
        for (const auto& j : jobs) {
            tot_sc += (size_t)(j.R / 128u) * (j.C / 128u);
            tot_w  += (size_t)(j.R / 16u) * (j.C / 128u) * (j.bits * 2u) * 32ULL;
        }
        printf("=== Shard %d/%d | %zu tensors | %.3f GiB ===\n",
               sidx + 1, n_shards, jobs.size(), (double)(tot_w * 4ULL + tot_sc * 2ULL) / (double)(1ULL << 30));

        std::vector<uint16_t> all_scales; all_scales.reserve(tot_sc);
        std::vector<uint32_t> all_weights; all_weights.reserve(tot_w);

        for (size_t ji = 0; ji < jobs.size(); ++ji) {
            const auto& j = jobs[ji];
            const uint32_t R = j.R, C = j.C, B = j.bits;
            const uint32_t n_bc = C >> 7, wpb = B * 2u;
            const uint32_t chunk_r = (R > 17408u) ? 16384u : R;
            const float shrink = tfq_common::qwen27_shrink(B);

            const uint8_t* src_w = cache.get_tensor(j.src_mod + ".weight");
            const uint8_t* src_s = src_is_bf16 ? nullptr : cache.get_tensor(j.src_mod + ".scales");
            const uint8_t* src_b = src_is_bf16 ? nullptr : cache.get_tensor(j.src_mod + ".biases");

            TfpnTensorMeta tm{};
            strncpy(tm.name, j.dst_name.c_str(), sizeof(tm.name) - 1);
            tm.R = R; tm.C = C; tm.b_bits = B;
            tm.sc_off     = (uint32_t)all_scales.size();
            tm.word_off   = all_weights.size();
            tm.word_count = (uint64_t)(R / 16u) * n_bc * wpb * 32ULL;

            double sig_tot = 0.0, err_tot = 0.0;
            for (uint32_t r0 = 0; r0 < R; r0 += chunk_r) {
                uint32_t rc = std::min(chunk_r, R - r0);
                uint32_t n_br = rc >> 7;
                uint32_t n_strips = rc >> 4;
                uint32_t n_streams = n_strips * 32u;
                size_t chunk_sc = (size_t)n_br * n_bc;
                size_t chunk_w  = (size_t)n_strips * n_bc * wpb * 32ULL;
                MatParams mp{rc, C, B, shrink, r0, 0u};

                if (src_is_bf16) {
                    const uint16_t* w_bf16 = reinterpret_cast<const uint16_t*>(src_w + (size_t)r0 * C * 2ULL);
                    tfq_common::check_finite_bf16(w_bf16, rc, C, j.src_mod + ".weight", r0);
                    TFQ_CUDA_CHECK(cudaMemcpy(d_in_w, w_bf16, (size_t)rc * C * 2ULL, cudaMemcpyHostToDevice));
                    bf16_row_fwht_kernel<<<dim3(n_bc, rc, 1), 32>>>(reinterpret_cast<const uint16_t*>(d_in_w), d_tmp, mp);
                } else {
                    size_t sb_off = (size_t)r0 * (C / 64u) * 2ULL;
                    size_t sb_len = (size_t)rc * (C / 64u) * 2ULL;
                    tfq_common::check_finite_bf16(reinterpret_cast<const uint16_t*>(src_s + sb_off), rc, C / 64u, j.src_mod + ".scales", r0);
                    tfq_common::check_finite_bf16(reinterpret_cast<const uint16_t*>(src_b + sb_off), rc, C / 64u, j.src_mod + ".biases", r0);
                    TFQ_CUDA_CHECK(cudaMemcpy(d_in_w, src_w + (size_t)r0 * C, (size_t)rc * C, cudaMemcpyHostToDevice));
                    TFQ_CUDA_CHECK(cudaMemcpy(d_in_s, src_s + sb_off, sb_len, cudaMemcpyHostToDevice));
                    TFQ_CUDA_CHECK(cudaMemcpy(d_in_b, src_b + sb_off, sb_len, cudaMemcpyHostToDevice));
                    q8_64_dequant_row_fwht_kernel<<<dim3(n_bc, rc, 1), 32>>>(d_in_w, d_in_s, d_in_b, d_tmp, mp);
                }
                col_fwht_and_ss_kernel<<<dim3(C, n_br, 1), 32>>>(d_tmp, d_rot, d_ss, mp);
                finalize_scales_and_pack_nax_kernel<<<dim3(n_strips, n_bc, 1), 32>>>(d_rot, d_ss, d_sc16, d_streams, mp);

                for (uint32_t s_off = 0; s_off < n_streams; s_off += CHUNK_STR) {
                    uint32_t take = std::min(CHUNK_STR, n_streams - s_off);
                    EncParams ep{rc, C, B, s_off, take};
                    launch_prism_encode(B, take, d_streams, d_lut[B], d_c64[B], d_grid[B], d_hist, d_outw, ep);
                }
                prism_verify_v3_snr_kernel<<<n_strips, 32>>>(d_outw, d_sc16, d_lut[B], d_c64[B], d_rot, d_se, mp);
                TFQ_CUDA_CHECK(cudaDeviceSynchronize());

                double c_sig = 0.0, c_err = 0.0;
                cuda_read_se_det(d_se, n_strips, c_sig, c_err);
                sig_tot += c_sig; err_tot += c_err;

                size_t old_sc = all_scales.size(), old_w = all_weights.size();
                all_scales.resize(old_sc + chunk_sc);
                all_weights.resize(old_w + chunk_w);
                TFQ_CUDA_CHECK(cudaMemcpy(all_scales.data() + old_sc, d_sc16, chunk_sc * 2, cudaMemcpyDeviceToHost));
                TFQ_CUDA_CHECK(cudaMemcpy(all_weights.data() + old_w, d_outw, chunk_w * 4, cudaMemcpyDeviceToHost));
            }
            tm.snr_db = (float)(10.0 * std::log10(sig_tot / std::max(err_tot, 1e-12)));
            snr_sum[B] += tm.snr_db; snr_cnt[B]++;
            metas.push_back(tm);

            if ((ji + 1) % 30 == 0 || ji + 1 == jobs.size()) {
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                printf("  [S%d %3zu/%zu] %-46s [%6u,%5u] %ub SNR=%.2f dB (%.1fs)\n",
                       sidx + 1, ji + 1, jobs.size(), j.dst_name.c_str(), R, C, B, tm.snr_db, el);
            }
        }

        char shard_path[512];
        snprintf(shard_path, sizeof(shard_path), "%s/qwen27_tfq3-%05d-of-%05d.tffpw", out_dir.c_str(), sidx + 1, n_shards);
        write_tfpn_shard(shard_path, metas, all_scales, all_weights);
        struct stat st{}; stat(shard_path, &st);
        total_tfq_bytes += (uint64_t)st.st_size;
        printf("Wrote %s (%.3f GiB)\n", shard_path, (double)st.st_size / (double)(1ULL << 30));
        cache.evict_all();
    }

    double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("DONE profile=%s in %.1fs | TFQ3 bytes=%.3f GiB | Mean SNR: 3b=%.2f dB (n=%llu), 4b=%.2f dB (n=%llu), 5b=%.2f dB (n=%llu)\n",
           profile.c_str(), tot_s, (double)total_tfq_bytes / (double)(1ULL << 30),
           snr_cnt[3] ? snr_sum[3] / snr_cnt[3] : 0.0, (unsigned long long)snr_cnt[3],
           snr_cnt[4] ? snr_sum[4] / snr_cnt[4] : 0.0, (unsigned long long)snr_cnt[4],
           snr_cnt[5] ? snr_sum[5] / snr_cnt[5] : 0.0, (unsigned long long)snr_cnt[5]);
    return 0;
}
