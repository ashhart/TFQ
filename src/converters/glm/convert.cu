// ============================================================================
// TensorFold TFQ3 (2D-AffineCoset/half4) GLM-5.3-Flash CUDA GPU Converter
//   - --mode heat543|heat43|heat443|uniform4|uniform3 : FP8 or BF16 MoE -> layer_XX.tffpw ("TFPW" v3)
//     (the routed experts' dtype is read from the checkpoint: F8_E4M3 with weight_scale_inv, or BF16)
//   - --mode nonmoe : 8-bit/64 non-MoE + shared + lm_head -> nonmoe.tffpw ("TFPN" v3)
// ============================================================================
#include "../cuda_encode_common.cuh"
#include "../moe_heat_rank.hpp"
#include "../tfq4/tfq4_format.hpp"
#include "../tfq4/tfq4_encode_common.cuh"

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

int main(int argc, char** argv) {
    std::string model_dir = "", out_dir = "", trace_path = "", tables_path = "", mode = "heat43";
    int l_start = 3, l_end = 44;
    bool skip_existing = false, shared_pw4 = true, fast4b = false, head_pw5 = false, all_pw5 = false, skip_head = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { model_dir.clear(); break; }
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--trace" && i + 1 < argc) trace_path = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
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
        fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--mode heat543|heat43|heat443|uniform4|uniform3|heat432|heat42|heat43225|uniform2|uniform225|nonmoe] [--layers S E] [--trace <file>] [--fast4b] [--shared-pw5] [--head-pw5] [--all-pw5] [--skip-head] [--skip-existing] [--tables <path>]\n", argv[0]);
        return 1;
    }
    if (mode != "heat543" && mode != "heat43" && mode != "heat443" && mode != "uniform4" && mode != "uniform3" &&
        mode != "heat432" && mode != "heat42" && mode != "heat43225" && mode != "uniform2" && mode != "uniform225" &&
        mode != "nonmoe") {
        fprintf(stderr, "Error: unknown --mode %s\n", mode.c_str());
        return 1;
    }
    mkdir(out_dir.c_str(), 0755);
    const bool mode_uses_2b   = (mode == "heat432" || mode == "heat42" || mode == "uniform2");
    const bool mode_uses_225b = (mode == "heat43225" || mode == "uniform225");
    const bool mode_uses_tfq4 = mode_uses_2b || mode_uses_225b;
    const bool explicit_tables = !tables_path.empty();
    if (!explicit_tables) tables_path = out_dir + "/pw_v3_tables.bin";

    PwV3TablesFile tbl;
    load_or_generate_tables(tables_path, tbl, explicit_tables);
    if (mode_uses_tfq4) {
        tfq4::write_hybrid_pwv4_tables(out_dir + "/pw_v4_e8p_tables.bin", mode_uses_225b);
    }
    auto grid3 = build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
    auto grid4 = build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
    auto grid5 = build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

    float h_fp8[256];
    for (int i = 0; i < 256; ++i) h_fp8[i] = fp8_e4m3_to_f32((uint8_t)i);
    float *d_fp8_lut = nullptr;
    TFQ_CUDA_CHECK(cudaMalloc(&d_fp8_lut, 1024));
    TFQ_CUDA_CHECK(cudaMemcpy(d_fp8_lut, h_fp8, 1024, cudaMemcpyHostToDevice));

    uint32_t *d_lut[6] = {}; uint2 *d_c64[6] = {}; uint16_t *d_grid[6] = {};
    auto up_tbl = [&](int b, const uint32_t* lut, size_t nl, const uint16_t c64[64][4], const std::vector<uint16_t>& gr) {
        TFQ_CUDA_CHECK(cudaMalloc(&d_lut[b], nl * 4)); TFQ_CUDA_CHECK(cudaMemcpy(d_lut[b], lut, nl * 4, cudaMemcpyHostToDevice));
        TFQ_CUDA_CHECK(cudaMalloc(&d_c64[b], 64 * 8)); TFQ_CUDA_CHECK(cudaMemcpy(d_c64[b], c64, 64 * 8, cudaMemcpyHostToDevice));
        TFQ_CUDA_CHECK(cudaMalloc(&d_grid[b], gr.size() * 2)); TFQ_CUDA_CHECK(cudaMemcpy(d_grid[b], gr.data(), gr.size() * 2, cudaMemcpyHostToDevice));
    };
    up_tbl(3, tbl.lut3_u32, 256, tbl.c64_3, grid3);
    up_tbl(4, tbl.lut4_u32, 1024, tbl.c64_4, grid4);
    up_tbl(5, tbl.lut5_u32, 1024, tbl.c64_5, grid5);

    __half* d_e8p_lut = nullptr;
    if (mode_uses_225b) {
        tfq4::PwV4E8Tables1024 t4{};
        tfq4::build_true_e8p_1024_tables(t4);
        TFQ_CUDA_CHECK(cudaMalloc(&d_e8p_lut, sizeof(t4.e8p_lut)));
        TFQ_CUDA_CHECK(cudaMemcpy(d_e8p_lut, t4.e8p_lut, sizeof(t4.e8p_lut), cudaMemcpyHostToDevice));
    } else if (mode_uses_2b) {
        tfq4::PwV4E8Tables t4{};
        tfq4::build_true_e8p_tables(t4);
        TFQ_CUDA_CHECK(cudaMalloc(&d_e8p_lut, sizeof(t4.e8p_lut)));
        TFQ_CUDA_CHECK(cudaMemcpy(d_e8p_lut, t4.e8p_lut, sizeof(t4.e8p_lut), cudaMemcpyHostToDevice));
    }

    ShardCache cache;
    cache.init(model_dir);
    constexpr uint32_t CHUNK_STR = 512u;

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
                for (const char* s : {"q_proj", "k_proj", "v_proj"}) {
                    snprintf(p, sizeof(p), "model.language_model.layers.%d.self_attn.%s", l, s);
                    tasks.push_back({p, 8192, 4096, nb_qkv});
                }
                snprintf(p, sizeof(p), "model.language_model.layers.%d.self_attn.o_proj", l);
                tasks.push_back({p, 4096, 8192, nb_proj});
            } else {
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
                snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.gate_proj", l); tasks.push_back({p, 12288, 4096, nb_proj});
                snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.up_proj",   l); tasks.push_back({p, 12288, 4096, nb_proj});
                snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.down_proj", l); tasks.push_back({p, 4096, 12288, nb_proj});
            } else if ((fast4b || all_pw5) && l >= 3 && l < 7) {
                snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.gate_proj", l); tasks.push_back({p, 2048, 4096, 5});
                snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.up_proj",   l); tasks.push_back({p, 2048, 4096, 5});
                snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.down_proj", l); tasks.push_back({p, 4096, 2048, 5});
            } else if (l >= 7) {
                uint32_t sb = shared_pw4 ? 4u : 5u;
                snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.gate_proj", l); tasks.push_back({p, 2048, 4096, sb});
                snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.up_proj",   l); tasks.push_back({p, 2048, 4096, sb});
                snprintf(p, sizeof(p), "model.language_model.layers.%d.mlp.shared_experts.down_proj", l); tasks.push_back({p, 4096, 2048, sb});
            }
        }
        if (!skip_head) tasks.push_back({"lm_head", 154880, 4096, nb_head});

        const size_t MAX_E = 16384ULL * 4096ULL;
        uint8_t *d_q8w; uint16_t *d_q8s, *d_q8b; float *d_tmp, *d_rot, *d_ss, *d_str, *d_se;
        __half *d_sc; uint32_t *d_hist, *d_pack;
        TFQ_CUDA_CHECK(cudaMalloc(&d_q8w, MAX_E));
        TFQ_CUDA_CHECK(cudaMalloc(&d_q8s, (MAX_E / 64) * 2));
        TFQ_CUDA_CHECK(cudaMalloc(&d_q8b, (MAX_E / 64) * 2));
        TFQ_CUDA_CHECK(cudaMalloc(&d_tmp, MAX_E * 4));
        TFQ_CUDA_CHECK(cudaMalloc(&d_rot, MAX_E * 4));
        TFQ_CUDA_CHECK(cudaMalloc(&d_ss,  8192ULL * 128ULL * 4ULL));
        TFQ_CUDA_CHECK(cudaMalloc(&d_str, MAX_E * 4));
        TFQ_CUDA_CHECK(cudaMalloc(&d_hist, (size_t)CHUNK_STR * 4096ULL * 256ULL * 4ULL));
        TFQ_CUDA_CHECK(cudaMalloc(&d_pack, MAX_E));
        TFQ_CUDA_CHECK(cudaMalloc(&d_sc,  8192 * 2));
        TFQ_CUDA_CHECK(cudaMalloc(&d_se,  2048ULL * 2ULL * 4ULL));

        std::vector<TfpnTensorMeta> metas(tasks.size());
        uint64_t total_sc = 0, total_wd = 0;
        for (size_t i = 0; i < tasks.size(); ++i) {
            memset(&metas[i], 0, sizeof(TfpnTensorMeta));
            strncpy(metas[i].name, tasks[i].name.c_str(), 79);
            metas[i].R = tasks[i].R; metas[i].C = tasks[i].C; metas[i].b_bits = tasks[i].bits;
            metas[i].sc_off = (uint32_t)total_sc; total_sc += (uint64_t)(tasks[i].R / 128) * (tasks[i].C / 128);
            metas[i].word_off = total_wd; metas[i].word_count = (uint64_t)(tasks[i].R / 16) * (tasks[i].C / 128) * (tasks[i].bits * 2u) * 32u;
            total_wd += metas[i].word_count;
        }
        std::vector<uint16_t> all_scales(total_sc);
        std::vector<uint32_t> all_weights(total_wd);

        for (size_t i = 0; i < tasks.size(); ++i) {
            const auto& tk = tasks[i];
            const uint8_t* sw = cache.get_tensor(tk.name + ".weight");
            const uint8_t* ss = cache.get_tensor(tk.name + ".scales");
            const uint8_t* sb = cache.get_tensor(tk.name + ".biases");
            uint32_t chunk_r = (tk.R > 16384u) ? 8192u : tk.R;
            double sig_t = 0.0, err_t = 0.0;
            uint64_t w_cur = metas[i].word_off; uint32_t s_cur = metas[i].sc_off;
            for (uint32_t r0 = 0; r0 < tk.R; r0 += chunk_r) {
                uint32_t rc = std::min(chunk_r, tk.R - r0);
                size_t sb_off = (size_t)r0 * (tk.C / 64u) * 2ULL, sb_len = (size_t)rc * (tk.C / 64u) * 2ULL;
                size_t nsc = (size_t)(rc / 128u) * (tk.C / 128u);
                size_t nwd = (size_t)(rc / 16u) * (tk.C / 128u) * (tk.bits * 2u) * 32ULL;
                TFQ_CUDA_CHECK(cudaMemcpy(d_q8w, sw + (size_t)r0 * tk.C, (size_t)rc * tk.C, cudaMemcpyHostToDevice));
                TFQ_CUDA_CHECK(cudaMemcpy(d_q8s, ss + sb_off, sb_len, cudaMemcpyHostToDevice));
                TFQ_CUDA_CHECK(cudaMemcpy(d_q8b, sb + sb_off, sb_len, cudaMemcpyHostToDevice));
                MatParams mp{rc, tk.C, tk.bits, 1.0f, r0, 0u};
                q8_64_dequant_row_fwht_kernel<<<dim3(tk.C / 128u, rc, 1), 32>>>(d_q8w, d_q8s, d_q8b, d_tmp, mp);
                col_fwht_and_ss_kernel<<<dim3(tk.C, rc / 128u, 1), 32>>>(d_tmp, d_rot, d_ss, mp);
                finalize_scales_and_pack_nax_kernel<<<dim3(rc / 16u, tk.C / 128u, 1), 32>>>(d_rot, d_ss, d_sc, d_str, mp);
                uint32_t n_str = (rc / 16u) * 32u;
                for (uint32_t off = 0; off < n_str; off += CHUNK_STR) {
                    uint32_t cnt = std::min(CHUNK_STR, n_str - off);
                    launch_prism_encode(tk.bits, cnt, d_str, d_lut[tk.bits], d_c64[tk.bits], d_grid[tk.bits], d_hist, d_pack, EncParams{rc, tk.C, tk.bits, off, cnt});
                }
                prism_verify_v3_snr_kernel<<<rc / 16u, 32>>>(d_pack, d_sc, d_lut[tk.bits], d_c64[tk.bits], d_rot, d_se, mp);
                TFQ_CUDA_CHECK(cudaDeviceSynchronize());
                double c_sig = 0.0, c_err = 0.0;
                cuda_read_se_det(d_se, rc / 16u, c_sig, c_err);
                sig_t += c_sig; err_t += c_err;
                TFQ_CUDA_CHECK(cudaMemcpy(all_scales.data() + s_cur, d_sc, nsc * 2, cudaMemcpyDeviceToHost));
                TFQ_CUDA_CHECK(cudaMemcpy(all_weights.data() + w_cur, d_pack, nwd * 4, cudaMemcpyDeviceToHost));
                s_cur += (uint32_t)nsc; w_cur += nwd;
            }
            metas[i].snr_db = (float)(10.0 * std::log10(sig_t / std::max(err_t, 1e-12)));
        }
        size_t off_meta = sizeof(TfpnHeader);
        size_t off_sc = align_up(off_meta + metas.size() * sizeof(TfpnTensorMeta), ALIGN_16K);
        size_t off_w  = align_up(off_sc + all_scales.size() * 2ULL, ALIGN_16K);
        uint64_t xxh = tfq_xxh64::hash_sections3(
            metas.data(), metas.size() * sizeof(TfpnTensorMeta),
            all_scales.data(), all_scales.size() * 2ULL,
            all_weights.data(), all_weights.size() * 4ULL
        );
        TfpnHeader hdr{MAGIC_TFPN, FORMAT_VER, (uint32_t)metas.size(), tfq_xxh64::FLAG_XXH64_PRESENT, off_meta, off_sc, off_w, all_scales.size(), all_weights.size(), xxh};
        std::vector<uint8_t> zeros(ALIGN_16K, 0);
        std::string out_path = out_dir + "/nonmoe.tffpw", tmp = out_path + ".tmp";
        FILE* fo = fopen(tmp.c_str(), "wb");
        if (!fo) { fprintf(stderr, "Cannot write %s\n", tmp.c_str()); return 1; }
        write_all_or_exit(fo, &hdr, sizeof(hdr), tmp);
        write_all_or_exit(fo, metas.data(), metas.size() * sizeof(TfpnTensorMeta), tmp);
        write_all_or_exit(fo, zeros.data(), off_sc - (off_meta + metas.size() * sizeof(TfpnTensorMeta)), tmp);
        write_all_or_exit(fo, all_scales.data(), all_scales.size() * 2ULL, tmp);
        write_all_or_exit(fo, zeros.data(), off_w - (off_sc + all_scales.size() * 2ULL), tmp);
        write_all_or_exit(fo, all_weights.data(), all_weights.size() * 4ULL, tmp);
        size_t endp = off_w + all_weights.size() * 4ULL, pade = align_up(endp, ALIGN_16K) - endp;
        write_all_or_exit(fo, zeros.data(), pade, tmp);
        close_or_exit(fo, tmp);
        rename_or_exit(tmp, out_path);
        printf("NON-MOE DONE -> %s (%.3f GiB)\n", out_path.c_str(), (double)(endp + pade) / (double)(1ULL << 30));
        return 0;
    }

    // MoE Layer Mode ("TFPW" v3): auto-detect GLM-5.3-Flash vs GLM-5.3 753B (glm_moe_dsa)
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
    const size_t MAX_E = (size_t)H * (size_t)I;
    uint8_t *d_fp8 = nullptr; uint16_t *d_bf16 = nullptr; float *d_scinv = nullptr, *d_tmp, *d_rot, *d_ss, *d_str, *d_se;
    __half *d_sc; uint32_t *d_hist, *d_pack;
    if (src_bf16) {
        TFQ_CUDA_CHECK(cudaMalloc(&d_bf16, MAX_E * 2ULL));
    } else {
        TFQ_CUDA_CHECK(cudaMalloc(&d_fp8, MAX_E));
        TFQ_CUDA_CHECK(cudaMalloc(&d_scinv, (size_t)N_SC * 4ULL));
    }
    TFQ_CUDA_CHECK(cudaMalloc(&d_tmp, MAX_E * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_rot, MAX_E * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_ss,  (size_t)N_SC * 128ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_str, MAX_E * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_hist, (size_t)CHUNK_STR * (size_t)(H / 4) * 256ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_pack, (size_t)(I / 16) * (size_t)(H / 128) * 10ULL * 32ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_sc,  (size_t)N_SC * 2ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_se,  (size_t)(std::max(H, I) / 16) * 2ULL * 4ULL));
    const uint32_t MAX_TFQ4_TG = (uint32_t)((MAX_E / 16ULL + 255ULL) / 256ULL);
    float* d_tfq4_met = nullptr;
    std::vector<float> h_tfq4_met;
    if (mode_uses_tfq4) {
        TFQ_CUDA_CHECK(cudaMalloc(&d_tfq4_met, (size_t)MAX_TFQ4_TG * 2ULL * 4ULL));
        h_tfq4_met.resize((size_t)MAX_TFQ4_TG * 2ULL);
    }

    auto heat_ranks = trace_path.empty() ? std::map<int, std::vector<int>>{} : load_heat_ranks(trace_path, N_EXP);
    const char* proj_names[3] = {"gate_proj", "up_proj", "down_proj"};
    const int proj_R[3] = {I, I, H}, proj_C[3] = {H, H, I};

    for (int lyr = l_start; lyr <= l_end; ++lyr) {
        char out_path[512]; snprintf(out_path, sizeof(out_path), "%s/layer_%02d.tffpw", out_dir.c_str(), lyr);
        if (skip_existing && tfpw_layer_complete(out_path)) continue;
        auto t_lyr = std::chrono::steady_clock::now();
        std::vector<int> ranks(N_EXP);
        bool used_trace = false, used_router = false;
        if (heat_ranks.count(lyr)) {
            ranks = heat_ranks[lyr];
            used_trace = true;
        } else {
            ranks = tfq_moe::rank_glm_experts_datafree(cache, lyr, N_EXP, H, &used_router, layer_prefix.c_str());
        }
        if (lyr == l_start) {
            // A BF16 checkpoint has no weight_scale_inv, so its data-free rank has no weight term.
            const char* datafree = src_bf16
                ? (used_router ? "DATA-FREE router sensitivity (BF16 source: no weight_scale_inv for the weight term)"
                               : "none (BF16 source without a router: index order)")
                : (used_router ? "DATA-FREE router+weight sensitivity (no calibration data)"
                               : "DATA-FREE expert weight-scale sensitivity (no calibration data)");
            printf("GLM MoE CUDA | H=%d I=%d E=%d prefix=%s src=%s | HeatAlloc: %s\n",
                H, I, N_EXP, layer_prefix.c_str(), src_dtype.c_str(),
                used_trace
                ? ("TRACE-GUIDED (uses calibration routing counts from " + trace_path + ")").c_str()
                : datafree);
            fflush(stdout);
        }
        std::vector<PrismExpertMeta> metas(N_EXP);
        memset(metas.data(), 0, metas.size() * sizeof(PrismExpertMeta));
        const int thr5    = (N_EXP == 288) ? 64  : (N_EXP / 4);
        const int thr4    = (N_EXP == 288) ? 160 : ((N_EXP * 5) / 8);
        const int thr43   = (N_EXP == 288) ? 96  : (N_EXP / 3);
        const int thr4_15 = (N_EXP * 15 + 50) / 100;
        const int thr3_40 = (N_EXP * 40 + 50) / 100;
        const int thr4_20 = (N_EXP * 20 + 50) / 100;
        uint64_t cur_word = 0; uint32_t cur_sc = 0;
        for (int e = 0; e < N_EXP; ++e) {
            int r = ranks[e];
            uint32_t b = 4u;
            if (mode == "heat543")         b = (r < thr5) ? 5u : (r < thr4 ? 4u : 3u);
            else if (mode == "heat443")    b = (r < thr4) ? 4u : 3u;
            else if (mode == "heat43")     b = (r < thr43) ? 4u : 3u;
            else if (mode == "uniform4")   b = 4u;
            else if (mode == "uniform3")   b = 3u;
            else if (mode == "heat432")    b = (r < thr4_15) ? 4u : ((r < thr3_40) ? 3u : 2u);
            else if (mode == "heat42")     b = (r < thr4_20) ? 4u : 2u;
            else if (mode == "heat43225")  b = (r < thr4_15) ? 4u : ((r < thr3_40) ? 3u : 225u);
            else if (mode == "uniform2")   b = 2u;
            else if (mode == "uniform225") b = 225u;
            metas[e].b_bits = b; metas[e].heat_rank = (uint32_t)r;
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
        for (int e = 0; e < N_EXP; ++e) {
            uint32_t b = metas[e].b_bits;
            const bool is_tfq4 = (b == tfq4::TFQ4_BITS_200 || b == tfq4::TFQ4_BITS_225);
            uint64_t wpm = tfq4::tfpw_proj_words((uint32_t)I, (uint32_t)H, b);
            for (int p = 0; p < 3; ++p) {
                uint32_t R = proj_R[p], C = proj_C[p];
                char base[256]; snprintf(base, sizeof(base), "%s.%d.mlp.experts.%d.%s", layer_prefix.c_str(), lyr, e, proj_names[p]);
                MatParams mp{R, C, b, 1.0f, 0u, 0u};
                if (src_bf16) {
                    std::string dtype; size_t nw = 0; int tR = 0, tC = 0;
                    const uint8_t* wp = cache.get_tensor(std::string(base) + ".weight", &dtype, &nw, &tR, &tC);
                    if (dtype != "BF16" || (uint32_t)tR != R || (uint32_t)tC != C || nw != (size_t)R * C * 2u) {
                        fprintf(stderr, "Unexpected %s.weight: dtype %s shape [%d, %d] bytes %zu (want BF16 [%u, %u])\n",
                            base, dtype.c_str(), tR, tC, nw, R, C);
                        return 1;
                    }
                    tfq_common::check_finite_bf16(reinterpret_cast<const uint16_t*>(wp), R, C, std::string(base) + ".weight");
                    TFQ_CUDA_CHECK(cudaMemcpy(d_bf16, wp, nw, cudaMemcpyHostToDevice));
                    bf16_row_fwht_kernel<<<dim3(C / 128u, R, 1), 32>>>(d_bf16, d_tmp, mp);
                } else {
                    size_t nw = 0, ns = 0;
                    const uint8_t* wp = cache.get_tensor(std::string(base) + ".weight", nullptr, &nw);
                    const uint8_t* sp = cache.get_tensor(std::string(base) + ".weight_scale_inv", nullptr, &ns);
                    tfq_common::check_finite_fp8_e4m3(wp, R, C, std::string(base) + ".weight");
                    tfq_common::check_finite_f32((const float*)sp, R / 128u, C / 128u, std::string(base) + ".weight_scale_inv");
                    TFQ_CUDA_CHECK(cudaMemcpy(d_fp8, wp, nw, cudaMemcpyHostToDevice));
                    TFQ_CUDA_CHECK(cudaMemcpy(d_scinv, sp, ns, cudaMemcpyHostToDevice));
                    fp8_dequant_row_fwht_kernel<<<dim3(C / 128u, R, 1), 32>>>(d_fp8, d_scinv, d_fp8_lut, d_tmp, mp);
                }
                col_fwht_and_ss_kernel<<<dim3(C, R / 128u, 1), 32>>>(d_tmp, d_rot, d_ss, mp);
                double c_sig = 0.0, c_err = 0.0;
                if (is_tfq4) {
                    float sm = (b == tfq4::TFQ4_BITS_225) ? tfq4::TRUE_E8P1024_BASE_SCALE : tfq4::TRUE_E8P_BASE_SCALE;
                    uint32_t tfq4_ntg = (uint32_t)((((uint64_t)R * (C / 16u)) + 255ull) / 256ull);
                    tfq4_enc::tfq4_finalize_scales_k<<<N_SC, 32>>>(d_ss, d_sc, N_SC, sm);
                    if (b == tfq4::TFQ4_BITS_225) {
                        tfq4_enc::tfq4_true_e8p_1024_encode_k<<<tfq4_ntg, 256>>>(
                            d_rot, d_sc, reinterpret_cast<const uint4*>(d_e8p_lut), d_pack, d_tfq4_met, R, C);
                    } else {
                        tfq4_enc::tfq4_true_e8p_encode_k<<<tfq4_ntg, 256>>>(
                            d_rot, d_sc, d_e8p_lut, d_pack, d_tfq4_met, R, C);
                    }
                    TFQ_CUDA_CHECK(cudaDeviceSynchronize());
                    TFQ_CUDA_CHECK(cudaMemcpy(h_tfq4_met.data(), d_tfq4_met, (size_t)tfq4_ntg * 2ULL * 4ULL, cudaMemcpyDeviceToHost));
                    for (uint32_t s = 0; s < tfq4_ntg; ++s) { c_sig += (double)h_tfq4_met[2u * s]; c_err += (double)h_tfq4_met[2u * s + 1u]; }
                } else {
                    finalize_scales_and_pack_nax_kernel<<<dim3(R / 16u, C / 128u, 1), 32>>>(d_rot, d_ss, d_sc, d_str, mp);
                    uint32_t n_str = (R / 16u) * 32u;
                    for (uint32_t off = 0; off < n_str; off += CHUNK_STR) {
                        uint32_t cnt = std::min(CHUNK_STR, n_str - off);
                        launch_prism_encode(b, cnt, d_str, d_lut[b], d_c64[b], d_grid[b], d_hist, d_pack, EncParams{R, C, b, off, cnt});
                    }
                    prism_verify_v3_snr_kernel<<<R / 16u, 32>>>(d_pack, d_sc, d_lut[b], d_c64[b], d_rot, d_se, mp);
                    TFQ_CUDA_CHECK(cudaDeviceSynchronize());
                    cuda_read_se_det(d_se, R / 16u, c_sig, c_err);
                }
                float snr = (float)(10.0 * std::log10(c_sig / std::max(c_err, 1e-12)));
                uint64_t w_off = (p == 0) ? metas[e].gate_word_off : (p == 1 ? metas[e].up_word_off : metas[e].down_word_off);
                uint32_t s_off = (p == 0) ? metas[e].gate_sc_off   : (p == 1 ? metas[e].up_sc_off   : metas[e].down_sc_off);
                if (p == 0) metas[e].gate_snr_db = snr; else if (p == 1) metas[e].up_snr_db = snr; else metas[e].down_snr_db = snr;
                TFQ_CUDA_CHECK(cudaMemcpy(all_weights.data() + w_off, d_pack, wpm * 4, cudaMemcpyDeviceToHost));
                TFQ_CUDA_CHECK(cudaMemcpy(all_scales.data() + s_off, d_sc, (size_t)N_SC * 2ULL, cudaMemcpyDeviceToHost));
            }
        }
        cache.evict_all();
        size_t off_meta = sizeof(PrismLayerHeader);
        size_t off_sc = align_up(off_meta + metas.size() * sizeof(PrismExpertMeta), ALIGN_16K);
        size_t off_w  = align_up(off_sc + all_scales.size() * 2ULL, ALIGN_16K);
        PrismLayerHeader hdr{};
        hdr.magic = MAGIC_TFPW; hdr.version = FORMAT_VER; hdr.layer_idx = (uint32_t)lyr;
        hdr.num_experts = N_EXP; hdr.num_tensors = N_EXP * 3; hdr.hidden_size = H; hdr.moe_inter = I;
        hdr.reserved0 = tfq_xxh64::FLAG_XXH64_PRESENT;
        hdr.raw_fp8_bytes = (uint64_t)N_EXP * 3ULL * MAX_E;
        hdr.stream_words = all_weights.size(); hdr.scale_halfs = all_scales.size();
        hdr.off_meta = off_meta; hdr.off_scales = off_sc; hdr.off_weights = off_w;
        hdr.reserved1[0] = tfq_xxh64::hash(metas.data(), metas.size() * sizeof(PrismExpertMeta));
        hdr.reserved1[1] = tfq_xxh64::hash(all_scales.data(), all_scales.size() * 2ULL);
        hdr.reserved1[2] = tfq_xxh64::hash(all_weights.data(), all_weights.size() * 4ULL);
        hdr.reserved1[3] = 0ULL;
        strncpy((char*)hdr.meta_tag, mode.c_str(), 31);
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
        printf("L%02d -> %s (%.3f GiB) in %.1fs\n", lyr, out_path, (double)(endp + pade) / (double)(1ULL << 30), el);
    }
    return 0;
}
