// ============================================================================
// Inkling-2 (NVFP4 + MXFP8 3D-packed MoE + BF16 non-MoE) NVIDIA CUDA
// TFQ3 (2D-AffineCoset/half4) GPU Converter
//   - Byte-for-byte identical to src/converters/inkling2/convert.mm (Metal).
// ============================================================================
#include "../cuda_encode_common.cuh"
#include "ink2_common.hpp"

__device__ __forceinline__ float ink2_e2m1_cu(uint32_t nib) {
    static constexpr float LUT[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return LUT[nib & 15u];
}

__device__ __forceinline__ float ink2_e8m0_cu(uint32_t sb) {
    if (sb == 0u) return 5.8774717541114375e-39f; // 0x1p-127f
    return __uint_as_float(sb << 23);
}

__global__ void nvfp4_dequant_row_fwht_kernel(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ scales,
    const float*   __restrict__ fp8_lut,
    float*         __restrict__ W_tmp,
    ink2::Ink2Params P
) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t bc = blockIdx.x, r = blockIdx.y, c0 = bc * 128u;
    const uint32_t sr = P.row0 + r * P.row_step;
    const uint8_t* crow = codes  + (size_t)sr * (P.C >> 1);
    const uint8_t* srow = scales + (size_t)sr * (P.C >> 4);
    const float outer = P.outer;

    const uint32_t c_0 = c0 + lane +  0u;
    const uint32_t c_1 = c0 + lane + 32u;
    const uint32_t c_2 = c0 + lane + 64u;
    const uint32_t c_3 = c0 + lane + 96u;

    const uint32_t b0 = crow[c_0 >> 1], n0 = (c_0 & 1u) ? (b0 >> 4) : (b0 & 15u);
    const uint32_t b1 = crow[c_1 >> 1], n1 = (c_1 & 1u) ? (b1 >> 4) : (b1 & 15u);
    const uint32_t b2 = crow[c_2 >> 1], n2 = (c_2 & 1u) ? (b2 >> 4) : (b2 & 15u);
    const uint32_t b3 = crow[c_3 >> 1], n3 = (c_3 & 1u) ? (b3 >> 4) : (b3 & 15u);

    float4 v;
    v.x = ((ink2_e2m1_cu(n0) * fp8_lut[srow[c_0 >> 4]]) * outer) * enc_sgn_bit(0x12345678u, c_0);
    v.y = ((ink2_e2m1_cu(n1) * fp8_lut[srow[c_1 >> 4]]) * outer) * enc_sgn_bit(0x12345678u, c_1);
    v.z = ((ink2_e2m1_cu(n2) * fp8_lut[srow[c_2 >> 4]]) * outer) * enc_sgn_bit(0x12345678u, c_2);
    v.w = ((ink2_e2m1_cu(n3) * fp8_lut[srow[c_3 >> 4]]) * outer) * enc_sgn_bit(0x12345678u, c_3);
    enc_fwht128_simd4(v, lane);
    float* dst = W_tmp + (size_t)r * P.C + c0;
    dst[lane +  0u] = v.x;
    dst[lane + 32u] = v.y;
    dst[lane + 64u] = v.z;
    dst[lane + 96u] = v.w;
}

__global__ void mxfp8_dequant_row_fwht_kernel(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ scales,
    const float*   __restrict__ fp8_lut,
    float*         __restrict__ W_tmp,
    ink2::Ink2Params P
) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t bc = blockIdx.x, r = blockIdx.y, c0 = bc * 128u;
    const uint32_t sr = P.row0 + r * P.row_step;
    const uint8_t* crow = codes  + (size_t)sr * P.C;
    const uint8_t* srow = scales + (size_t)sr * (P.C >> 5);

    const uint32_t c_0 = c0 + lane +  0u;
    const uint32_t c_1 = c0 + lane + 32u;
    const uint32_t c_2 = c0 + lane + 64u;
    const uint32_t c_3 = c0 + lane + 96u;

    float4 v;
    v.x = (fp8_lut[crow[c_0]] * ink2_e8m0_cu(srow[c_0 >> 5])) * enc_sgn_bit(0x12345678u, c_0);
    v.y = (fp8_lut[crow[c_1]] * ink2_e8m0_cu(srow[c_1 >> 5])) * enc_sgn_bit(0x12345678u, c_1);
    v.z = (fp8_lut[crow[c_2]] * ink2_e8m0_cu(srow[c_2 >> 5])) * enc_sgn_bit(0x12345678u, c_2);
    v.w = (fp8_lut[crow[c_3]] * ink2_e8m0_cu(srow[c_3 >> 5])) * enc_sgn_bit(0x12345678u, c_3);
    enc_fwht128_simd4(v, lane);
    float* dst = W_tmp + (size_t)r * P.C + c0;
    dst[lane +  0u] = v.x;
    dst[lane + 32u] = v.y;
    dst[lane + 64u] = v.z;
    dst[lane + 96u] = v.w;
}

int main(int argc, char** argv) {
    std::string model_dir = "", out_dir = "", trace_path = "", tables_path = "";
    std::string mode = "uniform3";
    int l_start = -1, l_end = -1;
    uint32_t nonmoe_bits = 5u;
    bool skip_existing = false, copy_residuals = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { model_dir.clear(); break; }
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--trace" && i + 1 < argc) trace_path = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--bits" && i + 1 < argc) nonmoe_bits = (uint32_t)atoi(argv[++i]);
        else if (a == "--tables" && i + 1 < argc) tables_path = argv[++i];
        else if (a == "--layers" && i + 2 < argc) { l_start = atoi(argv[++i]); l_end = atoi(argv[++i]); }
        else if (a == "--skip-existing") skip_existing = true;
        else if (a == "--copy-residuals") copy_residuals = true;
    }
    if (model_dir.empty() || out_dir.empty()) {
        fprintf(stderr,
            "Usage: %s --model <dir> --out <dir> [--mode uniform4|heat543|heat443|heat43|uniform3|nonmoe]\n"
            "          [--layers S E] [--trace <file>] [--bits 4|5] [--copy-residuals] [--skip-existing] [--tables <path>]\n", argv[0]);
        return 1;
    }
    if (mode != "uniform4" && mode != "heat543" && mode != "heat443" && mode != "heat43" &&
        mode != "uniform3" && mode != "nonmoe") {
        fprintf(stderr, "Error: unknown --mode %s\n", mode.c_str());
        return 1;
    }
    if (nonmoe_bits < 3u || nonmoe_bits > 5u) {
        fprintf(stderr, "Error: --bits must be 3, 4, or 5\n");
        return 1;
    }
    mkdir(out_dir.c_str(), 0755);
    const bool explicit_tables = !tables_path.empty();
    std::string dst_tables = out_dir + "/pw_v3_tables.bin";
    if (!explicit_tables) tables_path = dst_tables;

    PwV3TablesFile tbl;
    load_or_generate_tables(tables_path, tbl, explicit_tables);
    if (tables_path != dst_tables) ink2::copy_small_file(tables_path, dst_tables);

    auto grid3 = build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
    auto grid4 = build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
    auto grid5 = build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

    float h_fp8[256];
    for (int i = 0; i < 256; ++i) h_fp8[i] = ink2::e4m3_to_f32_host((uint8_t)i);
    float* d_fp8_lut = nullptr;
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

    ink2::ShardCache cache;
    cache.init(model_dir);

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
    constexpr uint32_t CHUNK_STR = 512u;

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
                    nm.resize(nm.size() - 7);
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

        size_t max_e = 0, max_sc = 0, max_r = 0, max_c = 0;
        for (const auto& tk : tasks) {
            max_e  = std::max(max_e,  (size_t)tk.R * tk.C);
            max_sc = std::max(max_sc, (size_t)(tk.R / 128) * (tk.C / 128));
            max_r  = std::max(max_r,  (size_t)tk.R);
            max_c  = std::max(max_c,  (size_t)tk.C);
        }
        uint16_t* d_bf16 = nullptr; float *d_tmp = nullptr, *d_rot = nullptr, *d_ss = nullptr, *d_str = nullptr, *d_se = nullptr;
        __half* d_sc = nullptr; uint32_t *d_hist = nullptr, *d_pack = nullptr;
        TFQ_CUDA_CHECK(cudaMalloc(&d_bf16, max_e * 2));
        TFQ_CUDA_CHECK(cudaMalloc(&d_tmp,  max_e * 4));
        TFQ_CUDA_CHECK(cudaMalloc(&d_rot,  max_e * 4));
        TFQ_CUDA_CHECK(cudaMalloc(&d_ss,   max_sc * 128ULL * 4ULL));
        TFQ_CUDA_CHECK(cudaMalloc(&d_str,  max_e * 4));
        TFQ_CUDA_CHECK(cudaMalloc(&d_hist, (size_t)CHUNK_STR * (max_c / 4) * 256ULL * 4ULL));
        TFQ_CUDA_CHECK(cudaMalloc(&d_pack, max_e));
        TFQ_CUDA_CHECK(cudaMalloc(&d_sc,   max_sc * 2));
        TFQ_CUDA_CHECK(cudaMalloc(&d_se,   (max_r / 16) * 2ULL * 4ULL));

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
            const uint8_t* ptr = cache.get_tensor(tk.name + ".weight");
            tfq_common::check_finite_bf16(reinterpret_cast<const uint16_t*>(ptr), tk.R, tk.C, tk.name + ".weight");
            TFQ_CUDA_CHECK(cudaMemcpy(d_bf16, ptr, (size_t)tk.R * tk.C * 2, cudaMemcpyHostToDevice));
            MatParams mp{tk.R, tk.C, tk.bits, 1.0f, 0u, 0u};
            bf16_row_fwht_kernel<<<dim3(tk.C / 128u, tk.R, 1), 32>>>(d_bf16, d_tmp, mp);
            col_fwht_and_ss_kernel<<<dim3(tk.C, tk.R / 128u, 1), 32>>>(d_tmp, d_rot, d_ss, mp);
            finalize_scales_and_pack_nax_kernel<<<dim3(tk.R / 16u, tk.C / 128u, 1), 32>>>(d_rot, d_ss, d_sc, d_str, mp);
            uint32_t n_str = (tk.R / 16u) * 32u;
            for (uint32_t off = 0; off < n_str; off += CHUNK_STR) {
                uint32_t cnt = std::min(CHUNK_STR, n_str - off);
                launch_prism_encode(tk.bits, cnt, d_str, d_lut[tk.bits], d_c64[tk.bits], d_grid[tk.bits], d_hist, d_pack, EncParams{tk.R, tk.C, tk.bits, off, cnt});
            }
            prism_verify_v3_snr_kernel<<<tk.R / 16u, 32>>>(d_pack, d_sc, d_lut[tk.bits], d_c64[tk.bits], d_rot, d_se, mp);
            TFQ_CUDA_CHECK(cudaDeviceSynchronize());
            double sig = 0.0, err_s = 0.0;
            cuda_read_se_det(d_se, tk.R / 16u, sig, err_s);
            metas[i].snr_db = (float)(10.0 * std::log10(sig / std::max(err_s, 1e-18)));
            TFQ_CUDA_CHECK(cudaMemcpy(all_scales.data() + metas[i].sc_off, d_sc, (size_t)(tk.R / 128u) * (tk.C / 128u) * 2, cudaMemcpyDeviceToHost));
            TFQ_CUDA_CHECK(cudaMemcpy(all_weights.data() + metas[i].word_off, d_pack, metas[i].word_count * 4, cudaMemcpyDeviceToHost));
        }
        size_t off_meta = sizeof(TfpnHeader);
        size_t off_sc   = align_up(off_meta + metas.size() * sizeof(TfpnTensorMeta), ALIGN_16K);
        size_t off_w    = align_up(off_sc + all_scales.size() * 2ULL, ALIGN_16K);
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
        if (copy_residuals) ink2::export_residuals_and_assets(cache, model_dir, out_dir, quant_names);
        return 0;
    }

    // MoE layer mode
    const uint32_t N_SC = (uint32_t)((H / 128) * (I / 128));
    const size_t MAX_ELEMS = (size_t)H * (size_t)I;
    const int MAX_C = std::max(H, I);
    const size_t W13_MAX_CODE_BYTES = (size_t)(2 * I) * (size_t)H;
    const size_t W13_MAX_SC_BYTES   = (size_t)(2 * I) * (size_t)(H / 16);

    uint8_t *d_codes = nullptr, *d_scales = nullptr;
    float *d_tmp = nullptr, *d_rot = nullptr, *d_ss = nullptr, *d_str = nullptr, *d_se = nullptr;
    __half* d_sc = nullptr; uint32_t *d_hist = nullptr, *d_pack = nullptr;
    TFQ_CUDA_CHECK(cudaMalloc(&d_codes,  W13_MAX_CODE_BYTES));
    TFQ_CUDA_CHECK(cudaMalloc(&d_scales, W13_MAX_SC_BYTES));
    TFQ_CUDA_CHECK(cudaMalloc(&d_tmp,    MAX_ELEMS * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_rot,    MAX_ELEMS * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_ss,     (size_t)N_SC * 128ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_str,    MAX_ELEMS * 4));
    TFQ_CUDA_CHECK(cudaMalloc(&d_hist,   (size_t)CHUNK_STR * (size_t)(MAX_C / 4) * 256ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_pack,   (size_t)(I / 16) * (size_t)(H / 128) * 10ULL * 32ULL * 4ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_sc,     (size_t)N_SC * 2ULL));
    TFQ_CUDA_CHECK(cudaMalloc(&d_se,     (size_t)(std::max(H, I) / 16) * 2ULL * 4ULL));

    auto heat_ranks = trace_path.empty() ? std::map<int, std::vector<int>>{} : ink2::load_heat_ranks(trace_path, N_EXP);

    for (int lyr = l_start; lyr <= l_end; ++lyr) {
        char out_path[512];
        snprintf(out_path, sizeof(out_path), "%s/layer_%02d.tffpw", out_dir.c_str(), lyr);
        if (skip_existing && tfpw_layer_complete(out_path)) continue;
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
        const uint8_t* p_w13 = cache.get_tensor(k_w13);
        const uint8_t* p_s13 = cache.get_tensor(k_s13);
        const uint8_t* p_w2  = cache.get_tensor(k_w2);
        const uint8_t* p_s2  = cache.get_tensor(k_s2);
        const float *p_a13 = nullptr, *p_d13 = nullptr, *p_a2 = nullptr, *p_d2 = nullptr;
        if (is_nvfp4) {
            p_a13 = reinterpret_cast<const float*>(cache.get_tensor(k_a13));
            p_d13 = reinterpret_cast<const float*>(cache.get_tensor(k_d13));
            p_a2  = reinterpret_cast<const float*>(cache.get_tensor(k_a2));
            p_d2  = reinterpret_cast<const float*>(cache.get_tensor(k_d2));
        }

        std::vector<PrismExpertMeta> metas(N_EXP);
        memset(metas.data(), 0, metas.size() * sizeof(PrismExpertMeta));
        const int thr5  = N_EXP / 4;
        const int thr4  = (N_EXP * 5) / 9;
        const int thr43 = N_EXP / 3;
        uint64_t cur_word = 0; uint32_t cur_sc = 0;
        for (int e = 0; e < N_EXP; ++e) {
            int r = ranks[e];
            uint32_t b = 4u;
            if (mode == "heat543")       b = (r < thr5) ? 5u : (r < (N_EXP * 5) / 8 ? 4u : 3u);
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

        const size_t w13_code_stride = is_nvfp4 ? (size_t)(2 * I) * (H / 2)  : (size_t)(2 * I) * H;
        const size_t w13_sc_stride   = is_nvfp4 ? (size_t)(2 * I) * (H / 16) : (size_t)(2 * I) * (H / 32);
        const size_t w2_code_stride  = is_nvfp4 ? (size_t)H * (I / 2)        : (size_t)H * I;
        const size_t w2_sc_stride    = is_nvfp4 ? (size_t)H * (I / 16)       : (size_t)H * (I / 32);

        auto encode_proj = [&](uint32_t R, uint32_t C, uint32_t b, uint32_t row0, uint32_t row_step,
                               float outer, uint64_t w_off, uint32_t s_off, float& out_snr) {
            ink2::Ink2Params ip{R, C, row0, row_step, outer, 0u};
            MatParams mp{R, C, b, 1.0f, 0u, 0u};
            if (is_nvfp4) nvfp4_dequant_row_fwht_kernel<<<dim3(C / 128u, R, 1), 32>>>(d_codes, d_scales, d_fp8_lut, d_tmp, ip);
            else          mxfp8_dequant_row_fwht_kernel<<<dim3(C / 128u, R, 1), 32>>>(d_codes, d_scales, d_fp8_lut, d_tmp, ip);
            col_fwht_and_ss_kernel<<<dim3(C, R / 128u, 1), 32>>>(d_tmp, d_rot, d_ss, mp);
            finalize_scales_and_pack_nax_kernel<<<dim3(R / 16u, C / 128u, 1), 32>>>(d_rot, d_ss, d_sc, d_str, mp);
            uint32_t n_str = (R / 16u) * 32u;
            for (uint32_t off = 0; off < n_str; off += CHUNK_STR) {
                uint32_t cnt = std::min(CHUNK_STR, n_str - off);
                launch_prism_encode(b, cnt, d_str, d_lut[b], d_c64[b], d_grid[b], d_hist, d_pack, EncParams{R, C, b, off, cnt});
            }
            prism_verify_v3_snr_kernel<<<R / 16u, 32>>>(d_pack, d_sc, d_lut[b], d_c64[b], d_rot, d_se, mp);
            TFQ_CUDA_CHECK(cudaDeviceSynchronize());
            double sig = 0.0, err_s = 0.0;
            cuda_read_se_det(d_se, R / 16u, sig, err_s);
            out_snr = (float)(10.0 * std::log10(sig / std::max(err_s, 1e-18)));
            snr_sum[b] += out_snr; snr_cnt[b]++;
            uint64_t wpm = (uint64_t)(R / 16u) * (C / 128u) * (b * 2u) * 32u;
            TFQ_CUDA_CHECK(cudaMemcpy(all_scales.data() + s_off, d_sc, (size_t)N_SC * 2ULL, cudaMemcpyDeviceToHost));
            TFQ_CUDA_CHECK(cudaMemcpy(all_weights.data() + w_off, d_pack, wpm * 4ULL, cudaMemcpyDeviceToHost));
        };

        for (int e = 0; e < N_EXP; ++e) {
            uint32_t b = metas[e].b_bits;
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
            TFQ_CUDA_CHECK(cudaMemcpy(d_codes,  w13_c, w13_code_stride, cudaMemcpyHostToDevice));
            TFQ_CUDA_CHECK(cudaMemcpy(d_scales, w13_s, w13_sc_stride,   cudaMemcpyHostToDevice));
            encode_proj((uint32_t)I, (uint32_t)H, b, 0u, 2u, outer13, metas[e].gate_word_off, metas[e].gate_sc_off, metas[e].gate_snr_db);
            encode_proj((uint32_t)I, (uint32_t)H, b, 1u, 2u, outer13, metas[e].up_word_off,   metas[e].up_sc_off,   metas[e].up_snr_db);

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
            TFQ_CUDA_CHECK(cudaMemcpy(d_codes,  w2_c, w2_code_stride, cudaMemcpyHostToDevice));
            TFQ_CUDA_CHECK(cudaMemcpy(d_scales, w2_s, w2_sc_stride,   cudaMemcpyHostToDevice));
            encode_proj((uint32_t)H, (uint32_t)I, b, 0u, 1u, outer2, metas[e].down_word_off, metas[e].down_sc_off, metas[e].down_snr_db);
        }

        size_t off_meta = sizeof(PrismLayerHeader);
        size_t off_sc   = align_up(off_meta + metas.size() * sizeof(PrismExpertMeta), ALIGN_16K);
        size_t off_w    = align_up(off_sc + all_scales.size() * 2ULL, ALIGN_16K);
        PrismLayerHeader hdr{};
        hdr.magic = MAGIC_TFPW; hdr.version = FORMAT_VER;
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

        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_lyr).count();
        printf("  [L%02d/%02d %s] %.3f GiB | 3b=%.2f(%d) 4b=%.2f(%d) 5b=%.2f(%d) dB | %.1fs\n",
               lyr, l_end, is_nvfp4 ? "NVFP4" : "MXFP8",
               (double)(endp + pade) / (double)(1ULL << 30),
               snr_cnt[3] ? snr_sum[3] / snr_cnt[3] : 0.0, snr_cnt[3] / 3,
               snr_cnt[4] ? snr_sum[4] / snr_cnt[4] : 0.0, snr_cnt[4] / 3,
               snr_cnt[5] ? snr_sum[5] / snr_cnt[5] : 0.0, snr_cnt[5] / 3,
               dt);
        fflush(stdout);
    }

    if (copy_residuals) ink2::export_residuals_and_assets(cache, model_dir, out_dir);
    printf("INKLING-2 CUDA DONE -> %s\n", out_dir.c_str());
    return 0;
}
