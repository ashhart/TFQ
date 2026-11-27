// ============================================================================
// PyTorch C++/CUDA Extension for TFQ3 (3/4/5-bit 2D-AffineCoset + 2D FWHT-128)
//   Directly binds `src/kernels/cuda/tfq3_cuda.cuh` to torch::Tensor so any
//   PyTorch, Hugging Face Transformers, or vLLM model can run TFQ3 checkpoints
//   without TensorFold — including Col-TP (`row_off`) and Row-TP (`k_sign_off`,
//   `bc0_buf`, `n_bc_buf` trellis state resume).
// ============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "../../src/kernels/cuda/tfq3_cuda.cuh"
#include "../../src/kernels/cuda/tfq4_cuda.cuh"

torch::Tensor tfq3_linear_cuda(
    torch::Tensor x_bf16,      // [..., orig_c] bfloat16 CUDA
    torch::Tensor words_u32,   // flat int32/uint32 packed TFPN words CUDA
    torch::Tensor scales_f16,  // [pad_r/128, n_bc_buf] float16 CUDA
    torch::Tensor tables_u8,   // [16384] uint8 pw_v3_tables.bin CUDA
    int64_t bits,
    int64_t pad_r,
    int64_t pad_c,
    int64_t orig_r,
    int64_t row_off = 0,       // absolute row offset (multiple of 128) for Col-TP
    int64_t k_sign_off = 0,    // absolute col offset (multiple of 128) for Row-TP
    int64_t bc0_buf = 0,       // starting col-block index inside words_u32 / scales_f16
    int64_t n_bc_buf = 0       // total col blocks stored in words_u32 / scales_f16 (0 -> pad_c/128)
) {
    TORCH_CHECK(x_bf16.is_cuda(), "x_bf16 must be a CUDA tensor");
    TORCH_CHECK(x_bf16.scalar_type() == torch::kBFloat16, "x_bf16 must be bfloat16");
    TORCH_CHECK(words_u32.is_cuda() && scales_f16.is_cuda() && tables_u8.is_cuda(), "TFQ3 buffers must be CUDA");
    TORCH_CHECK(bits == 3 || bits == 4 || bits == 5, "TFQ3 bits must be 3, 4, or 5");
    TORCH_CHECK((pad_r % 128) == 0 && (pad_c % 128) == 0, "pad_r and pad_c must be multiples of 128");
    TORCH_CHECK((row_off % 128) == 0 && row_off >= 0, "row_off must be a non-negative multiple of 128");
    TORCH_CHECK((k_sign_off % 128) == 0 && k_sign_off >= 0, "k_sign_off must be a non-negative multiple of 128");
    TORCH_CHECK(tables_u8.numel() >= (int64_t)sizeof(tfq3::PwV3Tables), "tables_u8 smaller than PwV3Tables");

    const c10::cuda::CUDAGuard device_guard(x_bf16.device());
    auto orig_sizes = x_bf16.sizes().vec();
    const int64_t orig_c = orig_sizes.back();
    const int64_t rows = x_bf16.numel() / orig_c;
    const int64_t n_bc_local = pad_c >> 7;
    const int64_t n_bc_stored = (n_bc_buf > 0) ? n_bc_buf : n_bc_local;
    TORCH_CHECK(bc0_buf >= 0 && bc0_buf + n_bc_local <= n_bc_stored, "bc0_buf + pad_c/128 exceeds n_bc_buf");

    torch::Tensor x2d = x_bf16.contiguous().view({rows, orig_c});
    if (orig_c < pad_c) {
        x2d = torch::nn::functional::pad(x2d, torch::nn::functional::PadFuncOptions({0, pad_c - orig_c}));
    }

    auto opts_bf16 = x_bf16.options().dtype(torch::kBFloat16);
    auto opts_f16  = x_bf16.options().dtype(torch::kFloat16);
    auto opts_f32  = x_bf16.options().dtype(torch::kFloat32);

    torch::Tensor y_pad = torch::empty({rows, pad_r}, opts_bf16);
    torch::Tensor xrot  = torch::empty({rows, pad_c}, opts_f16);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaError_t err;
    if (k_sign_off == 0 && bc0_buf == 0 && n_bc_stored == n_bc_local) {
        const size_t yr_elems = (rows <= 64) ? tfq3::required_yr_elems(
            static_cast<uint32_t>(rows), static_cast<uint32_t>(pad_r), static_cast<uint32_t>(pad_c)
        ) : 0ull;
        torch::Tensor yr = (yr_elems > 0ull) ? torch::empty({static_cast<int64_t>(yr_elems)}, opts_f32) : torch::empty({0}, opts_f32);
        err = tfq3::launch_dense(
            reinterpret_cast<const uint32_t*>(words_u32.data_ptr()),
            reinterpret_cast<const __half*>(scales_f16.data_ptr()),
            reinterpret_cast<const tfq3::PwV3Tables*>(tables_u8.data_ptr()),
            reinterpret_cast<const uint16_t*>(x2d.data_ptr()),
            y_pad.data_ptr(),
            reinterpret_cast<__half*>(xrot.data_ptr()),
            (yr_elems > 0ull) ? yr.data_ptr<float>() : nullptr,
            yr_elems,
            static_cast<uint32_t>(bits),
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(pad_r),
            static_cast<uint32_t>(pad_c),
            static_cast<uint32_t>(pad_c),
            static_cast<uint32_t>(pad_r),
            0u,
            false,
            static_cast<uint32_t>(row_off),
            stream
        );
    } else {
        err = tfq3::launch_dense_tp_sliced(
            reinterpret_cast<const uint32_t*>(words_u32.data_ptr()),
            reinterpret_cast<const __half*>(scales_f16.data_ptr()),
            reinterpret_cast<const tfq3::PwV3Tables*>(tables_u8.data_ptr()),
            reinterpret_cast<const uint16_t*>(x2d.data_ptr()),
            y_pad.data_ptr(),
            reinterpret_cast<__half*>(xrot.data_ptr()),
            static_cast<uint32_t>(bits),
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(pad_r),
            static_cast<uint32_t>(pad_c),
            static_cast<uint32_t>(row_off),
            static_cast<uint32_t>(k_sign_off),
            static_cast<uint32_t>(bc0_buf),
            static_cast<uint32_t>(n_bc_stored),
            static_cast<uint32_t>(pad_c),
            static_cast<uint32_t>(pad_r),
            false,
            stream
        );
    }
    TORCH_CHECK(err == cudaSuccess, "tfq3 CUDA dense launch failed: ", cudaGetErrorString(err));

    torch::Tensor y_out = (orig_r < pad_r) ? y_pad.slice(1, 0, orig_r).contiguous() : y_pad;
    orig_sizes.back() = orig_r;
    return y_out.view(orig_sizes);
}

torch::Tensor tfq4_linear_cuda(
    torch::Tensor x_bf16,      // [..., orig_c] bfloat16 CUDA
    torch::Tensor words_u32,   // [pad_r, pad_c/16] int32/uint32 packed TFP4 words CUDA
    torch::Tensor scales_f16,  // [pad_r/128, pad_c/128] float16 CUDA
    torch::Tensor e8p_lut,     // [256, 8] float16 (4096 B) OR [16384] uint8 PWV4 CUDA
    int64_t table_kind,        // 0/1 = trained 8-sign LUT, 2 = true QuIP# E8P (E8 + 1/4)
    int64_t pad_r,
    int64_t pad_c,
    int64_t orig_r,
    int64_t row_off = 0,       // absolute output row offset (multiple of 128, Col-TP)
    int64_t k_sign_off = 0     // absolute input col offset (multiple of 128, Row-TP)
) {
    TORCH_CHECK(x_bf16.is_cuda(), "x_bf16 must be a CUDA tensor");
    TORCH_CHECK(x_bf16.scalar_type() == torch::kBFloat16, "x_bf16 must be bfloat16");
    TORCH_CHECK(words_u32.is_cuda() && scales_f16.is_cuda() && e8p_lut.is_cuda(), "TFQ4 buffers must be CUDA");
    TORCH_CHECK(table_kind >= 0 && table_kind <= (int64_t)tfq4_cuda::TABLE_KIND_MAX_KNOWN,
                "Unsupported PWV4 table_kind (expected 0, 1, or 2)");
    TORCH_CHECK((pad_r % 128) == 0 && (pad_c % 128) == 0, "pad_r and pad_c must be multiples of 128");
    TORCH_CHECK((row_off % 128) == 0 && row_off >= 0, "row_off must be a non-negative multiple of 128");
    TORCH_CHECK((k_sign_off % 128) == 0 && k_sign_off >= 0, "k_sign_off must be a non-negative multiple of 128");

    const c10::cuda::CUDAGuard device_guard(x_bf16.device());
    auto orig_sizes = x_bf16.sizes().vec();
    const int64_t orig_c = orig_sizes.back();
    const int64_t rows = x_bf16.numel() / orig_c;
    if (rows == 0) {
        orig_sizes.back() = orig_r;
        return torch::empty(orig_sizes, x_bf16.options());
    }

    torch::Tensor w_c = words_u32.contiguous();
    torch::Tensor s_c = scales_f16.contiguous();
    torch::Tensor lut_c = e8p_lut.contiguous();
    const int64_t lut_bytes = lut_c.numel() * lut_c.element_size();
    const uint4* lut_ptr = nullptr;
    if (lut_bytes == 4096) {
        lut_ptr = reinterpret_cast<const uint4*>(lut_c.data_ptr());
    } else if (lut_bytes >= 16384 && lut_c.scalar_type() == torch::kUInt8) {
        lut_ptr = reinterpret_cast<const uint4*>(lut_c.data_ptr<uint8_t>() + 64);
    } else {
        TORCH_CHECK(false, "e8p_lut must be a 4096-byte fp16 [256,8] LUT or 16384-byte uint8 PWV4 buffer");
    }

    torch::Tensor x2d = x_bf16.contiguous().view({rows, orig_c});
    if (orig_c < pad_c) {
        x2d = torch::nn::functional::pad(x2d, torch::nn::functional::PadFuncOptions({0, pad_c - orig_c}));
    }

    auto opts_bf16 = x_bf16.options().dtype(torch::kBFloat16);
    auto opts_f16  = x_bf16.options().dtype(torch::kFloat16);
    torch::Tensor y_pad = torch::empty({rows, pad_r}, opts_bf16);
    torch::Tensor xrot  = torch::empty({rows, pad_c}, opts_f16);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = tfq4_cuda::launch_dense_bf16(
        reinterpret_cast<const uint32_t*>(w_c.data_ptr()),
        reinterpret_cast<const __half*>(s_c.data_ptr()),
        lut_ptr,
        reinterpret_cast<const uint16_t*>(x2d.data_ptr()),
        y_pad.data_ptr(),
        reinterpret_cast<__half*>(xrot.data_ptr()),
        static_cast<uint32_t>(table_kind),
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(pad_r),
        static_cast<uint32_t>(pad_c),
        static_cast<uint32_t>(pad_c),
        static_cast<uint32_t>(pad_r),
        static_cast<uint32_t>(row_off),
        static_cast<uint32_t>(k_sign_off),
        false,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "tfq4 CUDA dense launch failed: ", cudaGetErrorString(err));

    torch::Tensor y_out = (orig_r < pad_r) ? y_pad.slice(1, 0, orig_r).contiguous() : y_pad;
    orig_sizes.back() = orig_r;
    return y_out.view(orig_sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "tfq3_linear",
        &tfq3_linear_cuda,
        "TFQ3 3/4/5-bit fused 2D-FWHT + Tensor Core linear (CUDA, supports Col-TP row_off and Row-TP k_sign_off)",
        py::arg("x_bf16"),
        py::arg("words_u32"),
        py::arg("scales_f16"),
        py::arg("tables_u8"),
        py::arg("bits"),
        py::arg("pad_r"),
        py::arg("pad_c"),
        py::arg("orig_r"),
        py::arg("row_off") = 0,
        py::arg("k_sign_off") = 0,
        py::arg("bc0_buf") = 0,
        py::arg("n_bc_buf") = 0
    );
    m.def(
        "tfq4_linear",
        &tfq4_linear_cuda,
        "TFQ4 2-bit E8P fused 2D-FWHT linear (CUDA, supports table_kind 0/1 trained & 2 true QuIP# E8P, Col-TP & Row-TP)",
        py::arg("x_bf16"),
        py::arg("words_u32"),
        py::arg("scales_f16"),
        py::arg("e8p_lut"),
        py::arg("table_kind"),
        py::arg("pad_r"),
        py::arg("pad_c"),
        py::arg("orig_r"),
        py::arg("row_off") = 0,
        py::arg("k_sign_off") = 0
    );
}
