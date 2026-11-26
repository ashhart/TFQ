#pragma once
// ============================================================================
// Shared CUDA TFQ3 (2D-AffineCoset/half4 + 2D FWHT-128) GPU Encoder Kernels,
// Universal 16 KiB Codebook Generator ("PWV3"), and Mmap Safetensors Reader.
//
// Produces bit-compatible `.tffpw` ("TFPN" / "TFPW") and `pw_v3_tables.bin`
// files identical to the Apple Silicon Metal converter suite.
// ============================================================================
#include "tfq_tables_io.hpp"
#include "tfq_xxh64.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define TFQ_CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

static constexpr uint32_t MAGIC_TFPN = 0x4E504654u; // "TFPN"
static constexpr uint32_t MAGIC_TFPW = 0x57504654u; // "TFPW"
static constexpr uint32_t MAGIC_PWV3 = 0x33565750u; // "PWV3"
static constexpr uint32_t FORMAT_VER = 3u;
static constexpr size_t   ALIGN_16K  = 16384;

static inline size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

using PwV3TablesFile = tfq_tables::PwV3TablesFile;

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

// Output writes: a short write or failed close (e.g. disk full) must stop the converter with a nonzero exit.
static inline void write_all_or_exit(FILE* f, const void* p, size_t n, const std::string& path) {
    if (n && fwrite(p, 1, n, f) != n) { fprintf(stderr, "Write error on %s\n", path.c_str()); exit(1); }
}

static inline void close_or_exit(FILE* f, const std::string& path) {
    if (fclose(f) != 0) { fprintf(stderr, "Write error on %s\n", path.c_str()); exit(1); }
}

static inline void rename_or_exit(const std::string& from, const std::string& to) {
    if (rename(from.c_str(), to.c_str()) != 0) { fprintf(stderr, "Cannot rename %s to %s\n", from.c_str(), to.c_str()); exit(1); }
}

// --skip-existing: a TFPW layer file counts as done only if its size matches what its own header describes.
static inline bool tfpw_layer_complete(const std::string& path) {
    PrismLayerHeader h{};
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = fread(&h, sizeof(h), 1, f) == 1;
    fclose(f);
    struct stat st{};
    if (!ok || stat(path.c_str(), &st) != 0 || h.magic != MAGIC_TFPW || h.version != FORMAT_VER) return false;
    if (h.stream_words > (UINT64_MAX - h.off_weights - ALIGN_16K) / 4ULL) return false;
    return (uint64_t)st.st_size == align_up(h.off_weights + h.stream_words * 4ULL, ALIGN_16K);
}

struct MatParams { uint32_t R, C, b_bits; float shrink; uint32_t r_off, pad0; };
struct EncParams { uint32_t R, C, b_bits, stream_off, n_streams; };
struct PleParams { uint32_t rows, cols, dst_bits, pad; };
struct P2 { float x, y; };

static inline uint16_t host_f32_to_f16(float f) {
    return __half_as_ushort(__float2half(f));
}
static inline float host_f16_to_f32(uint16_t u) {
    return __half2float(__ushort_as_half(u));
}
static inline uint32_t pack_h2(float x, float y) {
    return (uint32_t)host_f32_to_f16(x) | ((uint32_t)host_f32_to_f16(y) << 16);
}

static inline float fp8_e4m3_to_f32(uint8_t b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return 0.0f;
    float v = (e == 0) ? (std::ldexp((float)m, -9)) : (std::ldexp(1.0f + (float)m / 8.0f, e - 7));
    return s ? -v : v;
}

static inline void load_or_generate_tables(const std::string& path, PwV3TablesFile& tbl, bool explicit_tables = false) {
    tfq_tables::load_or_write_pwv3_tables(path, tbl, explicit_tables);
}

static inline std::vector<uint16_t> build_grid(const uint32_t* lut_u32, int lut_size, int N_sym, int C_shift, int G, int KC) {
    return tfq_tables::build_grid(lut_u32, lut_size, N_sym, C_shift, G, KC);
}

// Portable mmap safetensors reader supporting both indexed multi-shard and single model.safetensors.
struct ShardCache {
    struct Meta { uint64_t off, len; std::string dtype; int R, C; };
    struct Entry {
        int fd = -1; uint8_t* ptr = nullptr; size_t len = 0; uint64_t data_base = 0;
        std::map<std::string, Meta> tensors;
    };
    std::string model_dir;
    std::map<std::string, std::string> weight_map;
    std::map<std::string, Entry> shards;

    void init(const std::string& dir) {
        model_dir = dir;
        std::ifstream f(dir + "/model.safetensors.index.json");
        if (f.good()) {
            std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            size_t pos = txt.find("\"weight_map\"");
            if (pos == std::string::npos) { fprintf(stderr, "No weight_map in %s\n", dir.c_str()); exit(1); }
            pos = txt.find('{', pos);
            size_t end = txt.find('}', pos);
            std::string body = txt.substr(pos + 1, end - pos - 1);
            size_t i = 0;
            while (i < body.size()) {
                size_t q1 = body.find('"', i); if (q1 == std::string::npos) break;
                size_t q2 = body.find('"', q1 + 1);
                size_t q3 = body.find('"', q2 + 1);
                size_t q4 = body.find('"', q3 + 1);
                weight_map[body.substr(q1 + 1, q2 - q1 - 1)] = body.substr(q3 + 1, q4 - q3 - 1);
                i = q4 + 1;
            }
            return;
        }
        Entry& e = get_shard("model.safetensors");
        for (const auto& kv : e.tensors) weight_map[kv.first] = "model.safetensors";
    }

    bool has_tensor(const std::string& name) const { return weight_map.count(name) > 0; }

    Entry& get_shard(const std::string& fn) {
        auto it = shards.find(fn);
        if (it != shards.end()) return it->second;
        Entry e;
        std::string full = model_dir + "/" + fn;
        e.fd = open(full.c_str(), O_RDONLY);
        if (e.fd < 0) { fprintf(stderr, "Cannot open %s\n", full.c_str()); exit(1); }
        struct stat st{}; fstat(e.fd, &st); e.len = (size_t)st.st_size;
        e.ptr = (uint8_t*)mmap(nullptr, e.len, PROT_READ, MAP_PRIVATE, e.fd, 0);
        if (e.ptr == MAP_FAILED) { fprintf(stderr, "mmap failed on %s\n", full.c_str()); exit(1); }
        uint64_t hlen = 0; memcpy(&hlen, e.ptr, 8);
        e.data_base = 8 + hlen;
        std::string hdr((const char*)e.ptr + 8, hlen);
        size_t p = 0;
        while (true) {
            size_t dk = hdr.find("\"data_offsets\"", p);
            if (dk == std::string::npos) break;
            size_t ob = hdr.rfind('{', dk), cb = hdr.find('}', dk), col = hdr.rfind(':', ob);
            size_t q2 = hdr.rfind('"', col), q1 = hdr.rfind('"', q2 - 1);
            std::string tname = hdr.substr(q1 + 1, q2 - q1 - 1);
            std::string obj = hdr.substr(ob, cb - ob + 1);
            size_t lb = hdr.find('[', dk), cm = hdr.find(',', lb), rb = hdr.find(']', cm);
            uint64_t o0 = std::stoull(hdr.substr(lb + 1, cm - lb - 1));
            uint64_t o1 = std::stoull(hdr.substr(cm + 1, rb - cm - 1));
            std::string dtype;
            size_t dp = obj.find("\"dtype\"");
            if (dp != std::string::npos) {
                size_t d1 = obj.find('"', obj.find(':', dp) + 1), d2 = obj.find('"', d1 + 1);
                dtype = obj.substr(d1 + 1, d2 - d1 - 1);
            }
            int R = 0, C = 0;
            size_t sp = obj.find("\"shape\"");
            if (sp != std::string::npos) {
                size_t slb = obj.find('[', sp), srb = obj.find(']', slb);
                std::string sb = obj.substr(slb + 1, srb - slb - 1);
                size_t scm = sb.find(',');
                if (scm != std::string::npos) { R = std::stoi(sb.substr(0, scm)); C = std::stoi(sb.substr(scm + 1)); }
            }
            e.tensors[tname] = {o0, o1 - o0, dtype, R, C};
            p = cb + 1;
        }
        shards[fn] = std::move(e);
        return shards[fn];
    }

    const uint8_t* get_tensor(const std::string& name, std::string* out_dtype = nullptr, size_t* out_len = nullptr, int* out_R = nullptr, int* out_C = nullptr) {
        auto it = weight_map.find(name);
        if (it == weight_map.end()) { fprintf(stderr, "Missing tensor %s\n", name.c_str()); exit(1); }
        Entry& sh = get_shard(it->second);
        auto tit = sh.tensors.find(name);
        if (tit == sh.tensors.end()) { fprintf(stderr, "Missing %s in %s\n", name.c_str(), it->second.c_str()); exit(1); }
        if (out_dtype) *out_dtype = tit->second.dtype;
        if (out_len) *out_len = (size_t)tit->second.len;
        if (out_R) *out_R = tit->second.R;
        if (out_C) *out_C = tit->second.C;
        return sh.ptr + sh.data_base + tit->second.off;
    }

    void evict_all() {
        for (auto& kv : shards) { if (kv.second.ptr) munmap(kv.second.ptr, kv.second.len); if (kv.second.fd >= 0) close(kv.second.fd); }
        shards.clear();
    }
    ~ShardCache() { evict_all(); }
};

// ============================================================================
// CUDA Device Helpers & GPU Encoder Kernels
// ============================================================================
__device__ __forceinline__ int2 enc_frag_home(uint32_t l) {
    return make_int2(static_cast<int>((l & 8u) + ((l & 1u) << 2)),
                     static_cast<int>(((l & 16u) >> 2) | ((l >> 1) & 3u)));
}

__device__ __forceinline__ float enc_sgn_bit(uint32_t seed, uint32_t i) {
    uint32_t h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ float enc_bf16_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ uint16_t enc_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    uint32_t r = u + 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(r >> 16);
}

__device__ __forceinline__ __half2 enc_u32_to_h2(uint32_t u) {
    return *reinterpret_cast<const __half2*>(&u);
}

__device__ __forceinline__ float enc_warp_sum(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) x += __shfl_xor_sync(0xFFFFFFFFu, x, d);
    return x;
}

__device__ __forceinline__ float enc_warp_min(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) x = fminf(x, __shfl_xor_sync(0xFFFFFFFFu, x, d));
    return x;
}

__device__ __forceinline__ void enc_fwht128_simd4(float4& v, uint32_t lane) {
    #pragma unroll
    for (int d = 1; d <= 16; d <<= 1) {
        float ox = __shfl_xor_sync(0xFFFFFFFFu, v.x, d);
        float oy = __shfl_xor_sync(0xFFFFFFFFu, v.y, d);
        float oz = __shfl_xor_sync(0xFFFFFFFFu, v.z, d);
        float ow = __shfl_xor_sync(0xFFFFFFFFu, v.w, d);
        bool lo = ((lane & static_cast<uint32_t>(d)) == 0u);
        v.x = lo ? (v.x + ox) : (ox - v.x);
        v.y = lo ? (v.y + oy) : (oy - v.y);
        v.z = lo ? (v.z + oz) : (oz - v.z);
        v.w = lo ? (v.w + ow) : (ow - v.w);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y;
    float a1 = v.z + v.w, b1 = v.z - v.w;
    constexpr float inv = 0.08838834764831845f;
    v = make_float4((a0 + a1) * inv, (b0 + b1) * inv, (a0 - a1) * inv, (b0 - b1) * inv);
}

__global__ void bf16_row_fwht_kernel(const uint16_t* __restrict__ W_bf16, float* __restrict__ W_tmp, MatParams P) {
    uint32_t r = blockIdx.y, bc = blockIdx.x, c0 = bc * 128u, lane = threadIdx.x;
    const uint16_t* row = W_bf16 + static_cast<uint64_t>(r) * P.C + c0;
    float4 v;
    v.x = enc_bf16_to_f32(row[lane +  0u]) * enc_sgn_bit(0x12345678u, c0 + lane +  0u);
    v.y = enc_bf16_to_f32(row[lane + 32u]) * enc_sgn_bit(0x12345678u, c0 + lane + 32u);
    v.z = enc_bf16_to_f32(row[lane + 64u]) * enc_sgn_bit(0x12345678u, c0 + lane + 64u);
    v.w = enc_bf16_to_f32(row[lane + 96u]) * enc_sgn_bit(0x12345678u, c0 + lane + 96u);
    enc_fwht128_simd4(v, lane);
    float* dst = W_tmp + static_cast<uint64_t>(r) * P.C + c0;
    dst[lane +  0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

__global__ void f32_row_fwht_kernel(const float* __restrict__ W_in, float* __restrict__ W_tmp, MatParams P) {
    uint32_t r = blockIdx.y, bc = blockIdx.x, c0 = bc * 128u, lane = threadIdx.x;
    const float* row = W_in + static_cast<uint64_t>(r) * P.C + c0;
    float4 v;
    v.x = row[lane +  0u] * enc_sgn_bit(0x12345678u, c0 + lane +  0u);
    v.y = row[lane + 32u] * enc_sgn_bit(0x12345678u, c0 + lane + 32u);
    v.z = row[lane + 64u] * enc_sgn_bit(0x12345678u, c0 + lane + 64u);
    v.w = row[lane + 96u] * enc_sgn_bit(0x12345678u, c0 + lane + 96u);
    enc_fwht128_simd4(v, lane);
    float* dst = W_tmp + static_cast<uint64_t>(r) * P.C + c0;
    dst[lane +  0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

__global__ void q8_64_dequant_row_fwht_kernel(
    const uint8_t* __restrict__ W_q8, const uint16_t* __restrict__ S_bf16, const uint16_t* __restrict__ B_bf16,
    float* __restrict__ W_tmp, MatParams P
) {
    uint32_t r = blockIdx.y, bc = blockIdx.x, c0 = bc * 128u, g64 = bc * 2u, n_g64 = P.C >> 6, lane = threadIdx.x;
    float s0 = enc_bf16_to_f32(S_bf16[static_cast<uint64_t>(r) * n_g64 + g64 + 0u]);
    float b0 = enc_bf16_to_f32(B_bf16[static_cast<uint64_t>(r) * n_g64 + g64 + 0u]);
    float s1 = enc_bf16_to_f32(S_bf16[static_cast<uint64_t>(r) * n_g64 + g64 + 1u]);
    float b1 = enc_bf16_to_f32(B_bf16[static_cast<uint64_t>(r) * n_g64 + g64 + 1u]);
    const uint8_t* row = W_q8 + static_cast<uint64_t>(r) * P.C + c0;
    float4 v;
    v.x = fmaf(static_cast<float>(row[lane +  0u]), s0, b0) * enc_sgn_bit(0x12345678u, c0 + lane +  0u);
    v.y = fmaf(static_cast<float>(row[lane + 32u]), s0, b0) * enc_sgn_bit(0x12345678u, c0 + lane + 32u);
    v.z = fmaf(static_cast<float>(row[lane + 64u]), s1, b1) * enc_sgn_bit(0x12345678u, c0 + lane + 64u);
    v.w = fmaf(static_cast<float>(row[lane + 96u]), s1, b1) * enc_sgn_bit(0x12345678u, c0 + lane + 96u);
    enc_fwht128_simd4(v, lane);
    float* dst = W_tmp + static_cast<uint64_t>(r) * P.C + c0;
    dst[lane +  0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

__global__ void q8_32_dequant_row_fwht_kernel(
    const uint8_t* __restrict__ W_q8, const uint16_t* __restrict__ S_bf16, const uint16_t* __restrict__ B_bf16,
    float* __restrict__ W_tmp, MatParams P
) {
    uint32_t r = blockIdx.y, bc = blockIdx.x, c0 = bc * 128u, g32 = bc * 4u, n_g32 = P.C >> 5, lane = threadIdx.x;
    uint64_t sb = static_cast<uint64_t>(r) * n_g32 + g32;
    float s0 = enc_bf16_to_f32(S_bf16[sb + 0u]), b0 = enc_bf16_to_f32(B_bf16[sb + 0u]);
    float s1 = enc_bf16_to_f32(S_bf16[sb + 1u]), b1 = enc_bf16_to_f32(B_bf16[sb + 1u]);
    float s2 = enc_bf16_to_f32(S_bf16[sb + 2u]), b2 = enc_bf16_to_f32(B_bf16[sb + 2u]);
    float s3 = enc_bf16_to_f32(S_bf16[sb + 3u]), b3 = enc_bf16_to_f32(B_bf16[sb + 3u]);
    const uint8_t* row = W_q8 + static_cast<uint64_t>(r) * P.C + c0;
    float4 v;
    v.x = fmaf(static_cast<float>(row[lane +  0u]), s0, b0) * enc_sgn_bit(0x12345678u, c0 + lane +  0u);
    v.y = fmaf(static_cast<float>(row[lane + 32u]), s1, b1) * enc_sgn_bit(0x12345678u, c0 + lane + 32u);
    v.z = fmaf(static_cast<float>(row[lane + 64u]), s2, b2) * enc_sgn_bit(0x12345678u, c0 + lane + 64u);
    v.w = fmaf(static_cast<float>(row[lane + 96u]), s3, b3) * enc_sgn_bit(0x12345678u, c0 + lane + 96u);
    enc_fwht128_simd4(v, lane);
    float* dst = W_tmp + static_cast<uint64_t>(r) * P.C + c0;
    dst[lane +  0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

__global__ void fp8_dequant_row_fwht_kernel(
    const uint8_t* __restrict__ W_fp8, const float* __restrict__ sc_inv, const float* __restrict__ fp8_lut,
    float* __restrict__ W_tmp, MatParams P
) {
    uint32_t r = blockIdx.y, bc = blockIdx.x, c0 = bc * 128u, lane = threadIdx.x;
    float sc = sc_inv[(r >> 7) * (P.C >> 7) + bc];
    const uint8_t* row = W_fp8 + static_cast<uint64_t>(r) * P.C + c0;
    float4 v;
    v.x = fp8_lut[row[lane +  0u]] * sc * enc_sgn_bit(0x12345678u, c0 + lane +  0u);
    v.y = fp8_lut[row[lane + 32u]] * sc * enc_sgn_bit(0x12345678u, c0 + lane + 32u);
    v.z = fp8_lut[row[lane + 64u]] * sc * enc_sgn_bit(0x12345678u, c0 + lane + 64u);
    v.w = fp8_lut[row[lane + 96u]] * sc * enc_sgn_bit(0x12345678u, c0 + lane + 96u);
    enc_fwht128_simd4(v, lane);
    float* dst = W_tmp + static_cast<uint64_t>(r) * P.C + c0;
    dst[lane +  0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

__global__ void col_fwht_and_ss_kernel(const float* __restrict__ W_tmp, float* __restrict__ W_rot, float* __restrict__ col_ss, MatParams P) {
    uint32_t c = blockIdx.x, br = blockIdx.y, r0 = br * 128u, rs = P.r_off + r0, lane = threadIdx.x;
    float4 v;
    v.x = W_tmp[static_cast<uint64_t>(r0 + lane +  0u) * P.C + c] * enc_sgn_bit(0x9E3779B9u, rs + lane +  0u);
    v.y = W_tmp[static_cast<uint64_t>(r0 + lane + 32u) * P.C + c] * enc_sgn_bit(0x9E3779B9u, rs + lane + 32u);
    v.z = W_tmp[static_cast<uint64_t>(r0 + lane + 64u) * P.C + c] * enc_sgn_bit(0x9E3779B9u, rs + lane + 64u);
    v.w = W_tmp[static_cast<uint64_t>(r0 + lane + 96u) * P.C + c] * enc_sgn_bit(0x9E3779B9u, rs + lane + 96u);
    enc_fwht128_simd4(v, lane);
    W_rot[static_cast<uint64_t>(r0 + lane +  0u) * P.C + c] = v.x;
    W_rot[static_cast<uint64_t>(r0 + lane + 32u) * P.C + c] = v.y;
    W_rot[static_cast<uint64_t>(r0 + lane + 64u) * P.C + c] = v.z;
    W_rot[static_cast<uint64_t>(r0 + lane + 96u) * P.C + c] = v.w;
    float ss = enc_warp_sum((v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w));
    uint32_t bi = br * (P.C >> 7) + (c >> 7);
    if (lane == 0u) col_ss[static_cast<uint64_t>(bi) * 128u + (c & 127u)] = ss;
}

__global__ void finalize_scales_and_pack_nax_kernel(
    const float* __restrict__ W_rot, const float* __restrict__ col_ss,
    __half* __restrict__ scales_f16, float* __restrict__ W_streams, MatParams P
) {
    uint32_t strip = blockIdx.x, bc = blockIdx.y, br = strip >> 3, n_bc = P.C >> 7, bi = br * n_bc + bc, lane = threadIdx.x;
    const float* cs = col_ss + static_cast<uint64_t>(bi) * 128u;
    float blk_sum = enc_warp_sum((cs[lane + 0u] + cs[lane + 32u]) + (cs[lane + 64u] + cs[lane + 96u]));
    float rms = sqrtf(blk_sum * (1.0f / 16384.0f) + 1e-12f) * P.shrink;
    __half h_rms = __float2half_rn(rms);
    if ((strip & 7u) == 0u && lane == 0u) scales_f16[bi] = h_rms;
    float inv_sc = 1.0f / __half2float(h_rms);
    int2 home = enc_frag_home(lane);
    uint32_t K_len = P.C >> 1;
    uint32_t stream_id = strip * 32u + lane;
    float* dst = W_streams + static_cast<uint64_t>(stream_id) * K_len + static_cast<uint64_t>(bc) * 64u;
    uint32_t r_base = strip * 16u, c_base = bc * 128u;
    #pragma unroll
    for (uint32_t kt = 0; kt < 8u; ++kt) {
        uint32_t cc0 = c_base + kt * 16u + static_cast<uint32_t>(home.x);
        uint32_t r0 = r_base + static_cast<uint32_t>(home.y), r1 = r0 + 8u;
        const float* p0 = W_rot + static_cast<uint64_t>(r0) * P.C + cc0;
        const float* p1 = W_rot + static_cast<uint64_t>(r1) * P.C + cc0;
        dst[kt * 8u + 0u] = p0[0] * inv_sc; dst[kt * 8u + 1u] = p0[1] * inv_sc;
        dst[kt * 8u + 2u] = p0[2] * inv_sc; dst[kt * 8u + 3u] = p0[3] * inv_sc;
        dst[kt * 8u + 4u] = p1[0] * inv_sc; dst[kt * 8u + 5u] = p1[1] * inv_sc;
        dst[kt * 8u + 6u] = p1[2] * inv_sc; dst[kt * 8u + 7u] = p1[3] * inv_sc;
    }
}

template <uint32_t B, uint32_t LUT_N, uint32_t C_SHIFT, uint32_t H_SHIFT, uint32_t G, uint32_t KC, uint32_t K_KEEP>
__global__ void prism_encode_v3_kernel(
    const float* __restrict__ W_streams,
    const uint32_t* __restrict__ lut_u32,
    const uint2* __restrict__ c64,
    const uint16_t* __restrict__ grid_cands,
    uint32_t* __restrict__ scratch_hist,
    uint32_t* __restrict__ out_packed,
    EncParams P
) {
    const uint32_t gid = blockIdx.x;
    if (gid >= P.n_streams) return;
    const uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u;
    const uint32_t stream_id = P.stream_off + gid;
    const uint32_t strip_id  = stream_id >> 5;
    const uint32_t my_lane   = stream_id & 31u;
    const uint32_t K_len     = P.C >> 1;
    const uint32_t n_pairs   = K_len >> 1;
    constexpr uint32_t BEAM     = 256u;
    constexpr uint32_t N_SYM    = 1u << (2u * B);
    constexpr uint32_t SYM_MASK = N_SYM - 1u;
    constexpr uint32_t C_MASK   = (1u << C_SHIFT) - 1u;
    constexpr float RMAX        = 3.2f;
    constexpr float INV_STEP    = static_cast<float>(G) / (2.0f * RMAX);

    const float* y_ptr = W_streams + static_cast<uint64_t>(stream_id) * K_len;
    uint32_t* hist_ptr = scratch_hist + static_cast<uint64_t>(gid) * n_pairs * 256ull;

    __shared__ float cur_cost[256], nxt_cost[256];
    __shared__ uint32_t cur_state[256], nxt_state[256];
    __shared__ uint32_t cur_hprev[256], nxt_hprev[256];
    __shared__ float sg_min[4], sg_sum[4], cmin_tg, inv_step_tg;
    __shared__ uint32_t bins[64], sg_scan[4], cut_bin_tg, lt_base_tg, eq_quota_tg;

    if (tid == 0u) {
        cur_cost[0]  = 0.0f;
        cur_state[0] = 0x13579BDFu ^ (my_lane * 0x9E3779B1u);
        cur_hprev[0] = 0u;
    }
    __syncthreads();
    uint32_t cur_n = 1u;

    for (uint32_t pair = 0u; pair < n_pairs; ++pair) {
        float x0 = y_ptr[pair * 2u];
        float y0 = y_ptr[pair * 2u + 1u];
        bool is_odd = (pair & 1u) != 0u;

        if (cur_n * K_KEEP <= BEAM) {
            if (tid < cur_n) {
                float bc = cur_cost[tid];
                uint32_t st = cur_state[tid];
                uint32_t h = st * 0x9E3779B1u;
                uint32_t h_c = is_odd ? cur_hprev[tid] : h;
                uint2 cs = __ldg(&c64[(h_c >> 16) & 63u]);
                __half2 c_s = enc_u32_to_h2(cs.x), c_t = enc_u32_to_h2(cs.y);
                float2 fs = __half22float2(c_s), ft = __half22float2(c_t);
                uint32_t h_lut = h >> H_SHIFT;
                uint32_t c_idx = h_lut & C_MASK;
                uint32_t sym_xor = (h_lut >> C_SHIFT) & SYM_MASK;
                float tx = (x0 - ft.x) / fs.x, ty = (y0 - ft.y) / fs.y;
                int gx = max(0, min(static_cast<int>(G) - 1, static_cast<int>((tx + RMAX) * INV_STEP)));
                int gy = max(0, min(static_cast<int>(G) - 1, static_cast<int>((ty + RMAX) * INV_STEP)));
                uint64_t gbase = ((static_cast<uint64_t>(c_idx) * G + gy) * G + gx) * KC;

                float bd[K_KEEP]; uint16_t bs[K_KEEP];
                #pragma unroll
                for (uint32_t k = 0; k < K_KEEP; ++k) { bd[k] = 1e30f; bs[k] = 0; }
                #pragma unroll
                for (uint32_t k = 0; k < KC; ++k) {
                    uint32_t i_sub = __ldg(&grid_cands[gbase + k]);
                    uint32_t idx = (i_sub << C_SHIFT) | c_idx;
                    __half2 q = __hfma2(enc_u32_to_h2(__ldg(&lut_u32[idx])), c_s, c_t);
                    float2 fq = __half22float2(q);
                    float dx = x0 - fq.x, dy = y0 - fq.y, d = dx * dx + dy * dy;
                    if (d < bd[K_KEEP - 1u]) {
                        bd[K_KEEP - 1u] = d; bs[K_KEEP - 1u] = static_cast<uint16_t>(i_sub ^ sym_xor);
                        #pragma unroll
                        for (uint32_t j = K_KEEP - 1u; j > 0u; --j) {
                            if (bd[j] < bd[j - 1u]) {
                                float td = bd[j]; bd[j] = bd[j - 1u]; bd[j - 1u] = td;
                                uint16_t ts = bs[j]; bs[j] = bs[j - 1u]; bs[j - 1u] = ts;
                            }
                        }
                    }
                }
                uint32_t st_sh = st << (2u * B);
                uint32_t next_hp = is_odd ? 0u : h;
                #pragma unroll
                for (uint32_t k = 0; k < K_KEEP; ++k) {
                    uint32_t dst = tid * K_KEEP + k;
                    uint32_t sym = bs[k];
                    nxt_cost[dst]  = bc + bd[k];
                    nxt_state[dst] = st_sh | sym;
                    nxt_hprev[dst] = next_hp;
                    hist_ptr[static_cast<uint64_t>(pair) * 256ull + dst] = tid | (sym << 8);
                }
            }
            cur_n *= K_KEEP;
            __syncthreads();
            #pragma unroll
            for (uint32_t s = 0; s < 2u; ++s) {
                uint32_t i = tid + s * 128u;
                if (i < cur_n) { cur_cost[i] = nxt_cost[i]; cur_state[i] = nxt_state[i]; cur_hprev[i] = nxt_hprev[i]; }
            }
            __syncthreads();
            continue;
        }

        float r_bc[2]; uint32_t r_st_sh[2], r_hp[2];
        float r_d[2][K_KEEP]; uint16_t r_sym[2][K_KEEP];
        float my_min = 1e30f, my_sum = 0.0f;

        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s) {
            uint32_t pi = tid + s * 128u;
            if (pi < cur_n) {
                float bc = cur_cost[pi];
                uint32_t st = cur_state[pi];
                uint32_t h = st * 0x9E3779B1u;
                uint32_t h_c = is_odd ? cur_hprev[pi] : h;
                uint2 cs = __ldg(&c64[(h_c >> 16) & 63u]);
                __half2 c_s = enc_u32_to_h2(cs.x), c_t = enc_u32_to_h2(cs.y);
                float2 fs = __half22float2(c_s), ft = __half22float2(c_t);
                uint32_t h_lut = h >> H_SHIFT;
                uint32_t c_idx = h_lut & C_MASK;
                uint32_t sym_xor = (h_lut >> C_SHIFT) & SYM_MASK;
                float tx = (x0 - ft.x) / fs.x, ty = (y0 - ft.y) / fs.y;
                int gx = max(0, min(static_cast<int>(G) - 1, static_cast<int>((tx + RMAX) * INV_STEP)));
                int gy = max(0, min(static_cast<int>(G) - 1, static_cast<int>((ty + RMAX) * INV_STEP)));
                uint64_t gbase = ((static_cast<uint64_t>(c_idx) * G + gy) * G + gx) * KC;

                float bd[K_KEEP]; uint16_t bs[K_KEEP];
                #pragma unroll
                for (uint32_t k = 0; k < K_KEEP; ++k) { bd[k] = 1e30f; bs[k] = 0; }
                #pragma unroll
                for (uint32_t k = 0; k < KC; ++k) {
                    uint32_t i_sub = __ldg(&grid_cands[gbase + k]);
                    uint32_t idx = (i_sub << C_SHIFT) | c_idx;
                    __half2 q = __hfma2(enc_u32_to_h2(__ldg(&lut_u32[idx])), c_s, c_t);
                    float2 fq = __half22float2(q);
                    float dx = x0 - fq.x, dy = y0 - fq.y, d = dx * dx + dy * dy;
                    if (d < bd[K_KEEP - 1u]) {
                        bd[K_KEEP - 1u] = d; bs[K_KEEP - 1u] = static_cast<uint16_t>(i_sub ^ sym_xor);
                        #pragma unroll
                        for (uint32_t j = K_KEEP - 1u; j > 0u; --j) {
                            if (bd[j] < bd[j - 1u]) {
                                float td = bd[j]; bd[j] = bd[j - 1u]; bd[j - 1u] = td;
                                uint16_t ts = bs[j]; bs[j] = bs[j - 1u]; bs[j - 1u] = ts;
                            }
                        }
                    }
                }
                r_bc[s] = bc; r_st_sh[s] = st << (2u * B); r_hp[s] = is_odd ? 0u : h;
                my_min = fminf(my_min, bc + bd[0]);
                #pragma unroll
                for (uint32_t k = 0; k < K_KEEP; ++k) { r_d[s][k] = bd[k]; r_sym[s][k] = bs[k]; my_sum += bc + bd[k]; }
            }
        }

        float smin = enc_warp_min(my_min);
        float ssum = enc_warp_sum(my_sum);
        if (lane == 0u) { sg_min[sg] = smin; sg_sum[sg] = ssum; }
        if (tid < 64u) bins[tid] = 0u;
        __syncthreads();

        if (tid == 0u) {
            float cmin = fminf(fminf(sg_min[0], sg_min[1]), fminf(sg_min[2], sg_min[3]));
            float cmean = ((sg_sum[0] + sg_sum[1]) + (sg_sum[2] + sg_sum[3])) / static_cast<float>(cur_n * K_KEEP);
            float span = fmaxf(1e-4f, (cmean - cmin) * 1.15f);
            cmin_tg = cmin; inv_step_tg = 63.0f / span;
        }
        __syncthreads();
        float cmin = cmin_tg, inv_step = inv_step_tg;

        uint8_t r_bi[2][K_KEEP];
        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s) {
            uint32_t pi = tid + s * 128u;
            if (pi < cur_n) {
                float bc = r_bc[s] - cmin;
                #pragma unroll
                for (uint32_t k = 0; k < K_KEEP; ++k) {
                    uint32_t bi = min(63u, static_cast<uint32_t>(fmaxf(0.0f, (bc + r_d[s][k]) * inv_step)));
                    r_bi[s][k] = static_cast<uint8_t>(bi);
                    atomicAdd(&bins[bi], 1u);
                }
            }
        }
        __syncthreads();

        if (tid == 0u) {
            uint32_t acc = 0u, cb = 63u, lb = 0u, eq = BEAM;
            bool found = false;
            for (uint32_t i = 0; i < 64u; ++i) {
                uint32_t cnt = bins[i];
                if (!found && acc + cnt >= BEAM) { cb = i; lb = acc; eq = BEAM - acc; found = true; }
                acc += cnt;
            }
            cut_bin_tg = cb; lt_base_tg = lb; eq_quota_tg = eq;
        }
        __syncthreads();
        uint32_t cut_bin = cut_bin_tg, lt_base = lt_base_tg, eq_quota = eq_quota_tg;

        // Deterministic block-wide exclusive scan over (my_eq << 16 | my_lt) in fixed (tid, s, k) order.
        uint32_t my_lt = 0u, my_eq = 0u;
        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s) {
            uint32_t pi = tid + s * 128u;
            if (pi < cur_n) {
                #pragma unroll
                for (uint32_t k = 0; k < K_KEEP; ++k) {
                    uint32_t bi = r_bi[s][k];
                    if (bi < cut_bin) my_lt++;
                    else if (bi == cut_bin) my_eq++;
                }
            }
        }
        uint32_t packed = (my_eq << 16) | my_lt;
        uint32_t incl = packed;
        #pragma unroll
        for (uint32_t d = 1u; d < 32u; d <<= 1u) {
            uint32_t v = __shfl_up_sync(0xFFFFFFFFu, incl, d);
            if (lane >= d) incl += v;
        }
        uint32_t excl = incl - packed;
        if (lane == 31u) sg_scan[sg] = incl;
        __syncthreads();
        uint32_t w_base = 0u;
        if (sg > 0u) w_base += sg_scan[0];
        if (sg > 1u) w_base += sg_scan[1];
        if (sg > 2u) w_base += sg_scan[2];
        uint32_t base_packed = excl + w_base;
        uint32_t pos_lt = base_packed & 0xFFFFu;
        uint32_t pos_eq = base_packed >> 16;

        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s) {
            uint32_t pi = tid + s * 128u;
            if (pi < cur_n) {
                float bc_raw = r_bc[s];
                uint32_t st_sh = r_st_sh[s], next_hp = r_hp[s];
                #pragma unroll
                for (uint32_t k = 0; k < K_KEEP; ++k) {
                    uint32_t bi = r_bi[s][k];
                    if (bi < cut_bin) {
                        uint32_t pos = pos_lt++;
                        if (pos < lt_base) {
                            uint32_t sym = r_sym[s][k];
                            nxt_cost[pos] = bc_raw + r_d[s][k]; nxt_state[pos] = st_sh | sym; nxt_hprev[pos] = next_hp;
                            hist_ptr[static_cast<uint64_t>(pair) * 256ull + pos] = pi | (sym << 8);
                        }
                    } else if (bi == cut_bin) {
                        uint32_t ep = pos_eq++;
                        if (ep < eq_quota) {
                            uint32_t pos = lt_base + ep;
                            uint32_t sym = r_sym[s][k];
                            nxt_cost[pos] = bc_raw + r_d[s][k]; nxt_state[pos] = st_sh | sym; nxt_hprev[pos] = next_hp;
                            hist_ptr[static_cast<uint64_t>(pair) * 256ull + pos] = pi | (sym << 8);
                        }
                    }
                }
            }
        }
        cur_n = BEAM;
        __syncthreads();
        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s) {
            uint32_t i = tid + s * 128u;
            cur_cost[i] = nxt_cost[i] - cmin; cur_state[i] = nxt_state[i]; cur_hprev[i] = nxt_hprev[i];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        uint32_t best_i = 0u; float best_c = 1e30f;
        for (uint32_t i = 0; i < BEAM; ++i) if (cur_cost[i] < best_c) { best_c = cur_cost[i]; best_i = i; }
        uint32_t idx = best_i;
        for (int pair = static_cast<int>(n_pairs) - 1; pair >= 0; --pair) {
            uint32_t rec = hist_ptr[static_cast<uint64_t>(pair) * 256ull + idx];
            hist_ptr[static_cast<uint64_t>(pair) * 256ull] = rec >> 8;
            idx = rec & 255u;
        }
        uint32_t n_bc = P.C >> 7;
        constexpr uint32_t wpb = B * 2u;
        uint64_t strip_word_base = static_cast<uint64_t>(strip_id) * n_bc * wpb * 32ull;
        for (uint32_t bc = 0u; bc < n_bc; ++bc) {
            uint32_t words[10];
            #pragma unroll
            for (uint32_t w = 0; w < wpb; ++w) words[w] = 0u;
            uint32_t bit_pos = 0u;
            for (uint32_t p = 0u; p < 32u; ++p) {
                uint32_t sym = hist_ptr[static_cast<uint64_t>(bc * 32u + p) * 256ull];
                for (int k = static_cast<int>(2u * B) - 1; k >= 0; --k) {
                    if ((sym >> k) & 1u) words[bit_pos >> 5] |= (1u << (31u - (bit_pos & 31u)));
                    bit_pos++;
                }
            }
            uint32_t* base = out_packed + strip_word_base + static_cast<uint64_t>(bc) * wpb * 32ull;
            if constexpr (B == 4u) {
                base[my_lane * 4u + 0u] = words[0]; base[my_lane * 4u + 1u] = words[1];
                base[my_lane * 4u + 2u] = words[2]; base[my_lane * 4u + 3u] = words[3];
                base[128u + my_lane * 4u + 0u] = words[4]; base[128u + my_lane * 4u + 1u] = words[5];
                base[128u + my_lane * 4u + 2u] = words[6]; base[128u + my_lane * 4u + 3u] = words[7];
            } else {
                #pragma unroll
                for (uint32_t w = 0; w < wpb; ++w) base[w * 32u + my_lane] = words[w];
            }
        }
    }
}

__global__ void prism_verify_v3_snr_kernel(
    const uint32_t* __restrict__ W_packed, const __half* __restrict__ scales,
    const uint32_t* __restrict__ lut_u32, const uint2* __restrict__ c64,
    const float* __restrict__ W_rot, float* __restrict__ out_se, MatParams P
) {
    uint32_t strip = blockIdx.x, lane = threadIdx.x;
    int2 home = enc_frag_home(lane);
    uint32_t n_bc = P.C >> 7, b = P.b_bits, wpb = b * 2u;
    const uint32_t* w_strip = W_packed + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const __half* sc_row = scales + static_cast<uint64_t>(strip >> 3) * n_bc;
    uint32_t st = 0x13579BDFu ^ (lane * 0x9E3779B1u);
    float sig = 0.0f, err = 0.0f;

    for (uint32_t bc = 0u; bc < n_bc; ++bc) {
        float sc = __half2float(sc_row[bc]);
        const uint32_t* base = w_strip + static_cast<uint64_t>(bc) * wpb * 32ull;
        uint32_t words[10];
        if (b == 4u) {
            for (uint32_t k = 0; k < 4u; ++k) words[k] = base[lane * 4u + k];
            for (uint32_t k = 0; k < 4u; ++k) words[k + 4u] = base[128u + lane * 4u + k];
        } else {
            for (uint32_t w = 0; w < wpb; ++w) words[w] = base[w * 32u + lane];
        }
        uint32_t bit_pos = 0u;
        for (uint32_t kt = 0u; kt < 8u; ++kt) {
            uint32_t syms[4];
            for (uint32_t p = 0; p < 4u; ++p) {
                uint32_t sym = 0u;
                for (uint32_t k = 0; k < 2u * b; ++k) {
                    sym = (sym << 1) | ((words[bit_pos >> 5] >> (31u - (bit_pos & 31u))) & 1u);
                    bit_pos++;
                }
                syms[p] = sym;
            }
            __half2 q0a, q0b, q1a, q1b;
            uint32_t h0 = st * 0x9E3779B1u;
            uint32_t st1 = (st << (2u * b)) | syms[0], h1 = st1 * 0x9E3779B1u;
            uint32_t st2 = (st1 << (2u * b)) | syms[1], h2 = st2 * 0x9E3779B1u;
            uint32_t st3 = (st2 << (2u * b)) | syms[2], h3 = st3 * 0x9E3779B1u;
            st = (st3 << (2u * b)) | syms[3];
            uint2 c0 = c64[(h0 >> 16) & 63u], c2 = c64[(h2 >> 16) & 63u];
            __half2 c0s = enc_u32_to_h2(c0.x), c0t = enc_u32_to_h2(c0.y);
            __half2 c2s = enc_u32_to_h2(c2.x), c2t = enc_u32_to_h2(c2.y);
            if (b == 3u) {
                q0a = __hfma2(enc_u32_to_h2(lut_u32[((syms[0] << 2) ^ (h0 >> 24)) & 255u]), c0s, c0t);
                q0b = __hfma2(enc_u32_to_h2(lut_u32[((syms[1] << 2) ^ (h1 >> 24)) & 255u]), c0s, c0t);
                q1a = __hfma2(enc_u32_to_h2(lut_u32[((syms[2] << 2) ^ (h2 >> 24)) & 255u]), c2s, c2t);
                q1b = __hfma2(enc_u32_to_h2(lut_u32[((syms[3] << 2) ^ (h3 >> 24)) & 255u]), c2s, c2t);
            } else if (b == 4u) {
                q0a = __hfma2(enc_u32_to_h2(lut_u32[((syms[0] << 2) ^ (h0 >> 22)) & 1023u]), c0s, c0t);
                q0b = __hfma2(enc_u32_to_h2(lut_u32[((syms[1] << 2) ^ (h1 >> 22)) & 1023u]), c0s, c0t);
                q1a = __hfma2(enc_u32_to_h2(lut_u32[((syms[2] << 2) ^ (h2 >> 22)) & 1023u]), c2s, c2t);
                q1b = __hfma2(enc_u32_to_h2(lut_u32[((syms[3] << 2) ^ (h3 >> 22)) & 1023u]), c2s, c2t);
            } else {
                q0a = __hfma2(enc_u32_to_h2(lut_u32[(syms[0] ^ (h0 >> 22)) & 1023u]), c0s, c0t);
                q0b = __hfma2(enc_u32_to_h2(lut_u32[(syms[1] ^ (h1 >> 22)) & 1023u]), c0s, c0t);
                q1a = __hfma2(enc_u32_to_h2(lut_u32[(syms[2] ^ (h2 >> 22)) & 1023u]), c2s, c2t);
                q1b = __hfma2(enc_u32_to_h2(lut_u32[(syms[3] ^ (h3 >> 22)) & 1023u]), c2s, c2t);
            }
            float2 fa0 = __half22float2(q0a), fb0 = __half22float2(q0b);
            float2 fa1 = __half22float2(q1a), fb1 = __half22float2(q1b);
            uint32_t cc0 = bc * 128u + kt * 16u + static_cast<uint32_t>(home.x);
            uint32_t r0 = strip * 16u + static_cast<uint32_t>(home.y), r1 = r0 + 8u;
            const float* p0 = W_rot + static_cast<uint64_t>(r0) * P.C + cc0;
            const float* p1 = W_rot + static_cast<uint64_t>(r1) * P.C + cc0;
            float a0 = p0[0], a1 = p0[1], a2 = p0[2], a3 = p0[3];
            float b0 = p1[0], b1 = p1[1], b2 = p1[2], b3 = p1[3];
            float d00 = a0 - fa0.x * sc, d01 = a1 - fa0.y * sc, d02 = a2 - fb0.x * sc, d03 = a3 - fb0.y * sc;
            float d10 = b0 - fa1.x * sc, d11 = b1 - fa1.y * sc, d12 = b2 - fb1.x * sc, d13 = b3 - fb1.y * sc;
            sig += ((a0*a0 + a1*a1) + (a2*a2 + a3*a3)) + ((b0*b0 + b1*b1) + (b2*b2 + b3*b3));
            err += ((d00*d00 + d01*d01) + (d02*d02 + d03*d03)) + ((d10*d10 + d11*d11) + (d12*d12 + d13*d13));
        }
    }
    sig = enc_warp_sum(sig); err = enc_warp_sum(err);
    if (lane == 0u) { out_se[strip * 2u + 0u] = sig; out_se[strip * 2u + 1u] = err; }
}

static inline void cuda_read_se_det(const float* d_se, uint32_t n_strips, double& out_sig, double& out_err) {
    std::vector<float> h(static_cast<size_t>(n_strips) * 2u);
    TFQ_CUDA_CHECK(cudaMemcpy(h.data(), d_se, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double s = 0.0, e = 0.0;
    for (uint32_t i = 0; i < n_strips; ++i) { s += h[i * 2u + 0u]; e += h[i * 2u + 1u]; }
    out_sig = s; out_err = e;
}

__global__ void requant_ple_q8_to_qb_kernel(
    const uint8_t* __restrict__ src_w, const uint16_t* __restrict__ src_s, const uint16_t* __restrict__ src_b,
    uint32_t* __restrict__ dst_w, uint16_t* __restrict__ dst_s, uint16_t* __restrict__ dst_b, PleParams P
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ng = P.cols >> 5, total_g = P.rows * ng;
    if (gid >= total_g) return;
    uint32_t r = gid / ng, g = gid - r * ng;
    float s_in = enc_bf16_to_f32(src_s[gid]), b_in = enc_bf16_to_f32(src_b[gid]);
    const uint8_t* wp = src_w + static_cast<uint64_t>(gid) * 32ull;
    float vals[32], vmin = 1e30f, vmax = -1e30f;
    for (uint32_t i = 0; i < 32u; ++i) {
        float v = fmaf(static_cast<float>(wp[i]), s_in, b_in);
        vals[i] = v; vmin = fminf(vmin, v); vmax = fmaxf(vmax, v);
    }
    uint32_t B = P.dst_bits;
    float qmax = static_cast<float>((1u << B) - 1u);
    uint16_t hb = enc_f32_to_bf16(vmin);
    float bf = enc_bf16_to_f32(hb);
    float sc = (vmax > bf) ? ((vmax - bf) / qmax) : 1.0f;
    uint16_t hs = enc_f32_to_bf16(sc);
    float sf = enc_bf16_to_f32(hs);
    float inv_s = (sf > 0.0f) ? (1.0f / sf) : 0.0f;
    dst_s[gid] = hs; dst_b[gid] = hb;
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    for (uint32_t i = 0; i < 32u; ++i) {
        float scaled = fminf(qmax, fmaxf(0.0f, rintf((vals[i] - bf) * inv_s)));
        uint32_t q = static_cast<uint32_t>(scaled);
        uint32_t bit = i * B, widx = bit >> 5, sh = bit & 31u;
        words[widx] |= (q << sh);
        if (sh + B > 32u) words[widx + 1u] |= (q >> (32u - sh));
    }
    uint64_t dst_off = static_cast<uint64_t>(r) * (ng * B) + static_cast<uint64_t>(g * B);
    for (uint32_t w = 0; w < B; ++w) dst_w[dst_off + w] = words[w];
}

inline void launch_prism_encode(
    uint32_t bits, uint32_t grid_n,
    const float* d_streams, const uint32_t* d_lut, const uint2* d_c64, const uint16_t* d_grid,
    uint32_t* d_hist, uint32_t* d_out, EncParams ep
) {
    if (bits == 3u)      prism_encode_v3_kernel<3u,  256u, 2u, 24u, 16u,  8u, 5u><<<grid_n, 128>>>(d_streams, d_lut, d_c64, d_grid, d_hist, d_out, ep);
    else if (bits == 4u) prism_encode_v3_kernel<4u, 1024u, 2u, 22u, 16u, 10u, 5u><<<grid_n, 128>>>(d_streams, d_lut, d_c64, d_grid, d_hist, d_out, ep);
    else                 prism_encode_v3_kernel<5u, 1024u, 0u, 22u, 32u, 12u, 6u><<<grid_n, 128>>>(d_streams, d_lut, d_c64, d_grid, d_hist, d_out, ep);
}
