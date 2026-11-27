#pragma once
// ============================================================================
// TFQ4 Cross-Platform Binary Format (PWV4 v4 + TFP4 v4, 2.001 bpw 8D E8P)
//   - Byte-identical on Apple Silicon Metal and NVIDIA CUDA (16 KiB-aligned)
//   - Shares exact 2D FWHT-128 rotation & 128x128 fp16 tile scales with TFQ3
//   - Deterministic data-free 256-entry positive-orthant 8D E8P codebook (4 KiB)
// ============================================================================
#include "../canonical_tables.hpp"
#include "../tfq_constants.hpp"
#include "../tfq_xxh64.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace tfq4 {

static constexpr uint32_t MAGIC_PWV4 = 0x34565750u; // "PWV4"
static constexpr uint32_t MAGIC_TFP4 = 0x34504654u; // "TFP4"
static constexpr uint32_t FORMAT_VER = 4u;
static constexpr size_t   ALIGN_16K  = 16384;

static constexpr uint32_t PWV4_TABLE_LEGACY_TRAINED = 0u; // legacy/default trained 256x8 (8 free sign bits)
static constexpr uint32_t PWV4_TABLE_TRAINED        = 1u; // explicit alias for trained 256x8 (8 free sign bits)
static constexpr uint32_t PWV4_TABLE_TRUE_E8P       = 2u; // true QuIP# E8P (E8 + 1/4, 256 abs + 7 sign + parity + coset)
static constexpr uint32_t PWV4_TABLE_TRUE_E8P_1024  = 3u; // true QuIP# E8P-1024 (E8 + 1/4, 1024 abs + 7 sign + parity + coset, 2.251 bpw)
static constexpr uint32_t PWV4_TABLE_MAX_KNOWN      = 3u;
static constexpr float    TRUE_E8P_BASE_SCALE       = 0.97f;
static constexpr float    TRUE_E8P1024_BASE_SCALE   = 0.84f;
static constexpr uint32_t TFQ4_BITS_200             = 2u;
static constexpr uint32_t TFQ4_BITS_225             = 225u;

struct PwV4E8Tables {
    uint32_t magic;
    uint32_t version;
    uint32_t k_codes;    // 256
    uint32_t dim;        // 8
    uint32_t table_kind; // 0=legacy/trained, 1=trained (alias), 2=true QuIP# E8P (E8 + 1/4)
    uint32_t pad0[11];
    uint16_t e8p_lut[256][8]; // fp16 positive-orthant 8D centroids (4,096 bytes)
    uint8_t  pad1[16384 - 64 - 4096];
};
static_assert(sizeof(PwV4E8Tables) == 16384, "PwV4E8Tables must be 16384 bytes");

struct PwV4E8Tables1024 {
    uint32_t magic;
    uint32_t version;
    uint32_t k_codes;    // 1024
    uint32_t dim;        // 8
    uint32_t table_kind; // 3=true QuIP# E8P-1024 (E8 + 1/4, first 256 entries bit-identical to table_kind=2)
    uint32_t pad0[11];
    uint16_t e8p_lut[1024][8]; // fp16 positive-orthant 8D centroids (16,384 bytes)
    uint8_t  pad1[32768 - 64 - 16384];
};
static_assert(sizeof(PwV4E8Tables1024) == 32768, "PwV4E8Tables1024 must be 32768 bytes");

struct Tfp4Header {
    uint32_t magic;
    uint32_t version;
    uint32_t num_tensors;
    uint32_t pad0;
    uint64_t off_meta;
    uint64_t off_scales;
    uint64_t off_weights;
    uint64_t total_scale_halfs;
    uint64_t total_weight_words;
    uint64_t pad1;
};
static_assert(sizeof(Tfp4Header) == 64, "Tfp4Header must be 64 bytes");

struct Tfp4TensorMeta {
    char     name[80];
    uint32_t R;          // padded multiple of 128
    uint32_t C;          // padded multiple of 128
    uint32_t b_bits;     // 2 (E8P 2.001 bpw, 8 words/128 cols) or 225 (E8P-1024 2.251 bpw, 9 words/128 cols)
    uint32_t sc_off;     // offset in fp16 scales
    uint64_t word_off;   // offset in uint32_t words
    uint64_t word_count; // (uint64_t)R * ((b_bits == 225) ? (C / 128) * 9 : (C / 16))
    float    snr_db;
    uint32_t pad[3];     // [0]=orig_R, [1]=orig_C, [2]=flags
};
static_assert(sizeof(Tfp4TensorMeta) == 128, "Tfp4TensorMeta must be 128 bytes");

static inline uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

static inline float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            int e = -14;
            while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            out = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

// Canonical 100% data-free 256-code positive-orthant 8D E8-initialized Voronoi
// Gaussian codebook (table_kind=0, reproducible bit-for-bit on Metal & CUDA).
// Regenerate via src/tools/tfq3_train_tables.cpp (Box-Muller over std::mt19937).
static inline void build_e8p_tables(PwV4E8Tables& t) {
    tfq_canonical::fill_canonical_pwv4(&t);
}

// Deterministic True QuIP# E8P (E8 + 1/4) table (table_kind=2):
//   - 256 base positive half-integer vectors in |D8 + 1/2| (227 with ||a||^2 <= 10
//     + 29 lowest-max-coordinate vectors from ||a||^2 = 12, exact in fp16).
//   - Codeword (u16): bits[7:0]=abs_idx, bits[14:8]=s[0..6], bit[15]=coset.
//     s[7] = popcount(s[0..6]) ^ (sum_d floor(a[d]) & 1) enforces s*a in D8 + 1/2;
//     q = ((-1)^coset) * (s * a - 1/4) spans 65,536 unique points strictly in E8 + 1/4.
static inline void build_true_e8p_tables(PwV4E8Tables& t) {
    std::memset(&t, 0, sizeof(t));
    t.magic      = MAGIC_PWV4;
    t.version    = FORMAT_VER;
    t.k_codes    = 256u;
    t.dim        = 8u;
    t.table_kind = PWV4_TABLE_TRUE_E8P;

    struct Cand {
        int n2_x4;
        int max_m;
        std::array<uint8_t, 8> m;
    };
    std::vector<Cand> cands;
    cands.reserve(512);
    // Pure integer enumeration over m_d in {0, 1, 2} (entries {0.5, 1.5, 2.5}); 3^8 = 6561.
    for (int code = 0; code < 6561; ++code) {
        int tmp = code, n2 = 0, mx = 0;
        std::array<uint8_t, 8> m{};
        for (int d = 7; d >= 0; --d) {
            int md = tmp % 3;
            tmp /= 3;
            m[d] = static_cast<uint8_t>(md);
            int v2 = 2 * md + 1;
            n2 += v2 * v2;
            if (md > mx) mx = md;
        }
        if (n2 <= 48) cands.push_back(Cand{n2, mx, m});
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        if (a.n2_x4 != b.n2_x4) return a.n2_x4 < b.n2_x4;
        if (a.max_m != b.max_m) return a.max_m < b.max_m;
        return a.m < b.m;
    });
    for (int k = 0; k < 256; ++k) {
        for (int d = 0; d < 8; ++d) {
            t.e8p_lut[k][d] = f32_to_f16_bits(static_cast<float>(cands[k].m[d]) + 0.5f);
        }
    }
}

// Deterministic True QuIP# E8P-1024 (E8 + 1/4) table (table_kind=3, 32,768 bytes):
//   - Enumerates all m_d in {0, 1, 2, 3} (a_d = m_d + 0.5 in {0.5, 1.5, 2.5, 3.5}, 4^8 = 65536 tuples),
//     keeps sum((2*m_d + 1)^2) <= 68, sorts by (n2_x4, max_m, lex m), takes first 1024.
//   - First 256 entries are 100% bit-identical to build_true_e8p_tables (table_kind=2), so one 32 KiB
//     table_kind=3 file decodes BOTH 2.001 bpw (0..255) and 2.251 bpw (0..1023) seamlessly.
static inline void build_true_e8p_1024_tables(PwV4E8Tables1024& t) {
    std::memset(&t, 0, sizeof(t));
    t.magic      = MAGIC_PWV4;
    t.version    = FORMAT_VER;
    t.k_codes    = 1024u;
    t.dim        = 8u;
    t.table_kind = PWV4_TABLE_TRUE_E8P_1024;

    struct Cand {
        int n2_x4;
        int max_m;
        std::array<uint8_t, 8> m;
    };
    std::vector<Cand> cands;
    cands.reserve(2048);
    // Pure integer enumeration over m_d in {0, 1, 2, 3} (entries {0.5, 1.5, 2.5, 3.5}); 4^8 = 65536.
    for (int code = 0; code < 65536; ++code) {
        int tmp = code, n2 = 0, mx = 0;
        std::array<uint8_t, 8> m{};
        for (int d = 7; d >= 0; --d) {
            int md = tmp & 3;
            tmp >>= 2;
            m[d] = static_cast<uint8_t>(md);
            int v2 = 2 * md + 1;
            n2 += v2 * v2;
            if (md > mx) mx = md;
        }
        if (n2 <= 68) cands.push_back(Cand{n2, mx, m});
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        if (a.n2_x4 != b.n2_x4) return a.n2_x4 < b.n2_x4;
        if (a.max_m != b.max_m) return a.max_m < b.max_m;
        return a.m < b.m;
    });
    for (int k = 0; k < 1024; ++k) {
        for (int d = 0; d < 8; ++d) {
            t.e8p_lut[k][d] = f32_to_f16_bits(static_cast<float>(cands[k].m[d]) + 0.5f);
        }
    }
}

static inline uint8_t true_e8p_abs_parity(const uint16_t row[8]) {
    uint32_t s = 0u;
    for (int d = 0; d < 8; ++d) {
        float v = f16_bits_to_f32(row[d]);
        s += static_cast<uint32_t>(std::lround(v - 0.5f));
    }
    return static_cast<uint8_t>(s & 1u);
}

static inline uint64_t align_up_16k(uint64_t x) {
    return (x + (uint64_t)ALIGN_16K - 1ull) & ~((uint64_t)ALIGN_16K - 1ull);
}

static inline uint64_t tfp4_words_per_row(uint32_t pad_C, uint32_t b_bits = TFQ4_BITS_200) {
    return (b_bits == TFQ4_BITS_225)
        ? ((uint64_t)(pad_C / 128u) * 9ull)
        : ((uint64_t)(pad_C / 128u) * 8ull);
}

static inline uint64_t tfp4_tensor_bytes(uint32_t pad_R, uint32_t pad_C, uint32_t b_bits = TFQ4_BITS_200) {
    uint64_t words = (uint64_t)pad_R * tfp4_words_per_row(pad_C, b_bits);
    uint64_t sc = (uint64_t)(pad_R / 128u) * (pad_C / 128u);
    return words * 4ull + sc * 2ull;
}

static inline uint64_t tfpw_proj_words(uint32_t R, uint32_t C, uint32_t b_bits) {
    if (b_bits == TFQ4_BITS_200) return (uint64_t)R * (C / 16u);
    if (b_bits == TFQ4_BITS_225) return (uint64_t)R * (C / 128u) * 9ull;
    return (uint64_t)(R / 16u) * (C / 128u) * (b_bits * 2u) * 32ull;
}

// Write pw_v4_e8p_tables.bin for hybrid MoE checkpoints without downgrading an existing 32 KiB table_kind=3 file.
static inline void write_hybrid_pwv4_tables(const std::string& path, bool need_1024) {
    if (!need_1024) {
        std::ifstream chk(path, std::ios::binary | std::ios::ate);
        if (chk.good() && chk.tellg() == (std::streamoff)sizeof(PwV4E8Tables1024)) {
            chk.seekg(0);
            uint32_t hdr[5] = {};
            if (chk.read(reinterpret_cast<char*>(hdr), sizeof(hdr)) &&
                hdr[0] == MAGIC_PWV4 && hdr[1] == FORMAT_VER &&
                hdr[2] == 1024u && hdr[3] == 8u && hdr[4] == PWV4_TABLE_TRUE_E8P_1024) {
                return; // existing 32 KiB table_kind=3 already decodes both 2b and 2.25b
            }
        }
    }
    std::string tmp = path + ".tmp";
    std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
    if (!ofs.good()) {
        fprintf(stderr, "Cannot write PWV4 tables: %s\n", tmp.c_str());
        std::exit(1);
    }
    if (need_1024) {
        PwV4E8Tables1024 t{};
        build_true_e8p_1024_tables(t);
        ofs.write(reinterpret_cast<const char*>(&t), sizeof(t));
    } else {
        PwV4E8Tables t{};
        build_true_e8p_tables(t);
        ofs.write(reinterpret_cast<const char*>(&t), sizeof(t));
    }
    ofs.close();
    if (ofs.fail() || std::rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "Write/rename error on PWV4 tables: %s\n", path.c_str());
        std::exit(1);
    }
}

static inline uint64_t write_tfp4_shard(
    const std::string& path,
    const std::vector<Tfp4TensorMeta>& metas,
    const std::vector<uint16_t>& scales,
    const std::vector<uint32_t>& words
) {
    const uint64_t off_meta = sizeof(Tfp4Header);
    const uint64_t off_sc   = align_up_16k(off_meta + (uint64_t)metas.size() * sizeof(Tfp4TensorMeta));
    const uint64_t off_wt   = align_up_16k(off_sc + (uint64_t)scales.size() * 2ull);
    const uint64_t tot_sz   = align_up_16k(off_wt + (uint64_t)words.size() * 4ull);
    Tfp4Header hdr{};
    hdr.magic = MAGIC_TFP4;
    hdr.version = FORMAT_VER;
    hdr.num_tensors = (uint32_t)metas.size();
    hdr.pad0 = tfq_xxh64::FLAG_XXH64_PRESENT;
    hdr.off_meta = off_meta;
    hdr.off_scales = off_sc;
    hdr.off_weights = off_wt;
    hdr.total_scale_halfs = scales.size();
    hdr.total_weight_words = words.size();
    hdr.pad1 = tfq_xxh64::hash_sections3(
        metas.data(), metas.size() * sizeof(Tfp4TensorMeta),
        scales.data(), scales.size() * 2ull,
        words.data(), words.size() * 4ull
    );

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.good()) {
        fprintf(stderr, "Cannot write TFP4 shard: %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<char> zeros(ALIGN_16K, 0);
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!metas.empty()) ofs.write(reinterpret_cast<const char*>(metas.data()), (std::streamsize)(metas.size() * sizeof(Tfp4TensorMeta)));
    uint64_t pos = off_meta + (uint64_t)metas.size() * sizeof(Tfp4TensorMeta);
    if (off_sc > pos) ofs.write(zeros.data(), (std::streamsize)(off_sc - pos));
    if (!scales.empty()) ofs.write(reinterpret_cast<const char*>(scales.data()), (std::streamsize)(scales.size() * 2ull));
    pos = off_sc + (uint64_t)scales.size() * 2ull;
    if (off_wt > pos) ofs.write(zeros.data(), (std::streamsize)(off_wt - pos));
    if (!words.empty()) ofs.write(reinterpret_cast<const char*>(words.data()), (std::streamsize)(words.size() * 4ull));
    pos = off_wt + (uint64_t)words.size() * 4ull;
    if (tot_sz > pos) ofs.write(zeros.data(), (std::streamsize)(tot_sz - pos));
    ofs.close();
    if (ofs.fail()) {
        fprintf(stderr, "Write error on TFP4 shard: %s\n", path.c_str());
        std::exit(1);
    }
    return tot_sz;
}

} // namespace tfq4
