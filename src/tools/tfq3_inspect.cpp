// Zero-dependency TFQ3 / TFQ4 binary shard validator and inspector.
// Supports pw_v3_tables.bin ("PWV3"), pw_v4_e8p_tables.bin ("PWV4"),
// dense/non-MoE shards ("TFPN"), MoE shards ("TFPW"), and 2-bit E8P shards ("TFP4").
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>
#include "../converters/tfq_xxh64.hpp"

static constexpr uint32_t MAGIC_PWV3 = 0x33565750u;
static constexpr uint32_t MAGIC_PWV4 = 0x34565750u;
static constexpr uint32_t MAGIC_TFPN = 0x4E504654u;
static constexpr uint32_t MAGIC_TFPW = 0x57504654u;
static constexpr uint32_t MAGIC_TFP4 = 0x34504654u;
static constexpr uint32_t FORMAT_VER = 3u;
static constexpr uint32_t TFQ4_VER   = 4u;
static constexpr uint64_t ALIGN_16K  = 16384ULL;

struct TfpnHeader {
    uint32_t magic, version, num_tensors, pad0;
    uint64_t off_meta, off_scales, off_weights, total_scale_halfs, total_weight_words, pad1;
};
static_assert(sizeof(TfpnHeader) == 64, "TfpnHeader must be 64 bytes");

struct TfpnTensorMeta {
    char name[80];
    uint32_t R, C, b_bits, sc_off;
    uint64_t word_off, word_count;
    float snr_db;
    uint32_t pad[3];
};
static_assert(sizeof(TfpnTensorMeta) == 128, "TfpnTensorMeta must be 128 bytes");

struct TfpwHeader {
    uint32_t magic, version, layer_idx, num_experts, num_tensors, hidden_size, moe_inter, reserved0;
    uint64_t raw_fp8_bytes, stream_words, scale_halfs, off_meta, off_scales, off_weights;
    uint64_t reserved1[4];
    char meta_tag[32];
};
static_assert(sizeof(TfpwHeader) == 144, "TfpwHeader must be 144 bytes");

struct TfpwExpertMeta {
    uint32_t b_bits, heat_rank;
    uint64_t gate_word_off, up_word_off, down_word_off;
    uint32_t gate_sc_off, up_sc_off, down_sc_off;
    float gate_snr_db, up_snr_db, down_snr_db;
    uint32_t reserved[2];
};
static_assert(sizeof(TfpwExpertMeta) == 64, "TfpwExpertMeta must be 64 bytes");

static uint64_t file_size(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) return 0;
    return (uint64_t)st.st_size;
}

static bool is_dir(const std::string& path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool f16_is_finite(uint16_t h) {
    return (h & 0x7C00u) != 0x7C00u;
}

static bool f16_is_nonneg_finite(uint16_t h) {
    if (!f16_is_finite(h)) return false;
    return (h & 0x8000u) == 0u || (h & 0x7FFFu) == 0u;
}

static bool check_range(uint64_t off, uint64_t len, uint64_t total) {
    return off <= total && len <= total - off;
}

// Element count -> byte count; false if the product does not fit in 64 bits.
static bool checked_bytes(uint64_t count, uint64_t elem, uint64_t& out) {
    if (count > UINT64_MAX / elem) return false;
    out = count * elem;
    return true;
}

// pad[0]/pad[1] hold orig_R/orig_C (0 = unpadded); padding is to the next multiple of 128.
static bool check_orig_dim(uint32_t orig, uint32_t padded) {
    return orig == 0 || (orig <= padded && padded - orig < 128u);
}

static bool extract_tensor_name(const char raw[80], std::string& out) {
    size_t len = 0;
    while (len < 80 && raw[len] != '\0') ++len;
    if (len == 0 || len == 80) return false;
    out.assign(raw, len);
    return true;
}

static bool inspect_tables(const std::string& path) {
    uint64_t sz = file_size(path);
    if (sz < 10816ULL) {
        fprintf(stderr, "FAIL %s: truncated (%llu bytes)\n", path.c_str(), (unsigned long long)sz);
        return false;
    }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    uint8_t buf[10816];
    if (fread(buf, 1, sizeof(buf), f) != sizeof(buf)) {
        fclose(f);
        fprintf(stderr, "FAIL %s: short read\n", path.c_str());
        return false;
    }
    fclose(f);

    uint32_t hdr[2];
    float ab[6];
    std::memcpy(hdr, buf, 8);
    std::memcpy(ab, buf + 8, 24);
    if (hdr[0] != MAGIC_PWV3 || hdr[1] != FORMAT_VER) {
        fprintf(stderr, "FAIL %s: bad magic/version (0x%08x v%u)\n", path.c_str(), hdr[0], hdr[1]);
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(ab[i])) {
            fprintf(stderr, "FAIL %s: non-finite table scale\n", path.c_str());
            return false;
        }
    }
    const size_t n_halfs = (10816 - 64) / 2;
    for (size_t i = 0; i < n_halfs; ++i) {
        uint16_t h;
        std::memcpy(&h, buf + 64 + i * 2, 2);
        if (!f16_is_finite(h)) {
            fprintf(stderr, "FAIL %s: non-finite fp16 entry in PWV3 LUT/c64 at half[%zu]\n", path.c_str(), i);
            return false;
        }
    }
    printf("OK   %-42s PWV3 v%u | %llu bytes | 3b(a=%.3f,b=%.3f) 4b(a=%.3f,b=%.3f) 5b(a=%.3f,b=%.3f)\n",
           path.c_str(), hdr[1], (unsigned long long)sz, ab[0], ab[1], ab[2], ab[3], ab[4], ab[5]);
    return true;
}

static bool inspect_tfpn(const std::string& path, bool verbose, std::unordered_set<std::string>* seen_names) {
    uint64_t sz = file_size(path);
    if (sz < sizeof(TfpnHeader)) { fprintf(stderr, "FAIL %s: too small\n", path.c_str()); return false; }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    TfpnHeader h{};
    if (fread(&h, sizeof(h), 1, f) != 1) {
        fclose(f); fprintf(stderr, "FAIL %s: short read\n", path.c_str()); return false;
    }
    if (h.magic != MAGIC_TFPN || h.version != FORMAT_VER) {
        fclose(f); fprintf(stderr, "FAIL %s: bad magic/version\n", path.c_str()); return false;
    }
    if ((h.off_scales % ALIGN_16K) != 0 || (h.off_weights % ALIGN_16K) != 0 || (sz % ALIGN_16K) != 0) {
        fclose(f); fprintf(stderr, "FAIL %s: scales/weights/file end not 16 KiB aligned\n", path.c_str()); return false;
    }
    uint64_t meta_bytes = (uint64_t)h.num_tensors * sizeof(TfpnTensorMeta);
    uint64_t sc_bytes = 0, w_bytes = 0;
    if (!checked_bytes(h.total_scale_halfs, 2ULL, sc_bytes) || !checked_bytes(h.total_weight_words, 4ULL, w_bytes) ||
        h.off_meta < sizeof(TfpnHeader) ||
        !check_range(h.off_meta, meta_bytes, h.off_scales) ||
        !check_range(h.off_scales, sc_bytes, h.off_weights) ||
        !check_range(h.off_weights, w_bytes, sz)) {
        fclose(f); fprintf(stderr, "FAIL %s: section overlap or truncation\n", path.c_str()); return false;
    }
    std::vector<TfpnTensorMeta> metas(h.num_tensors);
    if (fseeko(f, static_cast<off_t>(h.off_meta), SEEK_SET) != 0 ||
        (h.num_tensors > 0 && fread(metas.data(), sizeof(TfpnTensorMeta), h.num_tensors, f) != h.num_tensors)) {
        fclose(f); fprintf(stderr, "FAIL %s: short read\n", path.c_str()); return false;
    }
    if ((h.pad0 & tfq_xxh64::FLAG_XXH64_PRESENT) != 0u) {
        tfq_xxh64::State st(0ULL);
        st.update(metas.data(), static_cast<size_t>(meta_bytes));
        if (!tfq_xxh64::update_from_file(f, h.off_scales, sc_bytes, st) ||
            !tfq_xxh64::update_from_file(f, h.off_weights, w_bytes, st)) {
            fclose(f); fprintf(stderr, "FAIL %s: short read during XXH64 verification\n", path.c_str()); return false;
        }
        const uint64_t got = st.digest();
        if (got != h.pad1) {
            fclose(f);
            fprintf(stderr, "FAIL %s: TFPN XXH64 checksum mismatch (got 0x%016llx, expected 0x%016llx)\n",
                    path.c_str(), (unsigned long long)got, (unsigned long long)h.pad1);
            return false;
        }
    }
    fclose(f);

    std::unordered_set<std::string> local_seen;
    uint64_t params_total = 0;
    uint64_t cnt[6] = {};
    double snr_sum[6] = {};
    for (const auto& m : metas) {
        std::string tname;
        if (!extract_tensor_name(m.name, tname)) {
            fprintf(stderr, "FAIL %s: empty or unterminated tensor name\n", path.c_str());
            return false;
        }
        if (!local_seen.insert(tname).second || (seen_names && !seen_names->insert(tname).second)) {
            fprintf(stderr, "FAIL %s: duplicate tensor name '%s'\n", path.c_str(), tname.c_str());
            return false;
        }
        if ((m.b_bits != 3u && m.b_bits != 4u && m.b_bits != 5u) || m.R == 0 || m.C == 0 || (m.R % 128u) || (m.C % 128u) ||
            !check_orig_dim(m.pad[0], m.R) || !check_orig_dim(m.pad[1], m.C)) {
            fprintf(stderr, "FAIL %s: invalid tensor %s [%u,%u] %ub\n", path.c_str(), tname.c_str(), m.R, m.C, m.b_bits);
            return false;
        }
        if (!std::isfinite(m.snr_db)) {
            fprintf(stderr, "FAIL %s: non-finite SNR for tensor %s\n", path.c_str(), tname.c_str());
            return false;
        }
        uint64_t expect_words = (uint64_t)(m.R / 16u) * (m.C / 128u) * (m.b_bits * 2u) * 32ULL;
        uint64_t expect_sc = (uint64_t)(m.R / 128u) * (m.C / 128u);
        if (m.word_count != expect_words ||
            !check_range(m.word_off, m.word_count, h.total_weight_words) ||
            !check_range((uint64_t)m.sc_off, expect_sc, h.total_scale_halfs)) {
            fprintf(stderr, "FAIL %s: out-of-bounds tensor %s\n", path.c_str(), tname.c_str());
            return false;
        }
        params_total += (uint64_t)m.R * m.C;
        cnt[m.b_bits]++;
        snr_sum[m.b_bits] += m.snr_db;
        if (verbose) {
            uint32_t orig_r = m.pad[0] ? m.pad[0] : m.R;
            uint32_t orig_c = m.pad[1] ? m.pad[1] : m.C;
            if (orig_r != m.R || orig_c != m.C) {
                printf("  %-52s [%6u, %5u] (pad %ux%u) %ub  SNR=%5.2f dB\n",
                       tname.c_str(), orig_r, orig_c, m.R, m.C, m.b_bits, m.snr_db);
            } else {
                printf("  %-52s [%6u, %5u] %ub  SNR=%5.2f dB\n", tname.c_str(), m.R, m.C, m.b_bits, m.snr_db);
            }
        }
    }
    double bpw = params_total ? (double)(h.total_weight_words * 32ULL + h.total_scale_halfs * 16ULL) / (double)params_total : 0.0;
    printf("OK   %-42s TFPN v%u | %3u tensors | %6.3f GiB | %.3f bpw | 3b:%llu(%.2fdB) 4b:%llu(%.2fdB) 5b:%llu(%.2fdB)\n",
           path.c_str(), h.version, h.num_tensors, (double)sz / (double)(1ULL << 30), bpw,
           (unsigned long long)cnt[3], cnt[3] ? snr_sum[3] / cnt[3] : 0.0,
           (unsigned long long)cnt[4], cnt[4] ? snr_sum[4] / cnt[4] : 0.0,
           (unsigned long long)cnt[5], cnt[5] ? snr_sum[5] / cnt[5] : 0.0);
    return true;
}

static bool inspect_tfpw(const std::string& path) {
    uint64_t sz = file_size(path);
    if (sz < sizeof(TfpwHeader)) { fprintf(stderr, "FAIL %s: too small\n", path.c_str()); return false; }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    TfpwHeader h{};
    if (fread(&h, sizeof(h), 1, f) != 1) {
        fclose(f); fprintf(stderr, "FAIL %s: short read\n", path.c_str()); return false;
    }
    if (h.magic != MAGIC_TFPW || h.version != FORMAT_VER) {
        fclose(f); fprintf(stderr, "FAIL %s: bad magic/version\n", path.c_str()); return false;
    }
    if ((h.off_scales % ALIGN_16K) != 0 || (h.off_weights % ALIGN_16K) != 0 || (sz % ALIGN_16K) != 0) {
        fclose(f); fprintf(stderr, "FAIL %s: scales/weights/file end not 16 KiB aligned\n", path.c_str()); return false;
    }
    if (h.num_tensors != 3ULL * h.num_experts) {
        fclose(f); fprintf(stderr, "FAIL %s: num_tensors %u != 3 * num_experts %u\n", path.c_str(), h.num_tensors, h.num_experts); return false;
    }
    if (h.hidden_size == 0 || h.moe_inter == 0 || (h.hidden_size % 128u) || (h.moe_inter % 128u)) {
        fclose(f); fprintf(stderr, "FAIL %s: invalid MoE dims H=%u I=%u\n", path.c_str(), h.hidden_size, h.moe_inter); return false;
    }
    uint64_t meta_bytes = (uint64_t)h.num_experts * sizeof(TfpwExpertMeta);
    uint64_t sc_bytes = 0, w_bytes = 0;
    if (!checked_bytes(h.scale_halfs, 2ULL, sc_bytes) || !checked_bytes(h.stream_words, 4ULL, w_bytes) ||
        h.off_meta < sizeof(TfpwHeader) ||
        !check_range(h.off_meta, meta_bytes, h.off_scales) ||
        !check_range(h.off_scales, sc_bytes, h.off_weights) ||
        !check_range(h.off_weights, w_bytes, sz)) {
        fclose(f); fprintf(stderr, "FAIL %s: section overlap or truncation\n", path.c_str()); return false;
    }
    std::vector<TfpwExpertMeta> metas(h.num_experts);
    if (fseeko(f, static_cast<off_t>(h.off_meta), SEEK_SET) != 0 ||
        (h.num_experts > 0 && fread(metas.data(), sizeof(TfpwExpertMeta), h.num_experts, f) != h.num_experts)) {
        fclose(f); fprintf(stderr, "FAIL %s: short read\n", path.c_str()); return false;
    }
    if ((h.reserved0 & tfq_xxh64::FLAG_XXH64_PRESENT) != 0u) {
        const uint64_t got_meta = tfq_xxh64::hash(metas.data(), static_cast<size_t>(meta_bytes));
        if (got_meta != h.reserved1[0]) {
            fclose(f);
            fprintf(stderr, "FAIL %s: TFPW XXH64 meta checksum mismatch (got 0x%016llx, expected 0x%016llx)\n",
                    path.c_str(), (unsigned long long)got_meta, (unsigned long long)h.reserved1[0]);
            return false;
        }
        uint64_t got_sc = 0ULL, got_w = 0ULL;
        if (!tfq_xxh64::hash_file_slice(f, h.off_scales, sc_bytes, got_sc) ||
            !tfq_xxh64::hash_file_slice(f, h.off_weights, w_bytes, got_w)) {
            fclose(f); fprintf(stderr, "FAIL %s: short read during XXH64 verification\n", path.c_str()); return false;
        }
        if (got_sc != h.reserved1[1]) {
            fclose(f);
            fprintf(stderr, "FAIL %s: TFPW XXH64 scales checksum mismatch (got 0x%016llx, expected 0x%016llx)\n",
                    path.c_str(), (unsigned long long)got_sc, (unsigned long long)h.reserved1[1]);
            return false;
        }
        if (got_w != h.reserved1[2]) {
            fclose(f);
            fprintf(stderr, "FAIL %s: TFPW XXH64 weights checksum mismatch (got 0x%016llx, expected 0x%016llx)\n",
                    path.c_str(), (unsigned long long)got_w, (unsigned long long)h.reserved1[2]);
            return false;
        }
    }
    fclose(f);

    // Bucket index: 0=2b, 1=2.25b, 2=3b, 3=4b, 4=5b
    uint64_t cnt[5] = {};
    double snr_sum[5] = {};
    const uint64_t exp_sc = (uint64_t)(h.moe_inter / 128u) * (h.hidden_size / 128u);
    for (const auto& m : metas) {
        int bkt = -1;
        uint64_t gu_words = 0, dn_words = 0;
        if (m.b_bits == 2u) {
            bkt = 0;
            gu_words = (uint64_t)h.moe_inter * (h.hidden_size / 16u);
            dn_words = (uint64_t)h.hidden_size * (h.moe_inter / 16u);
        } else if (m.b_bits == 225u) {
            bkt = 1;
            gu_words = (uint64_t)h.moe_inter * (h.hidden_size / 128u) * 9ULL;
            dn_words = (uint64_t)h.hidden_size * (h.moe_inter / 128u) * 9ULL;
        } else if (m.b_bits == 3u || m.b_bits == 4u || m.b_bits == 5u) {
            bkt = (int)(m.b_bits - 1u);
            gu_words = (uint64_t)(h.moe_inter / 16u) * (h.hidden_size / 128u) * (m.b_bits * 2u) * 32ULL;
            dn_words = (uint64_t)(h.hidden_size / 16u) * (h.moe_inter / 128u) * (m.b_bits * 2u) * 32ULL;
        } else {
            fprintf(stderr, "FAIL %s: invalid expert bits (%u)\n", path.c_str(), m.b_bits);
            return false;
        }
        if (!std::isfinite(m.gate_snr_db) || !std::isfinite(m.up_snr_db) || !std::isfinite(m.down_snr_db)) {
            fprintf(stderr, "FAIL %s: non-finite expert SNR\n", path.c_str());
            return false;
        }
        if (!check_range(m.gate_word_off, gu_words, h.stream_words) ||
            !check_range(m.up_word_off, gu_words, h.stream_words) ||
            !check_range(m.down_word_off, dn_words, h.stream_words) ||
            !check_range((uint64_t)m.gate_sc_off, exp_sc, h.scale_halfs) ||
            !check_range((uint64_t)m.up_sc_off, exp_sc, h.scale_halfs) ||
            !check_range((uint64_t)m.down_sc_off, exp_sc, h.scale_halfs)) {
            fprintf(stderr, "FAIL %s: expert weight or scale out of bounds\n", path.c_str());
            return false;
        }
        cnt[bkt]++;
        snr_sum[bkt] += (m.gate_snr_db + m.up_snr_db + m.down_snr_db) / 3.0;
    }
    uint64_t params = (uint64_t)h.num_experts * 3ULL * h.hidden_size * h.moe_inter;
    double bpw = params ? (double)(h.stream_words * 32ULL + h.scale_halfs * 16ULL) / (double)params : 0.0;
    printf("OK   %-42s TFPW v%u | L%02u E=%u H=%u I=%u | %6.3f GiB | %.3f bpw |",
           path.c_str(), h.version, h.layer_idx, h.num_experts, h.hidden_size, h.moe_inter,
           (double)sz / (double)(1ULL << 30), bpw);
    if (cnt[0]) printf(" 2b:%llu(%.2fdB)", (unsigned long long)cnt[0], snr_sum[0] / cnt[0]);
    if (cnt[1]) printf(" 2.25b:%llu(%.2fdB)", (unsigned long long)cnt[1], snr_sum[1] / cnt[1]);
    printf(" 3b:%llu(%.2fdB) 4b:%llu(%.2fdB) 5b:%llu(%.2fdB)\n",
           (unsigned long long)cnt[2], cnt[2] ? snr_sum[2] / cnt[2] : 0.0,
           (unsigned long long)cnt[3], cnt[3] ? snr_sum[3] / cnt[3] : 0.0,
           (unsigned long long)cnt[4], cnt[4] ? snr_sum[4] / cnt[4] : 0.0);
    return true;
}

static float f16_to_f32_local(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0u) {
        if (mant == 0u) out = sign;
        else {
            int e = -14;
            while ((mant & 0x400u) == 0u) { mant <<= 1; --e; }
            mant &= 0x3FFu;
            out = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

static bool inspect_pwv4(const std::string& path) {
    uint64_t sz = file_size(path);
    if (sz != 16384ULL && sz != 32768ULL) {
        fprintf(stderr, "FAIL %s: expected 16384 or 32768 bytes (%llu)\n", path.c_str(), (unsigned long long)sz);
        return false;
    }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    std::vector<uint8_t> buf((size_t)sz);
    bool ok = (fread(buf.data(), 1, buf.size(), f) == buf.size());
    fclose(f);
    if (!ok) { fprintf(stderr, "FAIL %s: short read\n", path.c_str()); return false; }

    uint32_t hdr[5];
    std::memcpy(hdr, buf.data(), 20);
    if (hdr[0] != MAGIC_PWV4 || hdr[1] != TFQ4_VER || hdr[3] != 8u) {
        fprintf(stderr, "FAIL %s: bad PWV4 header\n", path.c_str());
        return false;
    }
    const uint32_t k_codes = hdr[2];
    const uint32_t table_kind = hdr[4];
    if (table_kind > 3u) {
        fprintf(stderr, "FAIL %s: unknown PWV4 table_kind %u (supported: 0/1=trained, 2=true-E8P, 3=true-E8P-1024)\n",
                path.c_str(), table_kind);
        return false;
    }
    if (table_kind == 3u) {
        if (sz != 32768ULL || k_codes != 1024u) {
            fprintf(stderr, "FAIL %s: PWV4 table_kind 3 requires 32768 bytes and k_codes=1024 (got %llu bytes, k_codes=%u)\n",
                    path.c_str(), (unsigned long long)sz, k_codes);
            return false;
        }
    } else {
        if (sz != 16384ULL || k_codes != 256u) {
            fprintf(stderr, "FAIL %s: PWV4 table_kind %u requires 16384 bytes and k_codes=256 (got %llu bytes, k_codes=%u)\n",
                    path.c_str(), table_kind, (unsigned long long)sz, k_codes);
            return false;
        }
    }
    for (size_t i = 0; i < (size_t)k_codes * 8u; ++i) {
        uint16_t h;
        std::memcpy(&h, buf.data() + 64 + i * 2, 2);
        if (!f16_is_nonneg_finite(h)) {
            fprintf(stderr, "FAIL %s: invalid PWV4 e8p_lut entry [%zu][%zu] (0x%04x)\n", path.c_str(), i / 8, i % 8, h);
            return false;
        }
        if (table_kind == 2u || table_kind == 3u) {
            float v = f16_to_f32_local(h);
            float two_v = v * 2.0f;
            float r = std::round(two_v);
            if (v < 0.5f || std::fabs(two_v - r) > 1e-3f || (((int)r) & 1) == 0) {
                fprintf(stderr, "FAIL %s: invalid true-E8P half-integer LUT entry [%zu][%zu] (%g)\n",
                        path.c_str(), i / 8, i % 8, v);
                return false;
            }
        }
    }
    const char* kind_lbl = (table_kind == 3u)
        ? "true-E8P-1024 (E8+1/4)"
        : ((table_kind == 2u) ? "true-E8P (E8+1/4)" : "trained-256x8");
    const uint32_t lut_kib = (k_codes * 8u * 2u) / 1024u;
    printf("OK   %-42s PWV4 v%u | %llu bytes | kind=%u (%s) %ux%uD (%u KiB LUT)\n",
           path.c_str(), hdr[1], (unsigned long long)sz, table_kind, kind_lbl, k_codes, hdr[3], lut_kib);
    return true;
}

static bool inspect_tfp4(const std::string& path, bool verbose, std::unordered_set<std::string>* seen_names) {
    uint64_t sz = file_size(path);
    if (sz < sizeof(TfpnHeader)) { fprintf(stderr, "FAIL %s: too small\n", path.c_str()); return false; }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    TfpnHeader h{};
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); fprintf(stderr, "FAIL %s: short read\n", path.c_str()); return false; }
    if (h.magic != MAGIC_TFP4 || h.version != TFQ4_VER ||
        (h.off_scales % ALIGN_16K) != 0 || (h.off_weights % ALIGN_16K) != 0 || (sz % ALIGN_16K) != 0) {
        fclose(f);
        fprintf(stderr, "FAIL %s: bad TFP4 header or 16 KiB alignment\n", path.c_str());
        return false;
    }
    uint64_t meta_bytes = (uint64_t)h.num_tensors * sizeof(TfpnTensorMeta);
    uint64_t sc_bytes = 0, w_bytes = 0;
    if (!checked_bytes(h.total_scale_halfs, 2ULL, sc_bytes) || !checked_bytes(h.total_weight_words, 4ULL, w_bytes) ||
        h.off_meta < sizeof(TfpnHeader) ||
        !check_range(h.off_meta, meta_bytes, h.off_scales) ||
        !check_range(h.off_scales, sc_bytes, h.off_weights) ||
        !check_range(h.off_weights, w_bytes, sz)) {
        fclose(f);
        fprintf(stderr, "FAIL %s: TFP4 section overlap or truncation\n", path.c_str());
        return false;
    }
    std::vector<TfpnTensorMeta> metas(h.num_tensors);
    if (fseeko(f, static_cast<off_t>(h.off_meta), SEEK_SET) != 0 ||
        (h.num_tensors > 0 && fread(metas.data(), sizeof(TfpnTensorMeta), h.num_tensors, f) != h.num_tensors)) {
        fclose(f);
        fprintf(stderr, "FAIL %s: short read\n", path.c_str());
        return false;
    }
    if ((h.pad0 & tfq_xxh64::FLAG_XXH64_PRESENT) != 0u) {
        tfq_xxh64::State st(0ULL);
        st.update(metas.data(), static_cast<size_t>(meta_bytes));
        if (!tfq_xxh64::update_from_file(f, h.off_scales, sc_bytes, st) ||
            !tfq_xxh64::update_from_file(f, h.off_weights, w_bytes, st)) {
            fclose(f);
            fprintf(stderr, "FAIL %s: short read during XXH64 verification\n", path.c_str());
            return false;
        }
        const uint64_t got = st.digest();
        if (got != h.pad1) {
            fclose(f);
            fprintf(stderr, "FAIL %s: TFP4 XXH64 checksum mismatch (got 0x%016llx, expected 0x%016llx)\n",
                    path.c_str(), (unsigned long long)got, (unsigned long long)h.pad1);
            return false;
        }
    }
    fclose(f);

    std::unordered_set<std::string> local_seen;
    uint64_t params_total = 0;
    uint64_t cnt_20 = 0, cnt_225 = 0;
    double snr_20 = 0.0, snr_225 = 0.0;
    for (const auto& m : metas) {
        std::string tname;
        if (!extract_tensor_name(m.name, tname)) {
            fprintf(stderr, "FAIL %s: empty or unterminated TFP4 tensor name\n", path.c_str());
            return false;
        }
        if (!local_seen.insert(tname).second || (seen_names && !seen_names->insert(tname).second)) {
            fprintf(stderr, "FAIL %s: duplicate TFP4 tensor name '%s'\n", path.c_str(), tname.c_str());
            return false;
        }
        if (!std::isfinite(m.snr_db)) {
            fprintf(stderr, "FAIL %s: non-finite SNR for TFP4 tensor %s\n", path.c_str(), tname.c_str());
            return false;
        }
        if ((m.b_bits != 2u && m.b_bits != 225u) || m.R == 0 || m.C == 0 || (m.R % 128u) || (m.C % 128u)) {
            fprintf(stderr, "FAIL %s: invalid TFP4 tensor %s [%u,%u] bits=%u\n", path.c_str(), tname.c_str(), m.R, m.C, m.b_bits);
            return false;
        }
        uint64_t exp_w = (m.b_bits == 225u)
            ? ((uint64_t)m.R * (m.C / 128u) * 9ULL)
            : ((uint64_t)m.R * (m.C / 16u));
        uint64_t exp_s = (uint64_t)(m.R / 128u) * (m.C / 128u);
        if (m.word_count != exp_w ||
            !check_orig_dim(m.pad[0], m.R) || !check_orig_dim(m.pad[1], m.C) ||
            !check_range(m.word_off, m.word_count, h.total_weight_words) ||
            !check_range((uint64_t)m.sc_off, exp_s, h.total_scale_halfs)) {
            fprintf(stderr, "FAIL %s: invalid TFP4 tensor %s\n", path.c_str(), tname.c_str());
            return false;
        }
        params_total += (uint64_t)m.R * m.C;
        if (m.b_bits == 225u) { cnt_225++; snr_225 += m.snr_db; }
        else                  { cnt_20++;  snr_20  += m.snr_db; }
        if (verbose) {
            const char* tag = (m.b_bits == 225u) ? "2.25b-E8P1024" : "2b-E8P";
            uint32_t orig_r = m.pad[0] ? m.pad[0] : m.R, orig_c = m.pad[1] ? m.pad[1] : m.C;
            if (orig_r != m.R || orig_c != m.C)
                printf("  %-52s [%6u, %5u] (pad %ux%u) %s SNR=%5.2f dB\n", tname.c_str(), orig_r, orig_c, m.R, m.C, tag, m.snr_db);
            else
                printf("  %-52s [%6u, %5u] %s SNR=%5.2f dB\n", tname.c_str(), m.R, m.C, tag, m.snr_db);
        }
    }
    double bpw = params_total ? (double)(h.total_weight_words * 32ULL + h.total_scale_halfs * 16ULL) / (double)params_total : 0.0;
    printf("OK   %-42s TFP4 v%u | %3u tensors | %6.3f GiB | %.3f bpw |",
           path.c_str(), h.version, h.num_tensors, (double)sz / (double)(1ULL << 30), bpw);
    if (cnt_20 || !cnt_225) printf(" E8P-2b:%llu(%.2fdB)", (unsigned long long)cnt_20, cnt_20 ? snr_20 / cnt_20 : 0.0);
    if (cnt_225) printf(" E8P-2.25b:%llu(%.2fdB)", (unsigned long long)cnt_225, snr_225 / cnt_225);
    printf("\n");
    return true;
}

static bool inspect_file(const std::string& path, bool verbose, std::unordered_set<std::string>* seen_names) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    uint32_t magic = 0;
    size_t n = fread(&magic, 4, 1, f);
    fclose(f);
    if (n != 1) { fprintf(stderr, "FAIL %s: empty file\n", path.c_str()); return false; }
    if (magic == MAGIC_PWV3) return inspect_tables(path);
    if (magic == MAGIC_PWV4) return inspect_pwv4(path);
    if (magic == MAGIC_TFPN) return inspect_tfpn(path, verbose, seen_names);
    if (magic == MAGIC_TFPW) return inspect_tfpw(path);
    if (magic == MAGIC_TFP4) return inspect_tfp4(path, verbose, seen_names);
    fprintf(stderr, "FAIL %s: unknown magic 0x%08x\n", path.c_str(), magic);
    return false;
}

int main(int argc, char** argv) {
    bool verbose = false;
    std::vector<std::string> targets;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose") verbose = true;
        else if (a == "-h" || a == "--help") { targets.clear(); break; }
        else targets.push_back(a);
    }
    if (targets.empty()) {
        fprintf(stderr, "Usage: %s [-v|--verbose] <shard.tffpw | shard.tffp4 | pw_v3_tables.bin | pw_v4_e8p_tables.bin | checkpoint_dir> [...]\n", argv[0]);
        return 1;
    }
    bool all_ok = true;
    for (const auto& t : targets) {
        if (is_dir(t)) {
            DIR* d = opendir(t.c_str());
            if (!d) { fprintf(stderr, "FAIL %s: cannot open dir\n", t.c_str()); return 1; }
            std::vector<std::string> files;
            while (auto* ent = readdir(d)) {
                std::string nm = ent->d_name;
                if (nm == "pw_v3_tables.bin" || nm == "pw_v4_e8p_tables.bin" ||
                    (nm.size() > 6 && (nm.substr(nm.size() - 6) == ".tffpw" || nm.substr(nm.size() - 6) == ".tffp4"))) {
                    files.push_back(t + "/" + nm);
                }
            }
            closedir(d);
            std::sort(files.begin(), files.end());
            if (files.empty()) {
                fprintf(stderr, "FAIL %s: no TFQ3/TFQ4 table or shard files found\n", t.c_str());
                return 1;
            }
            std::unordered_set<std::string> dir_seen;
            for (const auto& fp : files) all_ok &= inspect_file(fp, verbose, &dir_seen);
        } else {
            all_ok &= inspect_file(t, verbose, nullptr);
        }
    }
    return all_ok ? 0 : 1;
}
