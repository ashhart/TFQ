#pragma once
// ============================================================================
// Zero-dependency exact XXH64 (xxHash64, seed=0) for TFQ3 / TFQ4 integrity.
//
// Header flag & checksum layout (backwards-compatible with legacy zero fields):
//   - TFPN & TFP4:
//       pad0 bit 0 (FLAG_XXH64_PRESENT = 0x1u) = checksum present
//       pad1 = XXH64(seed=0) streamed over exact payload slices
//              [meta, scales, weights] (excluding inter-section zero padding)
//   - TFPW:
//       reserved0 bit 0 (FLAG_XXH64_PRESENT = 0x1u) = checksums present
//       reserved1[0] = XXH64(seed=0) over meta   [off_meta   .. +meta_bytes)
//       reserved1[1] = XXH64(seed=0) over scales [off_scales .. +sc_bytes)
//       reserved1[2] = XXH64(seed=0) over weights[off_weights.. +w_bytes)
//       reserved1[3] = 0
// ============================================================================
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace tfq_xxh64 {

static constexpr uint32_t FLAG_XXH64_PRESENT = 0x1u;

static constexpr uint64_t PRIME64_1 = 0x9E3779B185EBCA87ULL;
static constexpr uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
static constexpr uint64_t PRIME64_3 = 0x165667B19E3779F9ULL;
static constexpr uint64_t PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
static constexpr uint64_t PRIME64_5 = 0x27D4EB2F165667C5ULL;

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint32_t read_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

static inline uint64_t read_le64(const uint8_t* p) {
    return static_cast<uint64_t>(read_le32(p))
         | (static_cast<uint64_t>(read_le32(p + 4)) << 32);
}

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t merge_round64(uint64_t acc, uint64_t val) {
    val = round64(0ULL, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

static inline uint64_t avalanche64(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

struct State {
    uint64_t total_len = 0ULL;
    uint64_t v1 = 0ULL;
    uint64_t v2 = 0ULL;
    uint64_t v3 = 0ULL;
    uint64_t v4 = 0ULL;
    uint8_t  mem[32] = {};
    uint32_t memsize = 0u;

    explicit State(uint64_t seed = 0ULL) { reset(seed); }

    void reset(uint64_t seed = 0ULL) {
        total_len = 0ULL;
        memsize = 0u;
        v1 = seed + PRIME64_1 + PRIME64_2;
        v2 = seed + PRIME64_2;
        v3 = seed;
        v4 = seed - PRIME64_1;
    }

    void update(const void* data, size_t len) {
        if (len == 0u || data == nullptr) return;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        const uint8_t* const b_end = p + len;
        total_len += static_cast<uint64_t>(len);

        if (static_cast<size_t>(memsize) + len < 32u) {
            std::memcpy(mem + memsize, p, len);
            memsize += static_cast<uint32_t>(len);
            return;
        }
        if (memsize > 0u) {
            const size_t fill = 32u - static_cast<size_t>(memsize);
            std::memcpy(mem + memsize, p, fill);
            v1 = round64(v1, read_le64(mem + 0));
            v2 = round64(v2, read_le64(mem + 8));
            v3 = round64(v3, read_le64(mem + 16));
            v4 = round64(v4, read_le64(mem + 24));
            p += fill;
            memsize = 0u;
        }
        if (p + 32 <= b_end) {
            const uint8_t* const limit = b_end - 32;
            do {
                v1 = round64(v1, read_le64(p + 0));
                v2 = round64(v2, read_le64(p + 8));
                v3 = round64(v3, read_le64(p + 16));
                v4 = round64(v4, read_le64(p + 24));
                p += 32;
            } while (p <= limit);
        }
        if (p < b_end) {
            memsize = static_cast<uint32_t>(b_end - p);
            std::memcpy(mem, p, memsize);
        }
    }

    uint64_t digest() const {
        uint64_t h64;
        if (total_len >= 32ULL) {
            h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
            h64 = merge_round64(h64, v1);
            h64 = merge_round64(h64, v2);
            h64 = merge_round64(h64, v3);
            h64 = merge_round64(h64, v4);
        } else {
            h64 = v3 + PRIME64_5;
        }
        h64 += total_len;

        const uint8_t* p = mem;
        const uint8_t* const b_end = mem + memsize;
        while (p + 8 <= b_end) {
            const uint64_t k1 = round64(0ULL, read_le64(p));
            h64 ^= k1;
            h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
            p += 8;
        }
        if (p + 4 <= b_end) {
            h64 ^= static_cast<uint64_t>(read_le32(p)) * PRIME64_1;
            h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
            p += 4;
        }
        while (p < b_end) {
            h64 ^= static_cast<uint64_t>(*p) * PRIME64_5;
            h64 = rotl64(h64, 11) * PRIME64_1;
            ++p;
        }
        return avalanche64(h64);
    }
};

static inline uint64_t hash(const void* data, size_t len, uint64_t seed = 0ULL) {
    State st(seed);
    st.update(data, len);
    return st.digest();
}

// Combined XXH64 over [meta, scales, weights] for TFPN and TFP4 headers.
static inline uint64_t hash_sections3(
    const void* meta, size_t meta_bytes,
    const void* scales, size_t scales_bytes,
    const void* weights, size_t weights_bytes,
    uint64_t seed = 0ULL
) {
    State st(seed);
    st.update(meta, meta_bytes);
    st.update(scales, scales_bytes);
    st.update(weights, weights_bytes);
    return st.digest();
}

// Stream [off, off + len) from an already-open FILE* into `st`.
static inline bool update_from_file(FILE* f, uint64_t off, uint64_t len, State& st) {
    if (len == 0ULL) return true;
    if (fseeko(f, static_cast<off_t>(off), SEEK_SET) != 0) return false;
    std::vector<uint8_t> buf(1u << 20); // 1 MiB streaming buffer
    uint64_t rem = len;
    while (rem > 0ULL) {
        const size_t chunk = (rem > static_cast<uint64_t>(buf.size()))
                           ? buf.size()
                           : static_cast<size_t>(rem);
        if (fread(buf.data(), 1, chunk, f) != chunk) return false;
        st.update(buf.data(), chunk);
        rem -= static_cast<uint64_t>(chunk);
    }
    return true;
}

static inline bool hash_file_slice(FILE* f, uint64_t off, uint64_t len, uint64_t& out_hash, uint64_t seed = 0ULL) {
    State st(seed);
    if (!update_from_file(f, off, len, st)) return false;
    out_hash = st.digest();
    return true;
}

} // namespace tfq_xxh64
