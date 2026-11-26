#pragma once
// Shared FlashNext helper logic (Apple Silicon Metal + NVIDIA CUDA parity).
#include <cstdio>
#include <string>
#include <vector>

namespace tfq_flashnext {

static inline std::string canonical_ple_stem(int shard_idx) {
    char stem[192];
    std::snprintf(stem, sizeof(stem),
                  "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d",
                  shard_idx);
    return std::string(stem);
}

// mlx-vlm >= 0.7.12 names PLE n-gram tables `ngram_embedding.shards.N`;
// older builds use `ngram_embedding.shard_N`. Output names stay `shard_N`.
template <typename CacheT>
static inline std::string resolve_ple_src_stem(const CacheT& cache, int shard_idx) {
    std::string primary = canonical_ple_stem(shard_idx);
    if (cache.has_tensor(primary + ".weight")) return primary;

    char alt[192];
    std::snprintf(alt, sizeof(alt),
                  "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.%d",
                  shard_idx);
    if (cache.has_tensor(std::string(alt) + ".weight")) return std::string(alt);

    char legacy[192];
    std::snprintf(legacy, sizeof(legacy),
                  "language_model.model.embed_tokens_per_layer.shards.%d",
                  shard_idx);
    if (cache.has_tensor(std::string(legacy) + ".weight")) return std::string(legacy);

    return primary;
}

// --mtp adds the drafter (layer 48) to the requested layers; --mtp without --layers
// converts only the drafter.
static inline std::vector<int> build_layer_list(int l_start, int l_end, bool layers_given, bool do_mtp) {
    std::vector<int> layer_list;
    if (!do_mtp || layers_given) {
        for (int l = l_start; l <= l_end; ++l) layer_list.push_back(l);
    }
    if (do_mtp) layer_list.push_back(48);
    return layer_list;
}

} // namespace tfq_flashnext
