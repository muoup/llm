#include "standard_model.hpp"

#include <inference/attention.hpp>
#include <inference/feed_forward.hpp>
#include <inference/layer_normalize.hpp>
#include <inference/recursion_node.hpp>
#include "inference/inference.hpp"
#include "inference/rms_normalize.hpp"

InferenceModel minimal_model(size_t vocab_size) {
    const size_t dimensions = 128;

    InferenceModel model(dimensions, vocab_size);

    model.add_connection(0, 1);
    model.add_connection(1, 2);

    auto attn_layer = std::make_unique<AttentionLayer>(dimensions, 1, true);
    model.add_layer(
        std::make_unique<LayerNorm>(std::move(attn_layer), dimensions));

    auto ff_layer
        = std::make_unique<FeedForwardLayer>(dimensions, dimensions * 4);
    model.add_layer(
        std::make_unique<LayerNorm>(std::move(ff_layer), dimensions));

    auto empty_norm = std::make_unique<LayerNorm>(nullptr, dimensions);
    model.add_layer(std::move(empty_norm));

    model.finalize();
    model.randomize();
    return model;
}

InferenceModel standard_attention_model(size_t dimensions,
                                        size_t vocab_size,
                                        size_t num_blocks,
                                        size_t attention_heads,
                                        ActivationFunction activation) {
    size_t ffn_multiplier;
    if (activation == ActivationFunction::SwiGLU) {
        ffn_multiplier = (dimensions * 4 / 3) / dimensions;
        if (ffn_multiplier < 1)
            ffn_multiplier = 1;
    } else {
        ffn_multiplier = 4;
    }

    InferenceModel model(dimensions, vocab_size);

    size_t last_layer_idx = 0;

    for (size_t i = 0; i < num_blocks; ++i) {
        auto attn_layer = std::make_unique<AttentionLayer>(
            dimensions, attention_heads, true);
        auto attn_block
            = std::make_unique<RMSNorm>(std::move(attn_layer), dimensions);
        size_t attn_block_idx = model.add_layer(std::move(attn_block));

        if (last_layer_idx != 0) {
            model.add_connection(last_layer_idx, attn_block_idx);
        }

        auto ff_layer = std::make_unique<FeedForwardLayer>(
            dimensions, dimensions * ffn_multiplier, activation);
        auto ff_block
            = std::make_unique<RMSNorm>(std::move(ff_layer), dimensions);
        size_t ff_block_idx = model.add_layer(std::move(ff_block));

        model.add_connection(attn_block_idx, ff_block_idx);

        last_layer_idx = ff_block_idx;
    }

    size_t standardized_norm
        = model.add_layer(std::make_unique<LayerNorm>(nullptr, dimensions));
    model.add_connection(last_layer_idx, standardized_norm);

    model.randomize();
    model.finalize();
    return model;
}
