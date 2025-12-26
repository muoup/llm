#include "standard_model.hpp"

#include <inference/attention.hpp>
#include <inference/feed_forward.hpp>
#include <inference/layer_normalize.hpp>
#include <inference/linearized_attention.hpp>
#include <inference/recursion_node.hpp>
#include "inference/inference.hpp"
#include "kernels/optimizer.hpp"

InferenceModel minimal_model(size_t vocab_size) {
    const size_t dimensions = 128;

    InferenceModel model(dimensions, vocab_size);

    // model.add_connection(0, 1);
    // model.add_connection(1, 2);
    
    model.add_layer(std::make_unique<AttentionLayer>(dimensions, 1, true));
 
    // auto attn_layer = std::make_unique<AttentionLayer>(dimensions, 1, true);
    // model.add_layer(std::make_unique<LayerNorm>(std::move(attn_layer), dimensions));
 
    // model.add_layer(std::make_unique<FeedForwardLayer>(dimensions, dimensions * 4));
    // auto ff_layer = std::make_unique<FeedForwardLayer>(dimensions, dimensions * 4);
    // model.add_layer(std::make_unique<LayerNorm>(std::move(ff_layer), dimensions));

    // auto empty_norm = std::make_unique<LayerNorm>(nullptr, dimensions);
    // model.add_layer(std::move(empty_norm));
    
    model.finalize();
    model.randomize();
    return model;
}

InferenceModel standard_attention_model(size_t dimensions,
                                        size_t vocab_size,
                                        size_t num_blocks,
                                        size_t attention_heads) {
    constexpr size_t ffn_multiplier = 4;

    InferenceModel model(dimensions, vocab_size);
    CHECK_ERRORS("InferenceModel Constructor");

    size_t last_layer_idx = 0;

    for (size_t i = 0; i < num_blocks; ++i) {
        auto attn_layer
            = std::make_unique<AttentionLayer>(dimensions, attention_heads, true);
        CHECK_ERRORS("AttentionLayer Creation");
        auto attn_block
            = std::make_unique<LayerNorm>(std::move(attn_layer), dimensions);
        CHECK_ERRORS("LayerNorm Creation");
        size_t attn_block_idx = model.add_layer(std::move(attn_block));

        if (attn_block_idx != 0) {
            model.add_connection(last_layer_idx, attn_block_idx);
        }

        auto ff_layer = std::make_unique<FeedForwardLayer>(
            dimensions, dimensions * ffn_multiplier);
        CHECK_ERRORS("FeedForwardLayer Creation");
        auto ff_block
            = std::make_unique<LayerNorm>(std::move(ff_layer), dimensions);
        CHECK_ERRORS("LayerNorm Creation");
        size_t ff_block_idx = model.add_layer(std::move(ff_block));

        model.add_connection(attn_block_idx, ff_block_idx);

        last_layer_idx = ff_block_idx;
    }
    
    size_t standardized_norm = model.add_layer(std::make_unique<LayerNorm>(nullptr, dimensions));
    model.add_connection(last_layer_idx, standardized_norm);
    CHECK_ERRORS("Final LayerNorm Creation");
    
    model.randomize();
    model.finalize();
    kernel::optimizer::wait_for_operations();
    return model;
}

InferenceModel linearized_attention_model(size_t dimensions,
                                          size_t vocab_size,
                                          size_t num_blocks,
                                          size_t attention_heads) {
    constexpr size_t ffn_multiplier = 4;

    InferenceModel model(dimensions, vocab_size);

    size_t last_layer_idx = 0;

    for (size_t i = 0; i < num_blocks; ++i) {
        auto attn_layer = std::make_unique<LinearizedAttention>(
            dimensions, attention_heads);
        auto attn_block
            = std::make_unique<LayerNorm>(std::move(attn_layer), dimensions);
        size_t attn_block_idx = model.add_layer(std::move(attn_block));

        if (last_layer_idx != 0) {
            model.add_connection(last_layer_idx, attn_block_idx);
        }

        auto ff_layer = std::make_unique<FeedForwardLayer>(
            dimensions, dimensions * ffn_multiplier);
        auto ff_block
            = std::make_unique<LayerNorm>(std::move(ff_layer), dimensions);
        size_t ff_block_idx = model.add_layer(std::move(ff_block));

        model.add_connection(attn_block_idx, ff_block_idx);

        last_layer_idx = ff_block_idx;
    }

    model.randomize();
    model.finalize();
    return model;
}

InferenceModel standard_recursive_model(size_t dimensions,
                                        size_t vocab_size,
                                        size_t num_blocks,
                                        size_t attention_heads,
                                        size_t max_recursions) {
    constexpr size_t ffn_multiplier = 4;

    InferenceModel model(dimensions, vocab_size);
    auto attn_layer
        = std::make_unique<AttentionLayer>(dimensions, attention_heads, true);
    size_t attn = model.add_layer(
        std::make_unique<LayerNorm>(std::move(attn_layer), dimensions));

    auto ff_layer = std::make_unique<FeedForwardLayer>(
        dimensions, dimensions * ffn_multiplier);
    size_t ff = model.add_layer(
        std::make_unique<LayerNorm>(std::move(ff_layer), dimensions));

    model.add_connection(attn, ff);

    std::vector<std::unique_ptr<INode>> loop;
    for (size_t i = 0; i < num_blocks; ++i) {
        auto attn_layer
            = std::make_unique<AttentionLayer>(dimensions, attention_heads, false);
        loop.emplace_back(
            std::make_unique<LayerNorm>(std::move(attn_layer), dimensions));

        auto ff_layer = std::make_unique<FeedForwardLayer>(
            dimensions, dimensions * ffn_multiplier);
        loop.emplace_back(
            std::make_unique<LayerNorm>(std::move(ff_layer), dimensions));
    }

    size_t recursion_node_idx = model.add_layer(std::make_unique<RecursionNode>(
        dimensions, max_recursions, std::move(loop)));
    model.add_connection(ff, recursion_node_idx);

    model.randomize();
    model.finalize();
    return model;
}
