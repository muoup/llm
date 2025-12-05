#include "standard_model.hpp"

InferenceModel create_standard_model(size_t dimensions, size_t vocab_size, size_t ff_layers, size_t attention_heads) {
    constexpr size_t head_count = 8;
    constexpr size_t ffn_multiplier = 4;
    
    InferenceModel model(dimensions, vocab_size);

    size_t acc = 0;
    
    for (size_t i = 0; i < ff_layers; ++i) {
        size_t attn_idx = model.add_layer(
            std::make_unique<AttentionLayer>(dimensions, attention_heads));
        size_t ff_idx = model.add_layer(
            std::make_unique<FeedForwardLayer>(dimensions, dimensions * ffn_multiplier));

        if (acc != 0) {
            model.add_connection(acc, attn_idx);
        }
        
        model.add_connection(attn_idx, ff_idx);
        acc = ff_idx;
    }
 
    model.randomize();
    model.finalize();
    return model;
}