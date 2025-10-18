#pragma once

#include <string>
#include <vector>
#include <optional>

#include <tokenizer/token.h>
#include <util/matrix.h>

#include <network/embedding.h>
#include <network/feed_forward.h>
#include <network/attention.h>
#include <network/logit_layer.h>

struct llm;

// ---[ Serialization ]---
void save_llm(const llm &model, const std::string &path);
std::optional<llm> load_llm(const std::string &path);

struct llm {
    void randomize();

    matrix prediction_matrix(std::span<const token_id_t> tokens) const;
    token_id_t predict(std::span<const token_id_t> tokens) const;

    std::string to_string() const;

    llm(const size_t vocab_size, const size_t layer_count,
        const size_t dimensions, const size_t projection_scale = 4, const size_t head_count = 4)
        : m_dimensions(dimensions), m_layer_count(layer_count),
          m_embedding_layer(vocab_size, dimensions),
          m_attention_layers(layer_count, attention_layer{ dimensions, dimensions / head_count }),
          m_ff_layer(layer_count,
                     ff_layer{ dimensions, dimensions * projection_scale }),
          m_logit_layer(dimensions, vocab_size) {}

    size_t vocab_size() const { return m_logit_layer.vocab_size; }

    size_t m_dimensions;
    size_t m_layer_count;

    embedding_layer m_embedding_layer;
    std::vector<attention_layer> m_attention_layers;
    std::vector<ff_layer> m_ff_layer;
    logit_layer m_logit_layer;
};
