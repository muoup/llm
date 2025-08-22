#pragma once

#include <vector>
#include <cmath>

#include "../util/matrix.h"
#include "../tokenizer/token.h"

struct embedding {
    matrix data;

    explicit embedding(size_t dimensions)
        : data({ 1, dimensions }) {}
};

struct ff_layer {
    matrix w1, b1;
    matrix w2, b2;

    ff_layer(size_t dimensions, size_t projection_size)
        : w1({ dimensions, projection_size }), b1({ 1, projection_size }),
          w2({ projection_size, dimensions }), b2({ 1, dimensions }) {}
};

struct logit_layer {
    size_t vocab_size;
    matrix w, b;

    explicit logit_layer(const size_t dimensions, const size_t vocab_size)
        : vocab_size(vocab_size),
          w(dimensions, vocab_size),
          b(1, vocab_size) {}
};

static constexpr auto logistic(const float f) {
    return 1.0f / (1.0f + std::exp(-f));
}

struct llm {
    size_t m_dimensions;

    std::vector<embedding> m_embeddings;
    std::vector<ff_layer> m_ff_layer;
    logit_layer m_logit_layer;

    llm(const size_t vocab_size, const size_t layer_count,
        const size_t dimensions, const size_t projection_scale = 4)
        : m_dimensions(dimensions),
          m_embeddings(vocab_size, embedding { dimensions }),
          m_ff_layer(layer_count, ff_layer { dimensions, dimensions * projection_scale }),
          m_logit_layer(dimensions, vocab_size) {}

    void randomize();

    matrix embed_tokens(std::span<const token_id_t> tokens) const;
    static void positional_encoding(matrix& input);

    matrix forward_l1(const matrix& input, size_t layer) const;
    static matrix activate(const matrix& input);
    matrix forward_l2(const matrix& input, size_t layer) const;
    matrix generate_logits(const matrix &input) const;

    matrix feed_forward(const matrix& input, size_t layer) const;

    matrix prediction_matrix(std::span<const token_id_t> tokens) const;
    token_id_t predict(std::span<const token_id_t> tokens) const;

    std::string to_string() const;

    size_t vocab_size() const {
        return m_logit_layer.vocab_size;
    }

    // Input: [N x 1] tokens
    // Embedding: [N x Dimensions] embeddings

    // Layer W1: [Dimensions x DFF]
    // Layer W2: [DFF x Dimensions]

    // Embedding @ Layer W1 -> [N x DFF] Output
    // Output @ Layer W2 -> [N x Dimensions] Output
};