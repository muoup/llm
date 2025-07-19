#pragma once

#include <vector>
#include <cmath>
#include <sstream>

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

static auto logistic(const float f) {
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

    matrix prediction_matrix(const std::span<const token_id_t> tokens) const {
        const matrix input = embed_tokens(tokens);

        matrix forwarded { 0, 0 };

        for (size_t i = 0; i < m_ff_layer.size(); ++i) {
            forwarded = feed_forward(input, i);
        }

        const matrix logits = generate_logits(forwarded);
        matrix predictions { 1, m_logit_layer.vocab_size };

        const size_t last_logit_idx = logits.rows - 1;

        for (size_t i = 0; i < logits.cols; ++i) {
            predictions.set(0, i, logits.get(last_logit_idx, i));
        }

        return predictions;
    }

    token_id_t predict(const std::span<const token_id_t> tokens) const {
        const auto predictions = prediction_matrix(tokens);

        auto max_idx = 0;
        const size_t last_row = predictions.rows - 1;

        for (size_t i = 1; i < predictions.cols; i++) {
            if (predictions.get(last_row, i) > predictions.get(last_row, max_idx)) {
                max_idx = i;
            }
        }

        return max_idx;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "LLM with " << m_embeddings.size() << " embeddings and " << m_ff_layer.size() << " layers.\n";
        for (size_t i = 0; i < m_ff_layer.size(); ++i) {
            ss << "Layer " << i + 1 << ": W1 (" << m_ff_layer[i].w1.rows << " x " << m_ff_layer[i].w1.cols
               << "), W2 (" << m_ff_layer[i].w2.rows << " x " << m_ff_layer[i].w2.cols << ")\n";
            ss << "\n" << m_ff_layer[i].w1.to_string() << "\n";
            ss << "\n" << m_ff_layer[i].w2.to_string() << "\n";
        }
        return ss.str();
    }

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