//
// Created by user on 7/17/25.
//

#include "neural_net.h"

void llm::randomize() {
    constexpr auto min = -0.25f;
    constexpr auto max = 0.25f;

    for (auto &embedding : m_embeddings) {
        embedding.data.randomize(min, max);
    }

    for (auto &layer : m_ff_layer) {
        layer.w1.randomize(min, max);
        layer.b1.randomize(min, max);
        layer.w2.randomize(min, max);
        layer.b2.randomize(min, max);
    }

    m_logit_layer.w.randomize(min, max);
    m_logit_layer.b.randomize(min, max);
}

matrix llm::embed_tokens(const std::span<const token_id_t> tokens) const {
    matrix output { tokens.size(), m_dimensions };

    for (size_t token = 0; token < tokens.size(); ++token) {
        const auto &embedding = m_embeddings[tokens[token]];
        output.set_row_vector(token, embedding.data);
    }

    positional_encoding(output);

    return output;
}

void llm::positional_encoding(matrix& input) {
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            auto pe = 0.0f;

            if (i % 2 == 0) {
                pe = std::sin(i / std::pow(10000, j / static_cast<float>(input.cols)));
            } else {
                pe = std::cos(i / std::pow(10000, j / static_cast<float>(input.cols)));
            }

            input.offset(i, j, pe);
        }
    }
}

matrix llm::forward_l1(const matrix& input, const size_t layer) const {
    const auto& ff_layer = m_ff_layer.at(layer);

    matrix output = input.cross_multiply(ff_layer.w1);

    for (size_t i = 0; i < output.rows; ++i) {
        output.add_row_vector(i, ff_layer.b1);
    }

    return output;
}

matrix llm::activate(const matrix& input) {
    constexpr static auto relu = [](const float f) {
        return f < 0 ? 0 : f;
    };

    matrix output { input };
    return output.map(relu);
}

matrix llm::forward_l2(const matrix& input, const size_t layer) const {
    constexpr static auto relu = [](const float f) {
        return f < 0 ? 0 : f;
    };

    const auto& ff_layer = m_ff_layer.at(layer);

    matrix output = input.cross_multiply_map(ff_layer.w2, relu);

    for (size_t i = 0; i < output.rows; ++i) {
        output.add_row_vector(i, ff_layer.b2);
    }

    return output;
}