#include "neural_net.h"

#include <iostream>
#include <sstream>

void llm::randomize() {
    constexpr auto min = -0.5f;
    constexpr auto max = 0.5f;

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

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto &embedding = m_embeddings[tokens[i]];
        output.set_row_vector(i, embedding.data);
    }

    positional_encoding(output);
    return output;
}

void llm::positional_encoding(matrix& input) {
    for (size_t token_i = 0; token_i < input.rows; ++token_i) {
        for (size_t encoding_i = 0; encoding_i < input.cols / 2; ++encoding_i) {
            const auto inner = token_i / std::pow(10000, 2 * encoding_i / static_cast<float>(input.cols));

            input.offset(token_i, encoding_i, std::sin(inner));
            input.offset(token_i, encoding_i + 1, std::cos(inner));
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
    constexpr static auto leaky_relu = [](const float f) {
        return f < 0 ? 0.01f * f : f;
    };

    matrix output { input };
    return output.map(leaky_relu);
}

matrix llm::forward_l2(const matrix& input, const size_t layer) const {
    const auto& ff_layer = m_ff_layer.at(layer);

    matrix output = input.cross_multiply(ff_layer.w2);

    for (size_t i = 0; i < output.rows; ++i) {
        output.add_row_vector(i, ff_layer.b2);
    }

    return output;
}

matrix llm::generate_logits(const matrix& input) const {
    matrix logits = input.cross_multiply(m_logit_layer.w);

    for (size_t i = 0; i < logits.rows; ++i) {
        logits.add_row_vector(i, m_logit_layer.b);
    }

    return logits;
}

matrix llm::feed_forward(const matrix& input, const size_t layer) const {
    const matrix l1_output = forward_l1(input, layer);
    const matrix activated = activate(l1_output);
    const matrix l2_output = forward_l2(activated, layer);

    return l2_output;
}

matrix llm::prediction_matrix(const std::span<const token_id_t> tokens) const {
    matrix acc = embed_tokens(tokens);

    for (size_t i = 0; i < m_ff_layer.size(); ++i) {
        const matrix l1_output = forward_l1(acc, i);
        const matrix activated = activate(l1_output);
        const matrix l2_output = forward_l2(activated, i);

        acc.offset(l2_output);
    }

    const matrix logits = generate_logits(acc);
    return logits;
}

token_id_t llm::predict(const std::span<const token_id_t> tokens) const {
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

std::string llm::to_string() const {
    std::stringstream ss;
    ss << "LLM with " << m_embeddings.size() << " embeddings and " << m_ff_layer.size() << " layers.\n";
    for (size_t i = 0; i < m_ff_layer.size(); ++i) {
        ss << "Layer " << i + 1 << ": W1 (" << m_ff_layer[i].w1.rows << " x " << m_ff_layer[i].w1.cols
           << "), W2 (" << m_ff_layer[i].w2.rows << " x " << m_ff_layer[i].w2.cols << ")\n";
        ss << "\nff w1 i=" << i << " " << m_ff_layer[i].w1.to_string() << "\n";
        ss << "\nff b1 i=" << i << " " << m_ff_layer[i].b1.to_string() << "\n";
        ss << "\nff w2 i=" << i << " " << m_ff_layer[i].w2.to_string() << "\n";
        ss << "\nff b2 i=" << i << " " << m_ff_layer[i].b2.to_string() << "\n";
    }
    return ss.str();
}