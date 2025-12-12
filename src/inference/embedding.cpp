#include "embedding.hpp"

#include <kernels/matrix_kernels.hpp>
#include <kernels/optimizer.hpp>

#include <cassert>
#include <cmath>

EmbeddingLayer::EmbeddingLayer(size_t vocab_size, size_t dimensions)
    : m_embeddings(vocab_size, dimensions) {}

size_t EmbeddingLayer::parameterCount() const {
    return m_embeddings.size();
}

void EmbeddingLayer::randomize(float min, float max) {
    m_embeddings.randomize(min, max);
}

static void positional_encoding(matrix& input) {
    for (size_t token_i = 0; token_i < input.rows; ++token_i) {
        for (size_t encoding_i = 0; encoding_i < input.cols / 2; ++encoding_i) {
            const auto offset = encoding_i * 2;
            const auto inner
                = token_i
                  / std::pow(10000, offset / static_cast<float>(input.cols));
            input.offset(token_i, offset, std::sin(inner));
            input.offset(token_i, offset + 1, std::cos(inner));
        }
    }
}

matrix EmbeddingLayer::forward(const std::span<const token_id_t> tokens) const {
    matrix output = matrix(tokens.size(), this->get_dimensions());

    for (size_t i = 0; i < tokens.size(); ++i) {
        matrix embedding
            = kernel::matrix::get_row_vector(m_embeddings, tokens[i]);
        output.set_row_vector(i, embedding);
    }

    positional_encoding(output);
    return output;
}

void EmbeddingLayer::backpropogate(const std::span<const token_id_t> tokens,
                                   const matrix& x_gradient,
                                   float learning_rate) {
#pragma omp parallel for
    for (size_t t = 0; t < tokens.size() - 1; t++) {
        const auto& token = tokens[t];
        matrix embedding = kernel::matrix::get_row_vector(m_embeddings, token);
        matrix embedding_gradient_row
            = kernel::matrix::get_row_vector(x_gradient, t);
        adjust_parameter_matrix(embedding, embedding_gradient_row,
                                learning_rate);
    }
}

void EmbeddingLayer::save(std::ostream& out) const {
    m_embeddings.save(out);
}

EmbeddingLayer EmbeddingLayer::load(std::istream& in) {
    return { matrix::load(in) };
}
