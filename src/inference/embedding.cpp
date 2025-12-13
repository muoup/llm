#include "embedding.hpp"

#include <kernels/matrix_kernels.hpp>
#include <kernels/optimizer.hpp>
#include <kernels/embedding_layer.hpp>

#include <cassert>

size_t EmbeddingLayer::parameterCount() const {
    return m_embeddings.size();
}

void EmbeddingLayer::randomize(float min, float max) {
    m_embeddings.randomize(min, max);
}

matrix EmbeddingLayer::forward(const std::span<const token_id_t> tokens) const {
    matrix output = matrix(tokens.size(), this->get_dimensions());

    for (size_t i = 0; i < tokens.size(); ++i) {
        kernel::matrix::transfer_row(
            output, i, m_embeddings, tokens[i]);
    }

    kernel::embedding::positional_encoding(output);
    return output;
}

void EmbeddingLayer::backpropogate(const std::span<const token_id_t> tokens,
                                   const matrix& x_gradient,
                                   float learning_rate) {
    for (size_t t = 0; t < tokens.size(); t++) {
        const auto& token = tokens[t];
        matrix embedding = kernel::matrix::get_row_vector(m_embeddings, token);
        matrix embedding_gradient_row
            = kernel::matrix::get_row_vector(x_gradient, t);
        adjust_parameter_matrix(embedding, embedding_gradient_row,
                                learning_rate);
        
        m_embeddings.set_row_vector(token, embedding);
    }
}

void EmbeddingLayer::save(std::ostream& out) const {
    m_embeddings.save(out);
}

EmbeddingLayer EmbeddingLayer::load(std::istream& in) {
    return { matrix::load(in) };
}
