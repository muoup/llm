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
    matrix embedding_gradient(m_embeddings.rows, m_embeddings.cols);
                                       
    for (size_t t = 0; t < tokens.size(); t++) {
        const auto& token = tokens[t];
        kernel::matrix::add_row_vector(embedding_gradient, token, x_gradient, t);
    }
    
    kernel::optimizer::adjust_parameter_matrix(m_embeddings, embedding_gradient, learning_rate);
}

void EmbeddingLayer::save(std::ostream& out) const {
    m_embeddings.save(out);
}

EmbeddingLayer EmbeddingLayer::load(std::istream& in) {
    return { matrix::load(in) };
}
