#include "embedding.hpp"

#include <kernels/embedding_layer.hpp>
#include <kernels/matrix_kernels.hpp>
#include <kernels/optimizer.hpp>

#include <cassert>
#include <cmath>

#include <util/logger.hpp>

size_t EmbeddingLayer::parameterCount() const {
    return m_embeddings.size();
}

void EmbeddingLayer::randomize(float min, float max) {
    m_embeddings.xavier_randomize();
}

matrix EmbeddingLayer::forward(const std::span<const token_id_t> tokens) const {
    matrix output = matrix(tokens.size(), this->get_dimensions());
    kernel::optimizer::wait_for_operations();

    for (size_t i = 0; i < tokens.size(); ++i) {
        kernel::matrix::transfer_row(output, i, m_embeddings, tokens[i]);
        CHECK_ERRORS("EmbeddingLayer::forward row transfer");
    }

    output.scale(std::sqrt(static_cast<float>(this->get_dimensions())));

    LOG_DEBUG("  Embedding Layer Forward:");
    LOG_DEBUG("    output norm pre pos encoding: %f", output.norm());

    kernel::optimizer::wait_for_operations();
    kernel::embedding::positional_encoding(output);
    kernel::optimizer::wait_for_operations();

    LOG_DEBUG("    output norm: %f", output.norm());

    return output;
}

void EmbeddingLayer::backpropogate(const std::span<const token_id_t> tokens,
                                   const matrix& x_gradient,
                                   float learning_rate) {
    matrix embedding_gradient(m_embeddings.rows, m_embeddings.cols);
    kernel::optimizer::wait_for_operations();

    matrix scaled_x_gradient = x_gradient.clone();
    scaled_x_gradient.scale(std::sqrt(static_cast<float>(get_dimensions())));

    for (size_t t = 0; t < tokens.size(); t++) {
        const auto& token = tokens[t];
        kernel::matrix::add_row_vector(embedding_gradient, token, scaled_x_gradient,
                                       t);
        kernel::optimizer::wait_for_operations();
    }

    LOG_DEBUG("  Embedding Layer Gradients:");
    LOG_DEBUG("    embedding_gradient norm: %f", embedding_gradient.norm());

    kernel::optimizer::adjust_parameter_matrix(m_embeddings, embedding_gradient,
                                               learning_rate);
    kernel::optimizer::wait_for_operations();
}

void EmbeddingLayer::save(std::ostream& out) const {
    m_embeddings.save(out);
}

EmbeddingLayer EmbeddingLayer::load(std::istream& in) {
    return { matrix::load(in) };
}
