#include "embedding.hpp"

#include <kernels/layers/embedding_layer.hpp>
#include <kernels/matrix.hpp>
#include <kernels/optimizer.hpp>

#include <cassert>
#include <cmath>

#include <util/logger.hpp>
#include "kernels/scheduling.hpp"

size_t EmbeddingLayer::parameterCount() const {
    return m_embeddings.size();
}

void EmbeddingLayer::randomize(float min, float max) {
    m_embeddings.xavier_randomize();
}

matrix EmbeddingLayer::forward(const std::span<const token_id_t> tokens) const {
    matrix output = matrix(tokens.size(), this->get_dimensions());

    for (size_t i = 0; i < tokens.size(); ++i) {
        kernel::matrix::transfer_row(output, i, m_embeddings, tokens[i]);
        CHECK_ERRORS("EmbeddingLayer::forward row transfer");
    }

    output.scale(std::sqrt(static_cast<float>(this->get_dimensions())));

    LOG_DEBUG("  Embedding Layer Forward:");
    LOG_DEBUG("    output norm pre pos encoding: %f", output.norm());

    kernel::embedding::positional_encoding(output);
    LOG_DEBUG("    output norm: %f", output.norm());

    return output;
}

void EmbeddingLayer::backpropogate(const std::span<const token_id_t> tokens,
                                   const matrix& x_gradient,
                                   CentralOptimizer& optimizer) {
    matrix embedding_gradient(m_embeddings.rows, m_embeddings.cols);
    const float scale = std::sqrt(static_cast<float>(get_dimensions()));

    for (size_t t = 0; t < tokens.size(); t++) {
        const auto& token = tokens[t];
        kernel::matrix::atomic_add_row_vector(embedding_gradient, token, x_gradient, t,
                                              scale);
    }
    kernel::wait_for_all_streams();

    LOG_DEBUG("  Embedding Layer Gradients:");
    LOG_DEBUG("    embedding_gradient norm: %f", embedding_gradient.norm());

    // Sparse normalization: only normalized by the tokens actually updated
    // AdamW might be too aggressive for embeddings if we don't scale gradients by frequency.
    // However, for now, let's stick to standard AdamW update for simplicity.
    // The previous code had:
    // size_t normalization_count = tokens.size() * get_dimensions();
    // kernel::optimizer::regularize_weight_gradient(embedding_gradient, m_embeddings, nullptr, normalization_count);
    
    // We can simulate the normalization by scaling the gradient before passing to AdamW if needed,
    // but AdamW is adaptive.
    
    optimizer.update(m_embeddings, embedding_gradient);
    kernel::wait_for_all_streams();
}

void EmbeddingLayer::save(std::ostream& out) const {
    m_embeddings.save(out);
}

EmbeddingLayer EmbeddingLayer::load(std::istream& in) {
    return { matrix::load(in) };
}
