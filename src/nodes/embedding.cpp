#include "embedding.hpp"

#include <cmath>
#include <training/backpropogation.hpp>

void Embeddings::randomize(const float min, const float max) {
    this->m_data.randomize();
}

void EmbeddingLayer::save(std::ostream &out) const {
    out << this->m_dimensions << '|' << this->m_embeddings.size() << '|';
}

static EmbeddingLayer load(std::istream &in) {
    size_t dimensions, vocab_size;

    in >> dimensions;
    in.get();  // consume '|'
    in >> vocab_size;
    in.get();  // consume '|'

    EmbeddingLayer layer{ vocab_size, dimensions };

    for (size_t i = 0; i < vocab_size; ++i) {
        layer.m_embeddings.emplace_back(matrix::load(in));
    }

    return layer;
}

static void positional_encoding(matrix &input) {
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

std::vector<matrix> EmbeddingLayer::forward(
    const std::span<const token_id_t> tokens) const {
        
    matrix output { tokens.size(), m_dimensions };

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto &embedding = m_embeddings[tokens[i]];
        output.set_row_vector(i, embedding.m_data);
    }

    positional_encoding(output);

    return matrix::construct_vec(std::move(output));
}

void EmbeddingLayer::backpropagate(std::span<const token_id_t> tokens,
                                   std::span<const matrix> gradients,
                                   float learning_rate) {
    const auto &x_gradient = gradients[0];
    const size_t token_count = tokens.size();

#pragma omp parallel for
    for (size_t t = 0; t < tokens.size() - 1; t++) {
        const auto &token = tokens[t];
        auto &embedding = this->m_embeddings[token];

        matrix embedding_gradient_row{ 1, embedding.m_data.cols };
        for (size_t i = 0; i < embedding.m_data.cols; i++) {
            embedding_gradient_row.set(0, i, x_gradient.get(t, i));
        }

        regularize_weight_gradient(embedding_gradient_row, embedding.m_data);
        adjust_matrix(embedding.m_data, embedding_gradient_row, learning_rate);
    }
}
