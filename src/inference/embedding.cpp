#include "kernels/optimizer.hpp"
#include "embedding.hpp"

#include <cassert>
#include <cmath>

void Embeddings::randomize(const float min, const float max) {
    m_data.randomize(min, max);
}

EmbeddingLayer::EmbeddingLayer(size_t vocab_size, size_t dimensions)
    : m_dimensions(dimensions) {
    m_embeddings.reserve(vocab_size);
    
    for (size_t i = 0; i < vocab_size; ++i) {
        m_embeddings.emplace_back(dimensions);
    }
}

size_t EmbeddingLayer::parameterCount() const {
    return m_embeddings.size() * m_dimensions;
}

void EmbeddingLayer::randomize(float min, float max) {
    for (auto &embedding : m_embeddings) {
        embedding.randomize(min, max);
    }
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

matrix EmbeddingLayer::forward(const std::span<const token_id_t> tokens) const {
    matrix output = matrix(tokens.size(), m_dimensions);

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto &embedding = m_embeddings[tokens[i]];
        output.set_row_vector(i, embedding.m_data);
    }

    positional_encoding(output);
    return output;
}

void EmbeddingLayer::backpropogate(const std::span<const token_id_t> tokens,
                                   const matrix &x_gradient,
                                   float learning_rate) {
#pragma omp parallel for
    for (size_t t = 0; t < tokens.size() - 1; t++) {
        const auto &token = tokens[t];
        auto &embedding = m_embeddings[token];

        matrix embedding_gradient_row({ 1, embedding.m_data.cols });
        for (size_t i = 0; i < embedding.m_data.cols; i++) {
            embedding_gradient_row.set(0, i, x_gradient.get(t, i));
        }

        adjust_parameter_matrix(embedding.m_data, embedding_gradient_row, learning_rate);
    }
}

void EmbeddingLayer::save(std::ostream &out) const {
    const size_t embeddings = this->m_embeddings.size();
    const size_t dimensions = this->m_dimensions;
    
    out.write(reinterpret_cast<const char*>(&embeddings), sizeof(embeddings));
    out.put('|');
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.put('|');

    for (const auto &embedding : m_embeddings) {
        embedding.m_data.save(out);
    }
}

EmbeddingLayer EmbeddingLayer::load(std::istream &in) {
    size_t embeddings, dimensions;
    char pipe;

    in.read(reinterpret_cast<char*>(&embeddings), sizeof(embeddings));
    in.get(pipe);
    assert(pipe == '|');
    
    in.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    in.get(pipe);
    assert(pipe == '|');

    EmbeddingLayer layer(embeddings, dimensions);

    for (size_t i = 0; i < embeddings; ++i) {
        layer.m_embeddings.at(i).m_data = matrix::load(in);
    }

    return layer;
}
