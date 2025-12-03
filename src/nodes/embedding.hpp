#pragma once

#include <iostream>
#include <nodes/network_node.hpp>
#include <tokenizer/token.hpp>
#include <util/matrix.hpp>

struct Embeddings {
    matrix m_data;

    explicit Embeddings(size_t dimensions) : m_data({ 1, dimensions }) {}
    explicit Embeddings(matrix &&data) : m_data(std::move(data)) {}

    void randomize(float min, float max);
};

struct embedding_layer {
    std::vector<Embeddings> m_embeddings;
    size_t m_dimensions;

    embedding_layer(size_t vocab_size, size_t dimensions)
        : m_dimensions(dimensions) {
        for (size_t i = 0; i < vocab_size; ++i) {
            m_embeddings.emplace_back(dimensions);
        }
    }

    void randomize(float min, float max);

    matrix forward(std::span<const token_id_t> tokens) const;
    void backpropagate(std::span<const token_id_t> tokens,
                       const matrix &grad_output, float learning_rate);

    void save(std::iostream &out) const;
    static embedding_layer load(std::iostream &in);
};

// EmbeddingLayer itself is not an INode, exceptions must be made for the input
// and output types as they are not pure mappings from matrices to matrices, in
// this case, the embedding layer is a mapping from token IDs to the embedding
// matrix.
struct EmbeddingLayer {
    std::vector<Embeddings> m_embeddings;
    size_t m_dimensions;

    EmbeddingLayer(size_t vocab_size, size_t dimensions)
        : m_dimensions(dimensions) {
        for (size_t i = 0; i < vocab_size; ++i) {
            m_embeddings.emplace_back(dimensions);
        }
    }

    std::vector<matrix> forward(const std::span<const token_id_t> inputs) const;
    void backpropagate(const std::span<const token_id_t> tokens,
                       const std::span<const matrix> gradients, float learning_rate);

    void randomize(float min, float max);

    void save(std::ostream &out) const;
    static INode *load(std::istream &in);
};
