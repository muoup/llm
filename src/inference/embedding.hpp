#pragma once

#include <iostream>
#include <span>
#include <tokenizer/token.hpp>
#include <util/matrix.hpp>

// Forward declaration
struct InferenceModel;

// EmbeddingLayer itself is not an INode. It is a special entry layer mapping
// from token IDs to the embedding matrix.
class EmbeddingLayer {
public:
    EmbeddingLayer() : m_embeddings(0, 0) {}
    EmbeddingLayer(matrix&& embeddings) : m_embeddings(std::move(embeddings)) {}
    EmbeddingLayer(size_t dimensions, size_t vocab_size)
        : m_embeddings(vocab_size, dimensions) {}
    
    size_t parameterCount() const;

    matrix forward(const std::span<const token_id_t> inputs) const;
    void backpropogate(const std::span<const token_id_t> tokens, const matrix &x_gradient, float learning_rate);
    
    void randomize(float min, float max);

    void save(std::ostream &out) const;
    static EmbeddingLayer load(std::istream &in);

    size_t get_vocab_size() const { return m_embeddings.rows; }
    size_t get_dimensions() const { return m_embeddings.cols; }

    matrix m_embeddings;
};
