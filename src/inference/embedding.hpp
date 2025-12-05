#pragma once

#include <iostream>
#include <vector>
#include <span>
#include <tokenizer/token.hpp>
#include <util/matrix.hpp>

// Forward declaration
struct InferenceModel;

class Embeddings {
public:
    explicit Embeddings(size_t dimensions) : m_data({ 1, dimensions }) {}
    explicit Embeddings(matrix &&m_data) : m_data(std::move(m_data)) {}
    void randomize(float min, float max);

private:
    matrix m_data;
    friend class EmbeddingLayer;
    friend void backpropogate_embedding(InferenceModel &model, const std::span<const token_id_t> tokens, const matrix &x_gradient, float learning_rate);
};

// EmbeddingLayer itself is not an INode. It is a special entry layer mapping
// from token IDs to the embedding matrix.
class EmbeddingLayer {
public:
    EmbeddingLayer(size_t vocab_size, size_t dimensions);
    
    size_t parameterCount() const;

    matrix forward(const std::span<const token_id_t> inputs) const;
    void backpropogate(const std::span<const token_id_t> tokens, const matrix &x_gradient, float learning_rate);
    
    void randomize(float min, float max);

    void save(std::ostream &out) const;
    static EmbeddingLayer load(std::istream &in);

    size_t get_vocab_size() const { return m_embeddings.size(); }

private:
    std::vector<Embeddings> m_embeddings;
    size_t m_dimensions;
};
