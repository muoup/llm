#pragma once

#include <string>
#include <vector>

#include <tokenizer/token.h>
#include <util/matrix.h>

struct llm;

// ---[ Serialization ]---
void save_llm(const llm &model, const std::string &path);
void load_llm(llm &model, const std::string &path);

// ---[ Data Structs ]---
struct embedding {
    matrix data;

    explicit embedding(size_t dimensions) : data({ 1, dimensions }) {}
};

struct ff_layer {
    matrix w1, b1;
    matrix w2, b2;

    ff_layer(size_t dimensions, size_t projection_size)
        : w1({ dimensions, projection_size }), b1({ 1, projection_size }),
          w2({ projection_size, dimensions }), b2({ 1, dimensions }) {}
};

struct logit_layer {
    size_t vocab_size;
    matrix w, b;

    explicit logit_layer(const size_t dimensions, const size_t vocab_size)
        : vocab_size(vocab_size), w(dimensions, vocab_size), b(1, vocab_size) {}
};

struct llm {
    void randomize();

    matrix embed_tokens(std::span<const token_id_t> tokens) const;
    void positional_encoding(matrix &input) const;
    matrix feed_forward(const matrix &input, size_t layer) const;
    matrix prediction_matrix(std::span<const token_id_t> tokens) const;
    token_id_t predict(std::span<const token_id_t> tokens) const;

    matrix forward_l1(const matrix &input, size_t layer) const;
    matrix activate(const matrix &input) const;
    matrix forward_l2(const matrix &input, size_t layer) const;
    
    matrix generate_logits(const matrix& input) const;

    std::string to_string() const;

    llm(const size_t vocab_size, const size_t layer_count,
        const size_t dimensions, const size_t projection_scale = 4)
        : m_dimensions(dimensions), m_layer_count(layer_count),
          m_embeddings(vocab_size, embedding{ dimensions }),
          m_ff_layer(layer_count,
                     ff_layer{ dimensions, dimensions * projection_scale }),
          m_logit_layer(dimensions, vocab_size) {}

    size_t vocab_size() const { return m_logit_layer.vocab_size; }

    size_t m_dimensions;
    size_t m_layer_count;

    std::vector<embedding> m_embeddings;
    std::vector<ff_layer> m_ff_layer;
    logit_layer m_logit_layer;
};
