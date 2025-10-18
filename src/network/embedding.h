#pragma once

#include <util/matrix.h>
#include <tokenizer/token.h>

// ---[ Data Structs ]---
struct embedding {
    matrix data;

    explicit embedding(size_t dimensions) : data({ 1, dimensions }) {}
 
    void randomize(float min, float max);
};

struct embedding_layer {
    std::vector<embedding> m_embeddings;
    size_t m_dimensions;

    embedding_layer(size_t vocab_size, size_t dimensions)
        : m_dimensions(dimensions) {
            for (size_t i = 0; i < vocab_size; ++i) {
                m_embeddings.emplace_back(dimensions);
            }
        }

    void randomize(float min, float max);
    matrix apply(std::span<const token_id_t> tokens) const;
};
