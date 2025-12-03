#pragma once

#include <util/matrix.hpp>

// ---[ Data Structs ]---
struct logit_layer {
    size_t vocab_size;
    matrix w, b;

    explicit logit_layer(const size_t dimensions, const size_t vocab_size)
        : vocab_size(vocab_size), w(dimensions, vocab_size), b(1, vocab_size) {}
    
    void randomize(float min, float max);
    matrix apply(const matrix &input) const;
};
