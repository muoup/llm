#pragma once

#include <util/matrix.h>

struct attention_forward_result {
    matrix q, k, v;
    matrix scores;
};

struct attention_apply_result {
    matrix output;
    attention_forward_result forward_result;
};

// ---[ Data Structs ]---
struct attention_layer {
    matrix wq, wk, wv, wo;

    attention_layer(size_t dimensions, size_t head_size)
        : wq({ dimensions, head_size }), wk({ dimensions, head_size }),
          wv({ dimensions, head_size }), wo({ head_size, dimensions }) {}

    void randomize(float min, float max);
    attention_apply_result apply(const matrix &input) const;
};
