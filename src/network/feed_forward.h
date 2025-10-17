#pragma once

#include <util/matrix.h>

// ---[ Data Structs ]---
struct forward_result {
    matrix layer_input;
    matrix activation_input;
    matrix activation_output;
};

struct ff_apply_result {
    matrix output;
    forward_result forward_result;
};

// ---[ Data Structs ]---
struct ff_layer {
    matrix w1, b1;
    matrix w2, b2;

    ff_layer(size_t dimensions, size_t projection_size)
        : w1({ dimensions, projection_size }), b1({ 1, projection_size }),
          w2({ projection_size, dimensions }), b2({ 1, dimensions }) {}
    
    void randomize(float min, float max);
    ff_apply_result apply(const matrix &input) const;

private:
    static matrix forward_l1(const ff_layer& layer, const matrix& input);
    static matrix activate(const matrix& input);
    static matrix forward_l2(const ff_layer& layer, const matrix& input);
};
