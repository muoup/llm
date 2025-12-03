#pragma once

#include <util/matrix.hpp>

#include <istream>

struct attention_forward_result {
    matrix q, k, v;
    matrix scores;
    matrix output;
};

struct attention_apply_result {
    matrix output;
    attention_forward_result forward_result;
};

struct attention_layer {
    matrix wq, wk, wv, wo;

    attention_layer(size_t dimensions, size_t head_size)
        : wq({ dimensions, head_size }), wk({ dimensions, head_size }),
          wv({ dimensions, head_size }), wo({ head_size, dimensions }) {}

    void randomize(float min, float max);
    
    attention_apply_result forward(const matrix &input) const;
    void backpropogate(const attention_forward_result &forward_result,
                       const matrix &grad_output, float learning_rate);
    
    void save(std::ostream& out) const;
    static attention_layer load(std::istream& in);
};
