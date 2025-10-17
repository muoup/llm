#include "feed_forward.h"

// ---[ Layer Operations ]---

void ff_layer::randomize(const float min, const float max) {
    w1.randomize(min, max);
    b1.randomize(min, max);
    w2.randomize(min, max);
    b2.randomize(min, max);
}

// ---[ Operations ]---

matrix ff_layer::forward_l1(const ff_layer& layer, const matrix& input) {
    matrix output = input.cross_multiply(layer.w1);
    
    for (size_t i = 0; i < output.rows; ++i) {
        output.add_row_vector(i, layer.b1);
    }
    
    return output;
}

matrix ff_layer::activate(const matrix& input) {
    constexpr static auto leaky_relu = [](const float f) { return f < 0 ? 0.01f * f : f; };
    matrix output { input };
    return output.map(leaky_relu);
}

matrix ff_layer::forward_l2(const ff_layer& layer, const matrix& input) {
    matrix output = input.cross_multiply(layer.w2);
    
    for (size_t i = 0; i < output.rows; ++i) {
        output.add_row_vector(i, layer.b2);
    }
    
    return output;
}

ff_apply_result ff_layer::apply(const matrix& input) const {
    const matrix l1_output = ff_layer::forward_l1(*this, input);
    const matrix activated = ff_layer::activate(l1_output);
    const matrix l2_output = ff_layer::forward_l2(*this, activated);
    
    return {l2_output, {input, l1_output, activated}};
}