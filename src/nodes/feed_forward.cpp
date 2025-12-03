#include "feed_forward.hpp"

// ---[ Layer Operations ]---

void ff_layer::randomize(const float min, const float max) {
    w1.randomize(min, max);
    b1.randomize(min, max);
    w2.randomize(min, max);
    b2.randomize(min, max);
}

// ---[ Operations ]---

matrix ff_layer::forward_l1(const ff_layer& layer, const matrix& input) {
    matrix output = input.cross_multiplied(layer.w1);

    for (size_t i = 0; i < output.rows; ++i) {
        output.add_row_vector(i, layer.b1);
    }

    return output;
}

matrix ff_layer::activate(const matrix& input) {
    constexpr static auto leaky_relu = [](const float f) { return f < 0 ? 0.01f * f : f; };
    matrix output = std::move(input.clone().map(leaky_relu));
    
    return output;
}

matrix ff_layer::forward_l2(const ff_layer& layer, const matrix& input) {
    matrix output = input.cross_multiplied(layer.w2);

    for (size_t i = 0; i < output.rows; ++i) {
        output.add_row_vector(i, layer.b2);
    }

    return output;
}

ff_apply_result ff_layer::apply(const matrix& input) const {
    matrix l1_output = ff_layer::forward_l1(*this, input);
    matrix activated = ff_layer::activate(l1_output);
    matrix l2_output = ff_layer::forward_l2(*this, activated);

    return { 
        .output = std::move(l2_output), 
        .forward_result = {
            .layer_input = input.clone(), 
            .activation_input = std::move(l1_output), 
            .activation_output = std::move(activated)
        }
    };
}
