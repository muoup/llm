#include "attention.h"
#include <cmath>

// ---[ Layer Operations ]---
void attention_layer::randomize(const float min, const float max) {
    wq.randomize(min, max);
    wk.randomize(min, max);
    wv.randomize(min, max);
    wo.randomize(min, max);
}

attention_apply_result attention_layer::apply(const matrix &input) const {
    attention_forward_result forward_result;
    forward_result.q = input.cross_multiplied(wq);
    forward_result.k = input.cross_multiplied(wk);
    forward_result.v = input.cross_multiplied(wv);

    // Attention scores
    forward_result.scores = forward_result.q.cross_multiplied(forward_result.k.transposed());

    // Scale
    const float scale = 1.0f / std::sqrt(static_cast<float>(forward_result.q.cols));
    forward_result.scores.scale(scale);

    // Mask
    forward_result.scores.mask_upper_triangular();
    forward_result.scores.softmax();

    // Weighted sum
    matrix output = forward_result.scores.cross_multiplied(forward_result.v);
    forward_result.output = std::move(output);

    // Output projection
    return { .output = forward_result.output.cross_multiplied(wo), .forward_result = std::move(forward_result) };
}
