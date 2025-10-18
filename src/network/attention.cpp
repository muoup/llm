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
    forward_result.q = input.cross_multiply(wq);
    forward_result.k = input.cross_multiply(wk);
    forward_result.v = input.cross_multiply(wv);

    // Attention scores
    forward_result.scores = forward_result.q.cross_multiply(forward_result.k.transposed());

    // Scale
    const float scale = 1.0f / std::sqrt(static_cast<float>(forward_result.q.cols));
    forward_result.scores.scale(scale);

    // Softmax
    for (size_t i = 0; i < forward_result.scores.rows; ++i) {
        float max_val = forward_result.scores.get(i, 0);
        for (size_t j = 1; j < forward_result.scores.cols; ++j) {
            if (forward_result.scores.get(i, j) > max_val) {
                max_val = forward_result.scores.get(i, j);
            }
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < forward_result.scores.cols; ++j) {
            const float val = std::exp(forward_result.scores.get(i, j) - max_val);
            forward_result.scores.set(i, j, val);
            sum_exp += val;
        }

        for (size_t j = 0; j < forward_result.scores.cols; ++j) {
            forward_result.scores.set(i, j, forward_result.scores.get(i, j) / sum_exp);
        }
    }

    // Weighted sum
    matrix output = forward_result.scores.cross_multiply(forward_result.v);
    forward_result.output = std::move(output);

    // Output projection
    return { forward_result.output.cross_multiply(wo), forward_result };
}
