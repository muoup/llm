#include "attention.hpp"

#include <cmath>
#include <iostream>

#include "training/optimizer.hpp"

// ---[ Construction ]---

NodeType AttentionLayer::getType() const { return NodeType::Attention; }

AttentionLayer::AttentionLayer(size_t dimensions, size_t head_size,
                               size_t head_count)
    : dimensions(dimensions),
      head_size(head_size),
      head_count(head_count),
      wq({ dimensions, head_size }),
      wk({ dimensions, head_size }),
      wv({ dimensions, head_size }),
      wo({ head_size, dimensions }) {}

// ---[ Layer Operations ]---

void AttentionLayer::randomize(const float min, const float max) {
    wq.randomize(min, max);
    wk.randomize(min, max);
    wv.randomize(min, max);
    wo.randomize(min, max);
}

std::vector<matrix> AttentionLayer::forward(std::span<const matrix> inputs) {
    const matrix& input = inputs[0];

    matrix q = input.cross_multiplied(wq);
    matrix k = input.cross_multiplied(wk);
    matrix v = input.cross_multiplied(wv);

    // Attention scores
    matrix scores = q.cross_multiplied(k.transposed());

    // Scale
    const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
    scores.scale(scale);

    // Mask & Softmax
    scores.mask_upper_triangular();
    scores.softmax();

    // Weighted sum
    matrix weighted_sum = scores.cross_multiplied(v);

    // Output projection
    matrix attn_output = weighted_sum.cross_multiplied(wo);

    // Residual connection
    matrix final_output = input.clone();
    final_output.add(attn_output);

    return matrix::construct_vec(
        final_output,  // [0] = final output
        q,             // [1] = q
        k,             // [2] = k
        v,             // [3] = v
        scores,        // [4] = scores
        weighted_sum   // [5] = weighted_sum (pre-output-projection)
    );
}

std::vector<matrix> AttentionLayer::backpropogate(
    std::span<const matrix> inputs, std::span<const matrix> outputs,
    std::span<const matrix> gradients, float learning_rate) {
    constexpr float regularization_strength = 0.01f;

    const matrix& layer_input = inputs[0];
    const matrix& q = outputs[1];
    const matrix& k = outputs[2];
    const matrix& v = outputs[3];
    const matrix& scores = outputs[4];
    const matrix& weighted_sum = outputs[5];
    const matrix& post_layer_gradient = gradients[0];

    // Backprop through output projection (wo)
    matrix wo_gradient
        = weighted_sum.transposed().cross_multiplied(post_layer_gradient);
    matrix weighted_sum_gradient
        = post_layer_gradient.cross_multiplied(wo.transposed());

    regularize_weight_gradient(wo_gradient, wo, regularization_strength);
    adjust_matrix(wo, wo_gradient, learning_rate);

    // Backprop through weighted sum: weighted_sum = scores * v
    matrix v_gradient
        = scores.transposed().cross_multiplied(weighted_sum_gradient);
    matrix scores_gradient_v
        = weighted_sum_gradient.cross_multiplied(v.transposed());

    // Backprop through softmax
    matrix softmax_gradient({ scores_gradient_v.rows, scores_gradient_v.cols });
#pragma omp parallel for
    for (size_t i = 0; i < scores.rows; ++i) {
        float s_dot = 0.0f;
        for (size_t l = 0; l < scores.cols; ++l) {
            s_dot += scores_gradient_v.get(i, l) * scores.get(i, l);
        }
        for (size_t j = 0; j < scores.cols; ++j) {
            const float s_j = scores.get(i, j);
            const float g_j = scores_gradient_v.get(i, j);
            softmax_gradient.set(i, j, s_j * (g_j - s_dot));
        }
    }

    // Backprop through scaling
    const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
    softmax_gradient.scale(scale);

    // Backprop through attention scores to Q, K
    matrix q_gradient = softmax_gradient.cross_multiplied(k);
    matrix k_gradient = softmax_gradient.transposed().cross_multiplied(q);

    // Backprop through Q, K, V projections to their weight matrices
    matrix layer_input_t = layer_input.transposed();
    matrix wq_gradient = layer_input_t.cross_multiplied(q_gradient);
    matrix wk_gradient = layer_input_t.cross_multiplied(k_gradient);
    matrix wv_gradient = layer_input_t.cross_multiplied(v_gradient);

    regularize_weight_gradient(wq_gradient, wq, regularization_strength);
    adjust_matrix(wq, wq_gradient, learning_rate);
    regularize_weight_gradient(wk_gradient, wk, regularization_strength);
    adjust_matrix(wk, wk_gradient, learning_rate);
    regularize_weight_gradient(wv_gradient, wv, regularization_strength);
    adjust_matrix(wv, wv_gradient, learning_rate);

    // Gradient of the input: sum of contributions via each projection +
    // residual
    matrix input_gradient = q_gradient.cross_multiplied(wq.transposed());
    input_gradient.add(k_gradient.cross_multiplied(wk.transposed()));
    input_gradient.add(v_gradient.cross_multiplied(wv.transposed()));

    // Add gradient from the residual connection
    input_gradient.add(post_layer_gradient);

    return matrix::construct_vec(input_gradient);
}

// ---[ Serialization ]---

void AttentionLayer::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&head_size), sizeof(head_size));
    out.write(reinterpret_cast<const char*>(&head_count), sizeof(head_count));
    
    wq.save(out);
    wk.save(out);
    wv.save(out);
    wo.save(out);
}

AttentionLayer AttentionLayer::load(std::istream& in) {
    size_t dimensions, head_size, head_count;
    in.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    in.read(reinterpret_cast<char*>(&head_size), sizeof(head_size));
    in.read(reinterpret_cast<char*>(&head_count), sizeof(head_count));

    AttentionLayer layer(dimensions, head_size, head_count);

    layer.wq = matrix::load(in);
    layer.wk = matrix::load(in);
    layer.wv = matrix::load(in);
    layer.wo = matrix::load(in);

    return layer;
}
