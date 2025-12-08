#include "linearized_attention.hpp"
#include "training/optimizer.hpp"

LinearizedAttention::LinearizedAttention()
    : dimensions(0), head_size(0), head_count(0), heads(), wo() {}

LinearizedAttention::LinearizedAttention(size_t dimensions, size_t head_count)
    : dimensions(dimensions),
      head_size(dimensions / head_count),
      head_count(head_count),
      wo(matrix(head_size * head_count, dimensions)) {
    for (size_t i = 0; i < head_count; ++i) {
        heads.emplace_back(LinearAttentionHead{
            matrix(dimensions, head_size),  // wq
            matrix(dimensions, head_size),  // wk
            matrix(dimensions, head_size)   // wv
        });
    }
}

size_t LinearizedAttention::parameterCount() const {
    size_t count = 0;
    for (const auto& head : heads) {
        count += head.wq.rows * head.wq.cols;
        count += head.wk.rows * head.wk.cols;
        count += head.wv.rows * head.wv.cols;
    }
    count += wo.rows * wo.cols;
    return count;
}

std::vector<matrix> LinearizedAttention::forward(
    std::span<const matrix> inputs) const {
    const matrix& input = inputs[0];

    std::vector<matrix> returns;
    // placeholder for the final output to prevent insert(0) additional overhead
    returns.emplace_back();
    // placeholder for the concatenated heads
    returns.emplace_back(input.rows, head_count * head_size);

    for (size_t h = 0; h < head_count; ++h) {
        const LinearAttentionHead& head = heads[h];
        matrix q = input.cross_multiplied(head.wq);
        matrix k = input.cross_multiplied(head.wk);
        matrix v = input.cross_multiplied(head.wv);

        // Attention (approx) = Q x (K^T x V) / sqrt(d_k)

        // Scores = (K^T x V)
        matrix scores = k.t_cross_multiplied(v);
        const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
        scores.scale(scale);

        // Mask & Softmax
        scores.mask_upper_triangular();
        scores.softmax();

        // Weighted sum
        matrix weighted_sum = q.cross_t_multiplied(scores);

        matrix& concatenated_heads = returns[1];
        concatenated_heads.set_horizontal_slice(h * head_size, weighted_sum);

        // returns is not modified, so the output reference will remain valid
        // until this point
        returns.emplace_back(std::move(q));
        returns.emplace_back(std::move(k));
        returns.emplace_back(std::move(v));
        returns.emplace_back(std::move(scores));
    }

    matrix& final_output = returns[0];
    matrix& concatenated_heads = returns[1];

    final_output = concatenated_heads.cross_multiplied(wo);
    final_output.add(input);  // Residual connection

    // Expected returns layout:
    // [0] -> concatenated heads
    // [1] -> final output
    // [2] -> q1 (queries head 1)
    // [3] -> k1 (keys head 1)
    // [4] -> v1 (values head 2)
    // [5] -> scores1 (attention scores after softmax head 1)
    // [6] -> q2 (queries head 2)
    // [7] -> k2 (keys head 2)
    // ...

    return returns;
}

std::vector<matrix> LinearizedAttention::backpropogate(
    std::span<const matrix> inputs,
    std::span<const matrix> outputs,
    std::span<const matrix> gradients,
    float learning_rate) {
    // These functions are assumed to be in "training/optimizer.hpp"
    constexpr float regularization_strength = 0.01f;

    const matrix& layer_input = inputs[0];
    const matrix& post_layer_gradient = gradients[0];

    // Backpropagate through the residual connection.
    // The gradient from the output flows back to both the layer's input and the
    // pre-residual output.
    matrix input_gradient = post_layer_gradient.clone();

    // Backpropagate through the final projection: final_output =
    // concatenated_heads * wo.T
    matrix wo_gradient = post_layer_gradient.t_cross_multiplied(outputs[1]);
    matrix weighted_sum_gradient_full
        = post_layer_gradient.cross_t_multiplied(wo);

    regularize_weight_gradient(wo_gradient, wo, regularization_strength);
    adjust_parameter_matrix(wo, wo_gradient, learning_rate);

    for (size_t h = 0; h < head_count; ++h) {
        const matrix& q = outputs[2 + h * 4 + 0];
        const matrix& k = outputs[2 + h * 4 + 1];
        const matrix& v = outputs[2 + h * 4 + 2];
        const matrix& scores = outputs[2 + h * 4 + 3];

        matrix weighted_sum_gradient
            = weighted_sum_gradient_full.get_horizontal_slice(h * head_size,
                                                              head_size);

        // Backprop through weighted sum: weighted_sum = q * scores.T
        // Your version had this part mixed up with standard attention.
        matrix q_gradient = weighted_sum_gradient.cross_t_multiplied(scores);
        matrix scores_gradient_after_softmax
            = weighted_sum_gradient.t_cross_multiplied(q).transposed();

        // Backprop through softmax and mask
        matrix scores_gradient
            = scores.backprop_softmax(scores_gradient_after_softmax);
        scores_gradient.mask_upper_triangular();

        // Backprop through scaling
        const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
        scores_gradient.scale(scale);

        // Backprop through scores = k.T * v
        // This is the other key part that was incorrect in your attempt.
        matrix k_gradient = v.cross_multiplied(scores_gradient);
        matrix v_gradient = k.t_cross_multiplied(scores_gradient);

        // Backprop through Q, K, V projections to their weight matrices
        matrix wq_gradient = q_gradient.t_cross_multiplied(layer_input);
        matrix wk_gradient = k_gradient.t_cross_multiplied(layer_input);
        matrix wv_gradient = v_gradient.t_cross_multiplied(layer_input);

        LinearAttentionHead& head = heads[h];
        regularize_weight_gradient(wq_gradient, head.wq,
                                   regularization_strength);
        adjust_parameter_matrix(head.wq, wq_gradient, learning_rate);
        regularize_weight_gradient(wk_gradient, head.wk,
                                   regularization_strength);
        adjust_parameter_matrix(head.wk, wk_gradient, learning_rate);
        regularize_weight_gradient(wv_gradient, head.wv,
                                   regularization_strength);
        adjust_parameter_matrix(head.wv, wv_gradient, learning_rate);

        // Gradient of the input: sum of contributions via each projection
        input_gradient.add(q_gradient.cross_t_multiplied(head.wq));
        input_gradient.add(k_gradient.cross_t_multiplied(head.wk));
        input_gradient.add(v_gradient.cross_t_multiplied(head.wv));
    }

    return matrix::construct_vec(input_gradient);
}

void LinearizedAttention::randomize(float min, float max) {
    for (auto& head : heads) {
        head.wq.randomize(min, max);
        head.wk.randomize(min, max);
        head.wv.randomize(min, max);
    }
    wo.randomize(min, max);
}

void LinearizedAttention::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&head_count), sizeof(head_count));

    for (const auto& head : heads) {
        head.wq.save(out);
        head.wk.save(out);
        head.wv.save(out);
    }
    wo.save(out);
}

LinearizedAttention LinearizedAttention::load(std::istream& in) {
    size_t dimensions;
    size_t head_count;

    in.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    in.read(reinterpret_cast<char*>(&head_count), sizeof(head_count));

    LinearizedAttention layer;
    layer.dimensions = dimensions;
    layer.head_count = head_count;

    for (size_t i = 0; i < head_count; ++i) {
        layer.heads.emplace_back(
            /*  .wq = */ matrix::load(in),
            /*  .wk = */ matrix::load(in),
            /*  .wv = */ matrix::load(in));
    }

    layer.wo = matrix::load(in);
    return layer;
}
