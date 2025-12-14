#include "attention.hpp"

#include <cmath>
#include <span>
#include <vector>

#include <inference/network_node.hpp>
#include <kernels/optimizer.hpp>
#include <util/logger.hpp>

NodeType AttentionLayer::getType() const {
    return NodeType::Attention;
}

AttentionLayer::AttentionLayer(size_t dimensions, size_t head_count)
    : dimensions(dimensions),
      head_size(dimensions / head_count),
      head_count(head_count),
      wo(matrix(head_size * head_count, dimensions)) {
    for (size_t i = 0; i < head_count; ++i) {
        heads.emplace_back(AttentionHead{
            matrix(dimensions, head_size),  // wq
            matrix(dimensions, head_size),  // wk
            matrix(dimensions, head_size)   // wv
        });
    }
}

void AttentionLayer::randomize(const float min, const float max) {
    for (auto& head : heads) {
        head.wq.randomize(min, max);
        head.wk.randomize(min, max);
        head.wv.randomize(min, max);
    }

    wo.randomize(min, max);
}

size_t AttentionLayer::parameterCount() const {
    size_t count = wo.rows * wo.cols;
    for (const auto& head : heads) {
        count += head.wq.rows * head.wq.cols;
        count += head.wk.rows * head.wk.cols;
        count += head.wv.rows * head.wv.cols;
    }
    return count;
}

ForwardingResult AttentionLayer::forward(std::span<const matrix> inputs) const {
    const matrix& input = inputs[0];

    std::vector<matrix> returns;
    // placeholder for the final output to prevent insert(0) additional overhead
    returns.emplace_back();
    // placeholder for the concatenated heads
    returns.emplace_back(input.rows, head_count * head_size);

    for (size_t h = 0; h < head_count; ++h) {
        const AttentionHead& head = heads[h];
        matrix q = input.cross_multiplied(head.wq);
        matrix k = input.cross_multiplied(head.wk);
        matrix v = input.cross_multiplied(head.wv);

        // Attention score = Q x K^T / sqrt(d_k)
        matrix scores = q.cross_t_multiplied(k);
        const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
        kernel::optimizer::wait_for_operations();
        
        scores.scale(scale);

        // Mask & Softmax
        scores.mask_upper_triangular();
        kernel::optimizer::wait_for_operations();
        scores.softmax();
        kernel::optimizer::wait_for_operations();

        // Weighted sum
        matrix weighted_sum = scores.cross_multiplied(v);
        kernel::optimizer::wait_for_operations();

        matrix& concatenated_heads = returns[1];
        concatenated_heads.set_horizontal_slice(h * head_size, weighted_sum);

        // returns is not modified, so the output reference will remain valid
        // until this point
        returns.emplace_back(std::move(q));
        returns.emplace_back(std::move(k));
        returns.emplace_back(std::move(v));
        returns.emplace_back(std::move(scores));
    }
    
    kernel::optimizer::wait_for_operations();

    matrix& final_output = returns[0];
    matrix& concatenated_heads = returns[1];

    final_output = concatenated_heads.cross_multiplied(wo);
    
    logger::log(LogLevel::DEBUG, "  Attention Layer Forward:");
    logger::log(LogLevel::DEBUG, "    input norm: %f",
                input.norm());
    logger::log(LogLevel::DEBUG, "    concatenated_heads norm: %f",
                concatenated_heads.norm());
    logger::log(LogLevel::DEBUG, "    final_output norm: %f",
                final_output.norm());

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

    return standardResult(std::move(returns));
}

std::vector<matrix> AttentionLayer::backpropogate(
    const ForwardingResult& result,
    std::span<const matrix> inputs,
    std::span<const matrix> gradients,
    float learning_rate) {
    constexpr float regularization_strength = 0.01f;

    const matrix& layer_input = inputs[0];
    const matrix& concat_heads = result.outputs[0];
    const matrix& layer_output = result.outputs[1];
    const matrix& post_layer_gradient = gradients[0];

    logger::log(LogLevel::DEBUG, "  Attention Layer Backpropagation:");
    logger::log(LogLevel::DEBUG, "    layer_output norm: %f",
                layer_output.norm());
    logger::log(LogLevel::DEBUG, "    post_layer_gradient norm: %f",
                post_layer_gradient.norm());

    matrix wo_gradient = layer_output.t_cross_multiplied(post_layer_gradient);
    matrix attention_concat_gradient
        = post_layer_gradient.cross_t_multiplied(wo);

    kernel::optimizer::regularize_weight_gradient(wo_gradient, wo);
    kernel::optimizer::adjust_parameter_matrix(wo, wo_gradient, learning_rate);

    matrix input_gradient(layer_input.rows, layer_input.cols);

    for (size_t h = 0; h < head_count; ++h) {
        const matrix& q = result.outputs[2 + h * 4 + 0];
        const matrix& k = result.outputs[2 + h * 4 + 1];
        const matrix& v = result.outputs[2 + h * 4 + 2];
        const matrix& scores = result.outputs[2 + h * 4 + 3];

        // Slice the gradient for the current head's output
        matrix attention_gradient
            = attention_concat_gradient.get_horizontal_slice(h * head_size,
                                                             head_size);
        kernel::optimizer::wait_for_operations();

        // Backprop through weighted sum: weighted_sum = scores * v
        matrix v_gradient = scores.t_cross_multiplied(attention_gradient);
        matrix scores_gradient = attention_gradient.cross_t_multiplied(v);
        kernel::optimizer::wait_for_operations();

        // Backprop through softmax
        matrix pre_softmax_gradient = scores.backprop_softmax(scores_gradient);
        kernel::optimizer::wait_for_operations();

        // Backprop through scaling
        const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
        pre_softmax_gradient.scale(scale);
        kernel::optimizer::wait_for_operations();

        matrix q_gradient = pre_softmax_gradient.cross_multiplied(k);
        matrix k_gradient = pre_softmax_gradient.t_cross_multiplied(q);
        kernel::optimizer::wait_for_operations();

        // Backprop through Q, K, V projections to their weight matrices
        matrix wq_gradient = layer_input.t_cross_multiplied(q_gradient);
        matrix wk_gradient = layer_input.t_cross_multiplied(k_gradient);
        matrix wv_gradient = layer_input.t_cross_multiplied(v_gradient);
        kernel::optimizer::wait_for_operations();

        logger::log(LogLevel::DEBUG, "  Attention Head %zu Gradients:", h);
        logger::log(LogLevel::DEBUG, "    scores_gradient norm: %f",
                    scores_gradient.norm());
        logger::log(LogLevel::DEBUG, "    pre_softmax_gradient norm: %f",
                    pre_softmax_gradient.norm());
        logger::log(LogLevel::DEBUG, "    wq_gradient norm: %f",
                    wq_gradient.norm());
        logger::log(LogLevel::DEBUG, "    wk_gradient norm: %f",
                    wk_gradient.norm());
        logger::log(LogLevel::DEBUG, "    wv_gradient norm: %f",
                    wv_gradient.norm());

        AttentionHead& head = heads[h];
        kernel::optimizer::regularize_weight_gradient(wq_gradient, head.wq);
        kernel::optimizer::regularize_weight_gradient(wk_gradient, head.wk);
        kernel::optimizer::regularize_weight_gradient(wv_gradient, head.wv);

        kernel::optimizer::adjust_parameter_matrix(head.wq, wq_gradient,
                                                   learning_rate);
        kernel::optimizer::adjust_parameter_matrix(head.wk, wk_gradient,
                                                   learning_rate);
        kernel::optimizer::adjust_parameter_matrix(head.wv, wv_gradient,
                                                   learning_rate);

        // Gradient of the input: sum of contributions via each projection +
        matrix head_input_gradient = q_gradient.cross_t_multiplied(head.wq);
        head_input_gradient.add(k_gradient.cross_t_multiplied(head.wk));
        head_input_gradient.add(v_gradient.cross_t_multiplied(head.wv));

        input_gradient.add(head_input_gradient);
    }

    return matrix::construct_vec(input_gradient);
}

void AttentionLayer::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&head_size), sizeof(head_size));
    out.write(reinterpret_cast<const char*>(&head_count), sizeof(head_count));

    for (const auto& head : heads) {
        head.wq.save(out);
        head.wk.save(out);
        head.wv.save(out);
    }

    wo.save(out);
}

AttentionLayer AttentionLayer::load(std::istream& in) {
    size_t dimensions, head_size, head_count;
    in.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    in.read(reinterpret_cast<char*>(&head_size), sizeof(head_size));
    in.read(reinterpret_cast<char*>(&head_count), sizeof(head_count));

    AttentionLayer layer = AttentionLayer();
    layer.dimensions = dimensions;
    layer.head_size = head_size;
    layer.head_count = head_count;

    for (size_t i = 0; i < head_count; ++i) {
        layer.heads.emplace_back(
            /* .wq = */ matrix::load(in),
            /* .wk = */ matrix::load(in),
            /* .wv = */ matrix::load(in));
    }

    layer.wo = matrix::load(in);
    return layer;
}
