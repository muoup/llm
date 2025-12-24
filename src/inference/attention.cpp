#include "attention.hpp"

#include <cmath>
#include <span>
#include <vector>

#include <inference/network_node.hpp>
#include <kernels/optimizer.hpp>
#include <util/logger.hpp>
#include "kernels/matrix_kernels.hpp"

NodeType AttentionLayer::getType() const {
    return NodeType::Attention;
}

AttentionLayer::AttentionLayer(size_t dimensions,
                               size_t head_count,
                               bool masked)
    : dimensions(dimensions),
      head_size(dimensions / head_count),
      head_count(head_count),
      masked(masked),
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
        head.wq.leaky_kaiming_randomize();
        head.wk.leaky_kaiming_randomize();
        head.wv.leaky_kaiming_randomize();
    }

    wo.leaky_kaiming_randomize();
}

size_t AttentionLayer::parameterCount() const {
    size_t count = wo.size();
    for (const auto& head : heads) {
        count += head.wq.size();
        count += head.wk.size();
        count += head.wv.size();
    }
    return count;
}

ForwardingResult AttentionLayer::forward(std::span<const matrix> inputs,
                                         bool perf) const {
    const matrix& input = inputs[0];
    const size_t seq_len = input.rows;

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

        matrix scores = q.cross_t_multiplied(k);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        kernel::optimizer::wait_for_operations();

        scores.scale(scale);
        kernel::optimizer::wait_for_operations();

        if (masked) {
            scores.mask_upper_triangular();
            kernel::optimizer::wait_for_operations();
        }

        scores.softmax();
        kernel::optimizer::wait_for_operations();

        matrix weighted_sum = scores.cross_multiplied(v);
        kernel::optimizer::wait_for_operations();

        matrix& concatenated_heads = returns[1];
        concatenated_heads.set_horizontal_slice(h * head_size, weighted_sum);

        LOG_DEBUG("  Attention Head %zu Forward:", h);
        LOG_DEBUG("    wq norm: %f", head.wq.norm());
        LOG_DEBUG("    wk norm: %f", head.wk.norm());
        LOG_DEBUG("    wv norm: %f", head.wv.norm());
        LOG_DEBUG("    q norm: %f", q.norm());
        LOG_DEBUG("    k norm: %f", k.norm());
        LOG_DEBUG("    v norm: %f", v.norm());
        LOG_DEBUG("    scores norm: %f", scores.norm());
        LOG_DEBUG("    weighted_sum norm: %f", weighted_sum.norm());

        returns.emplace_back(std::move(q));
        returns.emplace_back(std::move(k));
        returns.emplace_back(std::move(v));
        returns.emplace_back(std::move(scores));
    }

    kernel::optimizer::wait_for_operations();

    matrix& final_output = returns[0];
    matrix& concatenated_heads = returns[1];

    final_output = concatenated_heads.cross_multiplied(wo);

    LOG_DEBUG("  Attention Layer Forward:");
    LOG_DEBUG("    input norm: %f", input.norm());
    LOG_DEBUG("    concatenated_heads norm: %f", concatenated_heads.norm());
    LOG_DEBUG("    final_output norm: %f", final_output.norm());

    // Expected returns layout:
    // [0] -> final output
    // [1] -> concatenated heads
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
    float learning_rate,
    bool perf) {
    constexpr float regularization_strength = 0.01f;

    const matrix& layer_input = inputs[0];
    const matrix& concat_heads = result.outputs[1];
    const matrix& post_layer_gradient = gradients[0];

    LOG_DEBUG("  Attention Layer Backpropagation:");
    LOG_DEBUG("    post_layer_gradient norm: %f", post_layer_gradient.norm());

    matrix wo_gradient = concat_heads.t_cross_multiplied(post_layer_gradient);
    matrix attention_concat_gradient
        = post_layer_gradient.cross_t_multiplied(wo);

    matrix input_gradient(layer_input.rows, layer_input.cols);

    for (size_t h = 0; h < head_count; ++h) {
        const matrix& q = result.outputs[2 + h * 4 + 0];
        const matrix& k = result.outputs[2 + h * 4 + 1];
        const matrix& v = result.outputs[2 + h * 4 + 2];
        const matrix& scores = result.outputs[2 + h * 4 + 3];

        const auto attention_gradient
            = attention_concat_gradient.get_horizontal_slice(h * head_size,
                                                             head_size);

        matrix v_gradient = kernel::matrix::t_cross_multiplied(scores, attention_gradient);
        matrix scores_gradient = kernel::matrix::cross_t_multiplied(attention_gradient, v);
        
        matrix raw_scores_gradient = scores.backprop_softmax(scores_gradient);
        
        const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
        raw_scores_gradient.scale(scale);
        
        matrix q_gradient = raw_scores_gradient.cross_multiplied(k);
        matrix k_gradient = raw_scores_gradient.t_cross_multiplied(q);
        
        matrix wq_gradient = layer_input.t_cross_multiplied(q_gradient);
        matrix wk_gradient = layer_input.t_cross_multiplied(k_gradient);
        matrix wv_gradient = layer_input.t_cross_multiplied(v_gradient);

        LOG_DEBUG("  Attention Head %zu Gradients:", h);
        LOG_DEBUG("    scores_gradient norm: %f", scores_gradient.norm());
        LOG_DEBUG("    raw_scores_gradient norm: %.10e",
                  raw_scores_gradient.norm());
        LOG_DEBUG("    wq_gradient norm: %f", wq_gradient.norm());
        LOG_DEBUG("    wk_gradient norm: %f", wk_gradient.norm());
        LOG_DEBUG("    wv_gradient norm: %f", wv_gradient.norm());

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

    kernel::optimizer::regularize_weight_gradient(wo_gradient, wo);
    kernel::optimizer::adjust_parameter_matrix(wo, wo_gradient, learning_rate);

    return matrix::construct_vec(input_gradient);
}

void AttentionLayer::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&head_size), sizeof(head_size));
    out.write(reinterpret_cast<const char*>(&head_count), sizeof(head_count));
    out.write(reinterpret_cast<const char*>(&masked), sizeof(masked));

    for (const auto& head : heads) {
        head.wq.save(out);
        head.wk.save(out);
        head.wv.save(out);
    }

    wo.save(out);
}

AttentionLayer AttentionLayer::load(std::istream& in) {
    size_t dimensions, head_size, head_count;
    bool masked;

    in.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    in.read(reinterpret_cast<char*>(&head_size), sizeof(head_size));
    in.read(reinterpret_cast<char*>(&head_count), sizeof(head_count));
    in.read(reinterpret_cast<char*>(&masked), sizeof(masked));

    AttentionLayer layer = AttentionLayer();
    layer.dimensions = dimensions;
    layer.head_size = head_size;
    layer.head_count = head_count;
    layer.masked = masked;

    for (size_t i = 0; i < head_count; ++i) {
        layer.heads.emplace_back(
            /* .wq = */ matrix::load(in),
            /* .wk = */ matrix::load(in),
            /* .wv = */ matrix::load(in));
    }

    layer.wo = matrix::load(in);
    return layer;
}
