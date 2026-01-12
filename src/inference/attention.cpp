#include "attention.hpp"

#include <cmath>
#include <span>
#include <vector>

#include <inference/network_node.hpp>
#include <kernels/optimizer.hpp>
#include <kernels/pools.hpp>
#include <kernels/matrix/host.hpp>
#include <kernels/matrix/cublas.hpp>
#include <util/logger.hpp>

constexpr size_t STREAMS_PER_HEAD = 0;

NodeType AttentionLayer::getType() const {
    return NodeType::Attention;
}

AttentionLayer::AttentionLayer(size_t dimensions,
                               size_t head_count,
                               bool masked,
                               DataType dtype)
    : dimensions(dimensions),
      head_size(dimensions / head_count),
      head_count(head_count),
      masked(masked),
      wo(matrix(head_size * head_count, dimensions, dtype)),
      streams(STREAMS_PER_HEAD * head_count) {
    for (size_t i = 0; i < head_count; ++i) {
        heads.emplace_back(AttentionHead{
            matrix(dimensions, head_size, dtype),  // wq
            matrix(dimensions, head_size, dtype),  // wk
            matrix(dimensions, head_size, dtype)   // wv
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
    returns.reserve(2 + head_count * 4);

    const kernel::kernel_stream_t global_stream = nullptr;

    // placeholder for the final output to prevent insert(0) additional overhead
    returns.emplace_back();
    // placeholder for the concatenated heads
    returns.emplace_back(kernel::matrix::async_allocate(
        input.rows, head_count * head_size, input.type));

    for (size_t h = 0; h < head_count; ++h) {
        returns.emplace_back();  // q
        returns.emplace_back();  // k
        returns.emplace_back();  // v
        returns.emplace_back();  // scores
    }

    for (size_t h = 0; h < head_count; ++h) {
        const AttentionHead& head = heads[h];
        const kernel::kernel_stream_t head_stream = nullptr;

        matrix q
            = kernel::matrix::cross_multiplied(input, head.wq, head_stream);
        matrix k
            = kernel::matrix::cross_multiplied(input, head.wk, head_stream);
        matrix v
            = kernel::matrix::cross_multiplied(input, head.wv, head_stream);

        matrix scores = kernel::matrix::cross_t_multiplied(q, k, head_stream);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

        kernel::matrix::scale(scores, scale, head_stream);

        if (masked) {
            kernel::matrix::mask_upper_triangle(
                scores, -std::numeric_limits<float>::infinity(), head_stream);
        }

        kernel::matrix::softmax(scores, head_stream);

        matrix weighted_sum
            = kernel::matrix::cross_multiplied(scores, v, head_stream);
        matrix& concatenated_heads = returns[1];
        kernel::matrix::set_horizontal_slice(concatenated_heads, h * head_size,
                                             weighted_sum, head_stream);

        LOG_DEBUG("  Attention Head %zu Forward:", h);
        LOG_DEBUG("    wq norm: %f", head.wq.norm());
        LOG_DEBUG("    wk norm: %f", head.wk.norm());
        LOG_DEBUG("    wv norm: %f", head.wv.norm());
        LOG_DEBUG("    q norm: %f", q.norm());
        LOG_DEBUG("    k norm: %f", k.norm());
        LOG_DEBUG("    v norm: %f", v.norm());
        LOG_DEBUG("    scores norm: %f", scores.norm());
        LOG_DEBUG("    weighted_sum norm: %f", weighted_sum.norm());

        returns[2 + h * 4 + 0] = std::move(q);
        returns[2 + h * 4 + 1] = std::move(k);
        returns[2 + h * 4 + 2] = std::move(v);
        returns[2 + h * 4 + 3] = std::move(scores);
    }

    matrix& final_output = returns[0];
    matrix& concatenated_heads = returns[1];

    final_output = kernel::matrix::cross_multiplied(concatenated_heads, wo,
                                                    global_stream);

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

    kernel::wait_for_all_streams();
    return standardResult(std::move(returns));
}

std::vector<matrix> AttentionLayer::backpropogate(
    const ForwardingResult& result,
    std::span<const matrix> inputs,
    std::span<const matrix> gradients,
    CentralOptimizer& optimizer,
    bool perf) {
    constexpr float regularization_strength = 0.01f;

    const matrix& layer_input = inputs[0];
    const matrix& concat_heads = result.outputs[1];
    const matrix& post_layer_gradient = gradients[0];

    LOG_DEBUG("  Attention Layer Backpropagation:");
    LOG_DEBUG("    post_layer_gradient norm: %f", post_layer_gradient.norm());

    const kernel::kernel_stream_t global_stream = nullptr;

    matrix wo_gradient = kernel::matrix::t_cross_multiplied(
        concat_heads, post_layer_gradient, global_stream);
    matrix attention_concat_gradient = kernel::matrix::cross_t_multiplied(
        post_layer_gradient, wo, global_stream);

    matrix input_gradient = kernel::matrix::async_allocate(
        layer_input.rows, layer_input.cols, layer_input.type);
    matrix raw_scores_gradient = kernel::matrix::async_allocate(
        layer_input.rows, layer_input.rows, layer_input.type);

    for (size_t h = 0; h < head_count; ++h) {
        const matrix& q = result.outputs[2 + h * 4 + 0];
        const matrix& k = result.outputs[2 + h * 4 + 1];
        const matrix& v = result.outputs[2 + h * 4 + 2];
        const matrix& scores = result.outputs[2 + h * 4 + 3];

        const kernel::kernel_stream_t head_stream = nullptr;

        const auto attention_gradient
            = attention_concat_gradient.get_horizontal_slice(h * head_size,
                                                             head_size);

        matrix v_gradient = kernel::matrix::t_cross_multiplied(
            scores, attention_gradient, head_stream);
        matrix scores_gradient = kernel::matrix::cross_t_multiplied(
            attention_gradient, v, head_stream);

        kernel::matrix::backprop_softmax(raw_scores_gradient, scores,
                                         scores_gradient, head_stream);

        const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
        kernel::matrix::scale(raw_scores_gradient, scale, head_stream);

        matrix q_gradient = kernel::matrix::cross_multiplied(
            raw_scores_gradient, k, head_stream);
        matrix k_gradient = kernel::matrix::t_cross_multiplied(
            raw_scores_gradient, q, head_stream);

        matrix wq_gradient = kernel::matrix::t_cross_multiplied(
            layer_input, q_gradient, head_stream);
        matrix wk_gradient = kernel::matrix::t_cross_multiplied(
            layer_input, k_gradient, head_stream);
        matrix wv_gradient = kernel::matrix::t_cross_multiplied(
            layer_input, v_gradient, head_stream);

        LOG_DEBUG("  Attention Head %zu Gradients:", h);
        LOG_DEBUG("    scores_gradient norm: %f", scores_gradient.norm());
        LOG_DEBUG("    raw_scores_gradient norm: %.10e",
                  raw_scores_gradient.norm());
        LOG_DEBUG("    wq_gradient norm: %f", wq_gradient.norm());
        LOG_DEBUG("    wk_gradient norm: %f", wk_gradient.norm());
        LOG_DEBUG("    wv_gradient norm: %f", wv_gradient.norm());

        AttentionHead& head = heads[h];

        matrix q_input_gradient = kernel::matrix::cross_t_multiplied(
            q_gradient, head.wq, head_stream);
        matrix k_input_gradient = kernel::matrix::cross_t_multiplied(
            k_gradient, head.wk, head_stream);
        matrix v_input_gradient = kernel::matrix::cross_t_multiplied(
            v_gradient, head.wv, head_stream);

        kernel::matrix::add(input_gradient, q_input_gradient, head_stream);
        kernel::matrix::add(input_gradient, k_input_gradient, head_stream);
        kernel::matrix::add(input_gradient, v_input_gradient, head_stream);

        // Wait for streams before update, or update handles streams?
        // CentralOptimizer doesn't expose streams yet. AdamW kernel does.
        // Assuming default stream for now for safety as CentralOptimizer update
        // is simple.
        kernel::wait_for_all_streams();

        optimizer.update(head.wq, wq_gradient);
        optimizer.update(head.wk, wk_gradient);
        optimizer.update(head.wv, wv_gradient);
    }

    kernel::wait_for_all_streams();
    optimizer.update(wo, wo_gradient);

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
    layer.streams = kernel::FixedStreamList(STREAMS_PER_HEAD * head_count);

    return layer;
}
