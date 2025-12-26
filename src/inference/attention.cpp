#include "attention.hpp"

#include <cmath>
#include <span>
#include <vector>

#include <inference/network_node.hpp>
#include <kernels/matrix_kernels.hpp>
#include <kernels/optimizer.hpp>
#include <util/logger.hpp>

NodeType AttentionLayer::getType() const {
    return NodeType::Attention;
}

constexpr size_t STREAMS_PER_HEAD = 3;

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

    for (size_t i = 0; i < STREAMS_PER_HEAD * head_size; ++i) {
        streams.emplace_back(kernel::matrix::create_kernel_stream());
    }
}

AttentionLayer::AttentionLayer()
    : dimensions(0), head_size(0), head_count(0), masked(false), wo() {}

AttentionLayer::~AttentionLayer() {
    for (auto& stream : streams)
        kernel::matrix::destroy_kernel_stream(stream);
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
    // placeholder for the final output to prevent insert(0) additional
    // overhead
    returns.emplace_back();
    // placeholder for the concatenated heads
    returns.emplace_back(input.rows, head_count * head_size);

    for (size_t h = 0; h < head_count; ++h) {
        returns.emplace_back();  // q
        returns.emplace_back();  // k
        returns.emplace_back();  // v
        returns.emplace_back();  // scores
    }

#pragma omp parallel for
    for (size_t h = 0; h < head_count; ++h) {
        const AttentionHead& head = heads[h];
        const size_t stream_offset = h * STREAMS_PER_HEAD;

        matrix q = kernel::matrix::cross_multiplied(input, head.wq,
                                                    streams[stream_offset + 0]);
        matrix k = kernel::matrix::cross_multiplied(input, head.wk,
                                                    streams[stream_offset + 1]);
        matrix v = kernel::matrix::cross_multiplied(input, head.wv,
                                                    streams[stream_offset + 2]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 0],
                                            streams[stream_offset + 1],
                                            streams[stream_offset + 2]);

        matrix scores = kernel::matrix::cross_t_multiplied(
            q, k, streams[stream_offset + 0]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 0]);

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        kernel::matrix::scale(scores, scale, streams[stream_offset + 0]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 0]);

        if (masked) {
            kernel::matrix::mask_upper_triangle(
                scores, -std::numeric_limits<float>::infinity(),
                streams[stream_offset + 0]);
            kernel::optimizer::wait_for_streams(streams[stream_offset + 0]);
        }

        kernel::matrix::softmax(scores, streams[stream_offset + 0]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 0]);

        matrix weighted_sum = kernel::matrix::cross_multiplied(
            scores, v, streams[stream_offset + 0]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 0]);

        matrix& concatenated_heads = returns[1];
        kernel::matrix::set_horizontal_slice(concatenated_heads, h * head_size,
                                             weighted_sum,
                                             streams[stream_offset + 0]);

#pragma omp critical
        {
            LOG_DEBUG("  Attention Head %zu Forward:", h);
            LOG_DEBUG("    wq norm: %f", head.wq.norm());
            LOG_DEBUG("    wk norm: %f", head.wk.norm());
            LOG_DEBUG("    wv norm: %f", head.wv.norm());
            LOG_DEBUG("    q norm: %f", q.norm());
            LOG_DEBUG("    k norm: %f", k.norm());
            LOG_DEBUG("    v norm: %f", v.norm());
            LOG_DEBUG("    scores norm: %f", scores.norm());
            LOG_DEBUG("    weighted_sum norm: %f", weighted_sum.norm());
        }

        returns[2 + h * 4 + 0] = std::move(q);
        returns[2 + h * 4 + 1] = std::move(k);
        returns[2 + h * 4 + 2] = std::move(v);
        returns[2 + h * 4 + 3] = std::move(scores);
    }

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

    matrix wo_gradient = kernel::matrix::t_cross_multiplied(
        concat_heads, post_layer_gradient, streams[0]);
    matrix attention_concat_gradient = kernel::matrix::cross_t_multiplied(
        post_layer_gradient, wo, streams[1]);
    matrix input_gradient(layer_input.rows, layer_input.cols);
    kernel::optimizer::wait_for_streams(streams[0], streams[1]);
    
    kernel::optimizer::adjust_regularize_parameter_matrix(wo_gradient, wo,
                                                          learning_rate);
    
    
    std::vector<matrix> head_input_gradients(head_count);

#pragma omp parallel for
    for (size_t h = 0; h < head_count; ++h) {
        const matrix& q = result.outputs[2 + h * 4 + 0];
        const matrix& k = result.outputs[2 + h * 4 + 1];
        const matrix& v = result.outputs[2 + h * 4 + 2];
        const matrix& scores = result.outputs[2 + h * 4 + 3];

        size_t stream_offset = h * STREAMS_PER_HEAD;
        const auto attention_gradient
            = attention_concat_gradient.get_horizontal_slice(h * head_size,
                                                             head_size);

        matrix v_gradient = kernel::matrix::t_cross_multiplied(
            scores, attention_gradient, streams[stream_offset + 0]);
        matrix scores_gradient = kernel::matrix::cross_t_multiplied(
            attention_gradient, v, streams[stream_offset + 1]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 1]);

        matrix raw_scores_gradient = kernel::matrix::backprop_softmax(
            scores, scores_gradient, streams[stream_offset + 1]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 1]);

        const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
        kernel::matrix::scale(raw_scores_gradient, scale,
                              streams[stream_offset + 1]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 0]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 1]);

        matrix q_gradient = kernel::matrix::cross_multiplied(
            raw_scores_gradient, k, streams[stream_offset + 0]);
        matrix k_gradient = kernel::matrix::t_cross_multiplied(
            raw_scores_gradient, q, streams[stream_offset + 1]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 0],
                                            streams[stream_offset + 1]);

        matrix wq_gradient = kernel::matrix::t_cross_multiplied(
            layer_input, q_gradient, streams[stream_offset + 0]);
        matrix wk_gradient = kernel::matrix::t_cross_multiplied(
            layer_input, k_gradient, streams[stream_offset + 1]);
        matrix wv_gradient = kernel::matrix::t_cross_multiplied(
            layer_input, v_gradient, streams[stream_offset + 2]);

        kernel::optimizer::norm_clip(wq_gradient, streams[stream_offset + 0]);
        kernel::optimizer::norm_clip(wk_gradient, streams[stream_offset + 1]);
        kernel::optimizer::norm_clip(wv_gradient, streams[stream_offset + 2]);
        kernel::optimizer::wait_for_streams(streams[stream_offset + 0],
                                            streams[stream_offset + 1],
                                            streams[stream_offset + 2]);

#pragma omp critical
        {
            LOG_DEBUG("  Attention Head %zu Gradients:", h);
            LOG_DEBUG("    scores_gradient norm: %f", scores_gradient.norm());
            LOG_DEBUG("    raw_scores_gradient norm: %.10e",
                      raw_scores_gradient.norm());
            LOG_DEBUG("    wq_gradient norm: %f", wq_gradient.norm());
            LOG_DEBUG("    wk_gradient norm: %f", wk_gradient.norm());
            LOG_DEBUG("    wv_gradient norm: %f", wv_gradient.norm());
        }

        AttentionHead& head = heads[h];
        kernel::optimizer::adjust_regularize_parameter_matrix(
            wq_gradient, head.wq, learning_rate, streams[stream_offset + 0]);
        kernel::optimizer::adjust_regularize_parameter_matrix(
            wk_gradient, head.wk, learning_rate, streams[stream_offset + 1]);
        kernel::optimizer::adjust_regularize_parameter_matrix(
            wv_gradient, head.wv, learning_rate, streams[stream_offset + 2]);

        matrix head_input_gradient = kernel::matrix::cross_t_multiplied(
            q_gradient, head.wq, streams[stream_offset + 0]);
        matrix input_k = kernel::matrix::cross_t_multiplied(
            k_gradient, head.wk, streams[stream_offset + 1]);
        matrix input_v = kernel::matrix::cross_t_multiplied(
            v_gradient, head.wv, streams[stream_offset + 2]);
        kernel::optimizer::wait_for_stream(streams[stream_offset + 0]);

        kernel::matrix::add(head_input_gradient, input_k,
                            streams[stream_offset + 1]);
        kernel::matrix::add(head_input_gradient, input_v,
                            streams[stream_offset + 2]);

        kernel::optimizer::wait_for_streams(streams[stream_offset + 1],
                                            streams[stream_offset + 2]);

        head_input_gradients[h] = std::move(head_input_gradient);
    }

    for (const auto& grad : head_input_gradients) {
        kernel::matrix::add(input_gradient, grad);
    }

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

    for (size_t i = 0; i < STREAMS_PER_HEAD * layer.head_count; ++i) {
        layer.streams.emplace_back(kernel::matrix::create_kernel_stream());
    }

    return layer;
}
