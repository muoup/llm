#include "feed_forward.hpp"

#include <iostream>

#include <inference/network_node.hpp>
#include <kernels/feed_forward.hpp>
#include <kernels/matrix_kernels.hpp>
#include <kernels/optimizer.hpp>
#include <util/logger.hpp>

NodeType FeedForwardLayer::getType() const {
    return NodeType::FeedForward;
}

FeedForwardLayer::FeedForwardLayer(size_t dimensions, size_t projection_size)
    : w1(dimensions, projection_size),
      b1(1, projection_size),
      w2(projection_size, dimensions),
      b2(1, dimensions) {}

size_t FeedForwardLayer::parameterCount() const {
    return (w1.rows * w1.cols) + (b1.rows * b1.cols) + (w2.rows * w2.cols)
           + (b2.rows * b2.cols);
}

void FeedForwardLayer::randomize(const float min, const float max) {
    w1.leaky_kaiming_randomize();
    b1.leaky_kaiming_randomize();
    w2.leaky_kaiming_randomize();
    b2.leaky_kaiming_randomize();
}

ForwardingResult FeedForwardLayer::forward(std::span<const matrix> inputs,
                                           bool perf) const {
    const matrix& input = inputs[0];

    matrix activation_input = kernel::matrix::cross_multiplied(input, w1);
    kernel::feed_forward::add_bias(activation_input, b1);
    matrix activation_output
        = kernel::feed_forward::leaky_relu_activation(activation_input);
    matrix final_output = kernel::matrix::cross_multiplied(activation_output, w2);
    kernel::feed_forward::add_bias(final_output, b2);

    LOG_DEBUG("  FF Layer Forward:");
    LOG_DEBUG("    input norm: %f", input.norm());
    LOG_DEBUG("    activation_input norm: %f", activation_input.norm());
    LOG_DEBUG("    activation_output norm: %f", activation_output.norm());
    LOG_DEBUG("    final_output norm: %f", final_output.norm());
    
    kernel::wait_for_all_streams();
    return standardResult(matrix::construct_vec(final_output, activation_input,
                                                activation_output));
}

std::vector<matrix> FeedForwardLayer::backpropogate(
    const ForwardingResult& result,
    std::span<const matrix> inputs,
    std::span<const matrix> gradients,
    CentralOptimizer& optimizer,
    bool perf) {
    const matrix& layer_input = inputs[0];
    const matrix& activation_input = result.outputs[1];
    const matrix& activation_output = result.outputs[2];
    const matrix& post_layer_gradient = gradients[0];

    matrix b2_gradient
        = kernel::feed_forward::sum_columns(post_layer_gradient);
    matrix w2_gradient
        = kernel::matrix::t_cross_multiplied(activation_output, post_layer_gradient);
    const matrix a1_gradient = kernel::matrix::cross_t_multiplied(post_layer_gradient, w2);

    matrix z1_gradient = kernel::feed_forward::leaky_relu_activation_backprop(
        activation_input, a1_gradient);

    matrix b1_gradient = kernel::feed_forward::sum_columns(z1_gradient);
    matrix w1_gradient = kernel::matrix::t_cross_multiplied(layer_input, z1_gradient);
    auto input_gradient = kernel::matrix::cross_t_multiplied(z1_gradient, w1);

    LOG_DEBUG("  FF Layer Gradients:");
    LOG_DEBUG("    w1_gradient norm: %f", w1_gradient.norm());
    LOG_DEBUG("    b1_gradient norm: %f", b1_gradient.norm());
    LOG_DEBUG("    w2_gradient norm: %f", w2_gradient.norm());
    LOG_DEBUG("    b2_gradient norm: %f", b2_gradient.norm());

    optimizer.update(w1, w1_gradient);
    optimizer.update(w2, w2_gradient);
    optimizer.update(b1, b1_gradient);
    optimizer.update(b2, b2_gradient);
    kernel::optimizer::norm_clip(input_gradient);

    kernel::wait_for_all_streams();
    return matrix::construct_vec(input_gradient);
}

void FeedForwardLayer::save(std::ostream& out) const {
    w1.save(out);
    b1.save(out);
    w2.save(out);
    b2.save(out);
}

FeedForwardLayer FeedForwardLayer::load(std::istream& in) {
    CHECK_ERRORS("FeedForwardLayer Load - Start");

    FeedForwardLayer layer(0, 0);
    auto matrix = matrix::load(in);
    CHECK_ERRORS("FeedForwardLayer Load - w1 pre-move");
    layer.w1 = std::move(matrix);
    CHECK_ERRORS("FeedForwardLayer Load - w1");
    layer.b1 = matrix::load(in);
    CHECK_ERRORS("FeedForwardLayer Load - b1");
    layer.w2 = matrix::load(in);
    CHECK_ERRORS("FeedForwardLayer Load - w2");
    layer.b2 = matrix::load(in);
    CHECK_ERRORS("FeedForwardLayer Load - b2");

    return layer;
}
