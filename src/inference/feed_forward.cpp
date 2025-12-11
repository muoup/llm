#include "feed_forward.hpp"

#include <iostream>

#include <inference/network_node.hpp>
#include <kernels/feed_forward.hpp>
#include <kernels/optimizer.hpp>

static matrix activate(const matrix& input) {
    constexpr static auto leaky_relu
        = [](const float f) { return f < 0 ? 0.01f * f : f; };

    return input.mapped(leaky_relu);
}

static matrix activate_derivative(const matrix& input) {
    constexpr static auto leaky_relu_derivative
        = [](const float f) { return f < 0 ? 0.01f : 1.0f; };

    return input.mapped(leaky_relu_derivative);
}

NodeType FeedForwardLayer::getType() const {
    return NodeType::FeedForward;
}

FeedForwardLayer::FeedForwardLayer(size_t dimensions, size_t projection_size)
    : w1({ dimensions, projection_size }),
      b1({ 1, projection_size }),
      w2({ projection_size, dimensions }),
      b2({ 1, dimensions }) {}

size_t FeedForwardLayer::parameterCount() const {
    return (w1.rows * w1.cols) + (b1.rows * b1.cols) + (w2.rows * w2.cols)
           + (b2.rows * b2.cols);
}

void FeedForwardLayer::randomize(const float min, const float max) {
    w1.randomize(min, max);
    b1.randomize(min, max);
    w2.randomize(min, max);
    b2.randomize(min, max);
}

ForwardingResult FeedForwardLayer::forward(
    std::span<const matrix> inputs) const {
    const matrix& input = inputs[0];

    std::cout << "Input Max/Min: " << input.max() << " " << input.min()
              << std::endl;
    std::cout << "Input ABSMAX: " << input.absmax() << std::endl;
    matrix activation_input = input.cross_multiplied(w1);
    kernel::feed_forward::add_bias(activation_input, b1);

    std::cout << "Activation Input Max/Min: " << activation_input.max() << " "
              << activation_input.min() << std::endl;
    std::cout << "Activation Input ABSMAX: " << activation_input.absmax()
              << std::endl;
    matrix activation_output = activate(activation_input);
    std::cout << "Activation Output Max/Min: " << activation_output.max() << " "
              << activation_output.min() << std::endl;

    matrix final_output = activation_output.cross_multiplied(w2);
    kernel::feed_forward::add_bias(final_output, b2);

    return standardResult(matrix::construct_vec(final_output, activation_input,
                                                activation_output));
}

std::vector<matrix> FeedForwardLayer::backpropogate(
    const ForwardingResult& result,
    std::span<const matrix> inputs,
    std::span<const matrix> gradients,
    float learning_rate) {
    const matrix& layer_input = inputs[0];
    const matrix& activation_input = result.outputs[1];
    const matrix& activation_output = result.outputs[2];
    const matrix& post_layer_gradient = gradients[0];

    matrix b2_gradient = kernel::feed_forward::sum_columns(post_layer_gradient);
    adjust_parameter_matrix(b2, b2_gradient, learning_rate);

    matrix w2_gradient
        = activation_output.t_cross_multiplied(post_layer_gradient);
    adjust_parameter_matrix(w2, w2_gradient, learning_rate);

    const matrix a1_gradient = post_layer_gradient.cross_t_multiplied(w2);
    matrix z1_gradient = kernel::feed_forward::relu_activation_backprop(
        activation_input, a1_gradient);

    matrix b1_gradient = matrix({ 1, z1_gradient.cols });
    adjust_parameter_matrix(b1, b1_gradient, learning_rate);

    matrix w1_gradient = layer_input.t_cross_multiplied(z1_gradient);
    adjust_parameter_matrix(w1, w1_gradient, learning_rate);

    auto input_gradient = z1_gradient.cross_t_multiplied(w1);
    return matrix::construct_vec(input_gradient);
}

void FeedForwardLayer::save(std::ostream& out) const {
    w1.save(out);
    b1.save(out);
    w2.save(out);
    b2.save(out);
}

FeedForwardLayer FeedForwardLayer::load(std::istream& in) {
    FeedForwardLayer layer(0, 0);  // 0, 0 to avoid unnecessary allocation
    layer.w1 = matrix::load(in);
    layer.b1 = matrix::load(in);
    layer.w2 = matrix::load(in);
    layer.b2 = matrix::load(in);

    return layer;
}
