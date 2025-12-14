#include "feed_forward.hpp"

#include <iostream>
#include <cmath>

#include <inference/network_node.hpp>
#include <kernels/feed_forward.hpp>
#include <kernels/matrix_kernels.hpp>
#include <kernels/optimizer.hpp>

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
    w1.randomize(min, max);
    b1.randomize(min, max);
    w2.randomize(min, max);
    b2.randomize(min, max);
}

ForwardingResult FeedForwardLayer::forward(
    std::span<const matrix> inputs) const {
    const matrix& input = inputs[0];
    
    matrix activation_input = input.cross_multiplied(w1);
    kernel::optimizer::wait_for_operations();
    
    kernel::feed_forward::add_bias(activation_input, b1);
    kernel::optimizer::wait_for_operations();

    matrix activation_output
        = kernel::feed_forward::leaky_relu_activation(activation_input);
    kernel::optimizer::wait_for_operations();

    matrix final_output = activation_output.cross_multiplied(w2);
    kernel::optimizer::wait_for_operations();
    
    kernel::feed_forward::add_bias(final_output, b2);
    kernel::optimizer::wait_for_operations();

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
    matrix activation_output = result.outputs[2].clone();
    const matrix& post_layer_gradient = gradients[0];
    
    kernel::optimizer::wait_for_operations();

    matrix b2_gradient = kernel::feed_forward::sum_columns(post_layer_gradient);
    matrix w2_gradient
        = activation_output.t_cross_multiplied(post_layer_gradient);
    const matrix a1_gradient = post_layer_gradient.cross_t_multiplied(w2);
    kernel::optimizer::wait_for_operations();
    
    matrix z1_gradient = kernel::feed_forward::leaky_relu_activation_backprop(
        activation_input, a1_gradient);
    kernel::optimizer::wait_for_operations();
    
    matrix b1_gradient = kernel::feed_forward::sum_columns(z1_gradient);
    matrix w1_gradient = layer_input.t_cross_multiplied(z1_gradient);
    auto input_gradient = z1_gradient.cross_t_multiplied(w1);

    kernel::optimizer::regularize_weight_gradient(w2_gradient, w2);
    kernel::optimizer::regularize_weight_gradient(w1_gradient, w1);
    kernel::optimizer::wait_for_operations();
    
    // std::cout << "  FF Layer Gradients:\n";
    // std::cout << "    w1_gradient norm: " << std::sqrt(w1_gradient.sum_of_squares()) << "\n";
    // std::cout << "    b1_gradient norm: " << std::sqrt(b1_gradient.sum_of_squares()) << "\n";
    // std::cout << "    w2_gradient norm: " << std::sqrt(w2_gradient.sum_of_squares()) << "\n";
    // std::cout << "    b2_gradient norm: " << std::sqrt(b2_gradient.sum_of_squares()) << "\n";

    kernel::optimizer::adjust_parameter_matrix(b2, b2_gradient, learning_rate);
    kernel::optimizer::adjust_parameter_matrix(w2, w2_gradient, learning_rate);
    kernel::optimizer::adjust_parameter_matrix(b1, b1_gradient, learning_rate);
    kernel::optimizer::adjust_parameter_matrix(w1, w1_gradient, learning_rate);

    kernel::optimizer::norm_clip(input_gradient);
    kernel::optimizer::wait_for_operations();
    
    return matrix::construct_vec(input_gradient);
}

void FeedForwardLayer::save(std::ostream& out) const {
    w1.save(out);
    b1.save(out);
    w2.save(out);
    b2.save(out);
}

FeedForwardLayer FeedForwardLayer::load(std::istream& in) {
    kernel::matrix::check_errors("FeedForwardLayer Load - Start");

    FeedForwardLayer layer(0, 0);
    auto matrix = matrix::load(in);
    kernel::matrix::check_errors("FeedForwardLayer Load - w1 pre-move");
    layer.w1 = std::move(matrix);
    kernel::matrix::check_errors("FeedForwardLayer Load - w1");
    layer.b1 = matrix::load(in);
    kernel::matrix::check_errors("FeedForwardLayer Load - b1");
    layer.w2 = matrix::load(in);
    kernel::matrix::check_errors("FeedForwardLayer Load - w2");
    layer.b2 = matrix::load(in);
    kernel::matrix::check_errors("FeedForwardLayer Load - b2");

    return layer;
}
