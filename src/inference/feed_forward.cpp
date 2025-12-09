#include "feed_forward.hpp"

#include <iostream>

#include "inference/network_node.hpp"
#include "training/optimizer.hpp"

static matrix activate(const matrix& input) {
    constexpr static auto leaky_relu
        = [](const float f) { return f < 0 ? 0.01f * f : f; };

    return input.mapped(leaky_relu);
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

    matrix activation_input = input.cross_multiplied(w1);
    for (size_t i = 0; i < activation_input.rows; ++i) {
        activation_input.add_row_vector(i, b1);
    }

    matrix activation_output = activate(activation_input);

    matrix final_output = activation_output.cross_multiplied(w2);
    for (size_t i = 0; i < final_output.rows; ++i) {
        final_output.add_row_vector(i, b2);
    }

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

    matrix b2_gradient({ 1, post_layer_gradient.cols });
    for (size_t i = 0; i < b2_gradient.cols; ++i) {
        b2_gradient.set(0, i, post_layer_gradient.col_sum(i));
    }
    adjust_parameter_matrix(b2, b2_gradient, learning_rate);

    matrix w2_gradient
        = activation_output.t_cross_multiplied(post_layer_gradient);
    regularize_weight_gradient(w2_gradient, w2);
    adjust_parameter_matrix(w2, w2_gradient, learning_rate);

    const matrix a1_gradient = post_layer_gradient.cross_t_multiplied(w2);
    matrix z1_gradient({ a1_gradient.rows, a1_gradient.cols });

    for (size_t i = 0; i < z1_gradient.rows; i++) {
        for (size_t j = 0; j < z1_gradient.cols; j++) {
            const auto z1_value = activation_input.get(i, j);
            const auto self_value = a1_gradient.get(i, j);
            z1_gradient.set(i, j, self_value * (z1_value > 0 ? 1.0f : 0.01f));
        }
    }

    matrix b1_gradient({ 1, z1_gradient.cols });
    for (size_t i = 0; i < z1_gradient.cols; ++i) {
        b1_gradient.set(0, i, z1_gradient.col_sum(i));
    }
    adjust_parameter_matrix(b1, b1_gradient, learning_rate);

    matrix w1_gradient = layer_input.t_cross_multiplied(z1_gradient);

    regularize_weight_gradient(w1_gradient, w1, 0.01f);
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
    auto w1 = matrix::load(in);
    auto b1 = matrix::load(in);
    auto w2 = matrix::load(in);
    auto b2 = matrix::load(in);

    FeedForwardLayer layer(0, 0);  // 0, 0 to avoid unnecessary allocation
    layer.w1 = std::move(w1);
    layer.b1 = std::move(b1);
    layer.w2 = std::move(w2);
    layer.b2 = std::move(b2);

    return layer;
}
