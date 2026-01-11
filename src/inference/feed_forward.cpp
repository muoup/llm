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

FeedForwardLayer::FeedForwardLayer(size_t dimensions,
                                   size_t projection_size,
                                   ActivationFunction activation)
    : w1(dimensions, projection_size),
      b1(1, projection_size),
      w2(projection_size, dimensions),
      b2(1, dimensions),
      w3(),
      b3(),
      activation(activation) {
    if (activation == ActivationFunction::SwiGLU) {
        w3 = matrix(projection_size, dimensions);
        b3 = matrix(1, dimensions);
    }
}

size_t FeedForwardLayer::parameterCount() const {
    return (w1.rows * w1.cols) + (b1.rows * b1.cols) + (w2.rows * w2.cols)
           + (b2.rows * b2.cols);
}

void FeedForwardLayer::randomize(const float min, const float max) {
    w1.leaky_kaiming_randomize();
    b1.leaky_kaiming_randomize();
    w2.leaky_kaiming_randomize();
    b2.leaky_kaiming_randomize();

    if (activation == ActivationFunction::SwiGLU) {
        w3.leaky_kaiming_randomize();
        b3.leaky_kaiming_randomize();
    }
}

ForwardingResult FeedForwardLayer::forward(std::span<const matrix> inputs,
                                           bool perf) const {
    const matrix& input = inputs[0];

    if (activation == ActivationFunction::SwiGLU) {
        matrix gate_input = kernel::matrix::cross_multiplied(input, w1);
        kernel::feed_forward::add_bias(gate_input, b1);
        matrix gate_output = kernel::feed_forward::silu_activation(gate_input);

        matrix value_input = kernel::matrix::cross_multiplied(input, w2);
        kernel::feed_forward::add_bias(value_input, b2);

        kernel::matrix::element_wise_multiply(gate_output, value_input);

        matrix final_output = kernel::matrix::cross_multiplied(gate_output, w3);
        kernel::feed_forward::add_bias(final_output, b3);

        LOG_DEBUG("  FF Layer Forward (SwiGLU):");
        LOG_DEBUG("    input norm: %f", input.norm());
        LOG_DEBUG("    gate_input norm: %f", gate_input.norm());
        LOG_DEBUG("    value_input norm: %f", value_input.norm());
        LOG_DEBUG("    gate_output norm: %f", gate_output.norm());
        LOG_DEBUG("    final_output norm: %f", final_output.norm());

        kernel::wait_for_all_streams();
        return standardResult(matrix::construct_vec(final_output, gate_input,
                                                    value_input, gate_output));
    }

    matrix activation_input = kernel::matrix::cross_multiplied(input, w1);
    kernel::feed_forward::add_bias(activation_input, b1);

    matrix activation_output;
    if (activation == ActivationFunction::GeLU) {
        activation_output
            = kernel::feed_forward::gelu_activation(activation_input);
    } else {
        activation_output
            = kernel::feed_forward::leaky_relu_activation(activation_input);
    }

    matrix final_output
        = kernel::matrix::cross_multiplied(activation_output, w2);
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
    const matrix& post_layer_gradient = gradients[0];

    if (activation == ActivationFunction::SwiGLU) {
        const matrix& gate_input = result.outputs[1];
        const matrix& value_input = result.outputs[2];
        const matrix& gate_output = result.outputs[3];

        matrix b3_gradient
            = kernel::feed_forward::sum_columns(post_layer_gradient);
        matrix w3_gradient = kernel::matrix::t_cross_multiplied(
            gate_output, post_layer_gradient);
        matrix gate_value_gradient
            = kernel::matrix::cross_t_multiplied(post_layer_gradient, w3);

        matrix gate_gradient = kernel::matrix::clone(gate_value_gradient);
        kernel::matrix::element_wise_multiply(gate_gradient, value_input);

        matrix value_gradient = kernel::matrix::clone(gate_value_gradient);
        kernel::matrix::element_wise_multiply(value_gradient, gate_output);

        matrix b1_gradient = kernel::feed_forward::sum_columns(gate_gradient);
        matrix w1_gradient
            = kernel::matrix::t_cross_multiplied(layer_input, gate_gradient);
        matrix gate_input_gradient
            = kernel::matrix::cross_t_multiplied(gate_gradient, w1);

        matrix b2_gradient = kernel::feed_forward::sum_columns(value_gradient);
        matrix w2_gradient
            = kernel::matrix::t_cross_multiplied(layer_input, value_gradient);
        matrix value_input_gradient
            = kernel::matrix::cross_t_multiplied(value_gradient, w2);

        matrix gate_output_gradient
            = kernel::feed_forward::silu_activation_backprop(
                gate_input, gate_input_gradient);

        matrix input_gradient = kernel::matrix::clone(gate_output_gradient);
        input_gradient.add(value_input_gradient);

        LOG_DEBUG("  FF Layer Gradients (SwiGLU):");
        LOG_DEBUG("    w1_gradient norm: %f", w1_gradient.norm());
        LOG_DEBUG("    b1_gradient norm: %f", b1_gradient.norm());
        LOG_DEBUG("    w2_gradient norm: %f", w2_gradient.norm());
        LOG_DEBUG("    b2_gradient norm: %f", b2_gradient.norm());
        LOG_DEBUG("    w3_gradient norm: %f", w3_gradient.norm());
        LOG_DEBUG("    b3_gradient norm: %f", b3_gradient.norm());

        optimizer.update(w1, w1_gradient);
        optimizer.update(w2, w2_gradient);
        optimizer.update(w3, w3_gradient);
        optimizer.update(b1, b1_gradient);
        optimizer.update(b2, b2_gradient);
        optimizer.update(b3, b3_gradient);
        kernel::optimizer::norm_clip(input_gradient);

        kernel::wait_for_all_streams();
        return matrix::construct_vec(input_gradient);
    }

    const matrix& activation_input = result.outputs[1];
    const matrix& activation_output = result.outputs[2];

    matrix b2_gradient = kernel::feed_forward::sum_columns(post_layer_gradient);
    matrix w2_gradient = kernel::matrix::t_cross_multiplied(
        activation_output, post_layer_gradient);
    const matrix a1_gradient
        = kernel::matrix::cross_t_multiplied(post_layer_gradient, w2);

    matrix z1_gradient;
    if (activation == ActivationFunction::GeLU) {
        z1_gradient = kernel::feed_forward::gelu_activation_backprop(
            activation_input, a1_gradient);
    } else {
        z1_gradient = kernel::feed_forward::leaky_relu_activation_backprop(
            activation_input, a1_gradient);
    }

    matrix b1_gradient = kernel::feed_forward::sum_columns(z1_gradient);
    matrix w1_gradient
        = kernel::matrix::t_cross_multiplied(layer_input, z1_gradient);
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
    out.write(reinterpret_cast<const char*>(&activation), sizeof(activation));
    w1.save(out);
    b1.save(out);
    w2.save(out);
    b2.save(out);

    if (activation == ActivationFunction::SwiGLU) {
        w3.save(out);
        b3.save(out);
    }
}

FeedForwardLayer FeedForwardLayer::load(std::istream& in) {
    CHECK_ERRORS("FeedForwardLayer Load - Start");

    ActivationFunction activation_type;
    in.read(reinterpret_cast<char*>(&activation_type), sizeof(activation_type));

    size_t dimensions = 0;
    size_t projection_size = 0;

    auto w1_mat = matrix::load(in);
    dimensions = w1_mat.rows;
    projection_size = w1_mat.cols;

    FeedForwardLayer layer(dimensions, projection_size, activation_type);
    layer.w1 = std::move(w1_mat);

    CHECK_ERRORS("FeedForwardLayer Load - w1");
    layer.b1 = matrix::load(in);
    CHECK_ERRORS("FeedForwardLayer Load - b1");
    layer.w2 = matrix::load(in);
    CHECK_ERRORS("FeedForwardLayer Load - w2");
    layer.b2 = matrix::load(in);
    CHECK_ERRORS("FeedForwardLayer Load - b2");

    if (activation_type == ActivationFunction::SwiGLU) {
        layer.w3 = matrix::load(in);
        CHECK_ERRORS("FeedForwardLayer Load - w3");
        layer.b3 = matrix::load(in);
        CHECK_ERRORS("FeedForwardLayer Load - b3");
    }

    return layer;
}
