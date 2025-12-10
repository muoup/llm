#include "layer_normalize.hpp"

#include <training/optimizer.hpp>
#include <inference/inference.hpp>
#include <inference/network_node.hpp>

#include <cmath>

LayerNorm::LayerNorm(std::unique_ptr<INode> inner_node, size_t dimensions, float epsilon)
    : dimensions(dimensions),
      epsilon(epsilon),
      gamma(1, dimensions),
      beta(1, dimensions),
      inner_node(std::move(inner_node)) {}

size_t LayerNorm::parameterCount() const {
    return (gamma.rows * gamma.cols) + (beta.rows * beta.cols) + inner_node->parameterCount();
}

void LayerNorm::randomize(float min, float max) {
    gamma.randomize(min, max);
    beta.randomize(min, max);
}

NodeType LayerNorm::getType() const {
    return NodeType::LayerNorm;
}

ForwardingResult LayerNorm::forward(std::span<const matrix> inputs) const {
    const matrix& input = inputs[0];
    matrix normalized_input(input.rows, input.cols);
    matrix mean(input.rows, 1);
    matrix inv_variance(input.rows, 1);

    for (size_t i = 0; i < input.rows; ++i) {
        float row_mean = 0.0f;
        for (size_t j = 0; j < input.cols; ++j) {
            row_mean += input.get(i, j);
        }
        row_mean /= static_cast<float>(input.cols);
        mean.set(i, 0, row_mean);

        float variance = 0.0f;
        for (size_t j = 0; j < input.cols; ++j) {
            float diff = input.get(i, j) - row_mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(input.cols);
        inv_variance.set(i, 0, 1.0f / std::sqrt(variance + epsilon));

        for (size_t j = 0; j < input.cols; ++j) {
            float normalized = (input.get(i, j) - row_mean) * inv_variance.get(i, 0);
            float scaled = normalized * gamma.get(0, j) + beta.get(0, j);
            normalized_input.set(i, j, scaled);
        }
    }

    auto inner_node_outputs = inner_node->forward(std::span<const matrix>{ &normalized_input, 1 });
    matrix final_output = inner_node_outputs.outputs[0].clone();
    final_output.add(input); // Residual connection
    
    std::vector<matrix> return_vec = std::move(inner_node_outputs.outputs);
    return_vec.emplace_back(std::move(final_output));
    return_vec.emplace_back(std::move(normalized_input));
    return_vec.emplace_back(std::move(mean));
    return_vec.emplace_back(std::move(inv_variance));
    return standardResult(std::move(return_vec));
}

std::vector<matrix> LayerNorm::backpropogate(const ForwardingResult& result,
                                             std::span<const matrix> inputs,
                                             std::span<const matrix> gradients,
                                             float learning_rate) {
    const matrix& layer_input = inputs[0];
    const matrix& normalized_input = result.outputs[result.outputs.size() - 3];
    const matrix& mean = result.outputs[result.outputs.size() - 2];
    const matrix& inv_variance = result.outputs[result.outputs.size() - 1];
    const matrix& grad_output = gradients[0];

    // The gradient dL/dy splits at the residual connection y = x + z
    // The gradient for the residual path (x) is grad_output.
    // The gradient for the main path (z) is also grad_output.
    matrix grad_residual = grad_output.clone();

    auto inner_backprop_outputs = inner_node->backpropogate(
        result,
        std::span<const matrix>{ &normalized_input, 1 },
        std::span<const matrix>{ &grad_output, 1 },
        learning_rate);
    matrix& grad_normalized = inner_backprop_outputs[0];

    matrix grad_input(layer_input.rows, layer_input.cols);
    matrix grad_gamma(1, dimensions);
    matrix grad_beta(1, dimensions);

    for (size_t i = 0; i < layer_input.rows; i++) {
        float row_mean = mean.get(i, 0);
        float row_inv_var = inv_variance.get(i, 0);

        float d_norm_sum = 0.0f;
        float d_norm_dot_x_norm = 0.0f;

        for (size_t j = 0; j < layer_input.cols; j++) {
            float grad_norm_val = grad_normalized.get(i, j);
            float normalized_val = (layer_input.get(i, j) - row_mean) * row_inv_var;

            grad_beta.offset(0, j, grad_norm_val);
            grad_gamma.offset(0, j, grad_norm_val * normalized_val);

            float d_norm = grad_norm_val * gamma.get(0, j);
            d_norm_sum += d_norm;
            d_norm_dot_x_norm += d_norm * normalized_val;
        }

        for (size_t j = 0; j < layer_input.cols; j++) {
            float normalized_val = (layer_input.get(i, j) - row_mean) * row_inv_var;
            float d_norm = grad_normalized.get(i, j) * gamma.get(0, j);

            float grad_in = (dimensions * d_norm) - d_norm_sum - (normalized_val * d_norm_dot_x_norm);
            grad_in *= row_inv_var / static_cast<float>(dimensions);
            
            grad_input.set(i, j, grad_in);
        }
    }

    adjust_parameter_matrix(gamma, grad_gamma, learning_rate);
    adjust_parameter_matrix(beta, grad_beta, learning_rate);
    
    // Add the gradient from the residual path
    grad_input.add(grad_residual);
    return matrix::construct_vec(grad_input);
}

void LayerNorm::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    gamma.save(out);
    beta.save(out);
    
    auto node_type = inner_node->getType();
    out.write(reinterpret_cast<const char*>(&node_type), sizeof(node_type));
    inner_node->save(out);
}

LayerNorm LayerNorm::load(std::istream& in) {
    LayerNorm layer;
    in.read(reinterpret_cast<char*>(&layer.dimensions), sizeof(layer.dimensions));
    in.read(reinterpret_cast<char*>(&layer.epsilon), sizeof(layer.epsilon));
    layer.gamma = matrix::load(in);
    layer.beta = matrix::load(in);
    layer.inner_node = load_node(in);
    return layer;
}