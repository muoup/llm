#include "layer_normalize.hpp"

#include <inference/inference.hpp>
#include <inference/network_node.hpp>
#include <kernels/layer_norm.hpp>
#include <kernels/matrix_kernels.hpp>
#include <kernels/optimizer.hpp>
#include <util/logger.hpp>

LayerNorm::LayerNorm(std::unique_ptr<INode> inner_node,
                     size_t dimensions,
                     float epsilon)
    : dimensions(dimensions),
      epsilon(epsilon),
      gamma(1, dimensions),
      beta(1, dimensions),
      inner_node(std::move(inner_node)) {}

size_t LayerNorm::parameterCount() const {
    return (gamma.rows * gamma.cols) + (beta.rows * beta.cols)
           + inner_node->parameterCount();
}

void LayerNorm::randomize(float min, float max) {
    gamma.randomize(min, max);
    beta.randomize(min, max);
    inner_node->randomize(min, max);
}

NodeType LayerNorm::getType() const {
    return NodeType::LayerNorm;
}

ForwardingResult LayerNorm::forward(std::span<const matrix> inputs) const {
    const matrix& input = inputs[0];

    auto results
        = kernel::layer_norm::layer_normalization(input, gamma, beta, epsilon);

    MATRIX_ASSERT(results.normalized.rows == input.rows
                      && results.normalized.cols == input.cols,
                  "LayerNorm forward: normalized output dimensions mismatch "
                  "(expected %zux%zu, got %zux%zu)",
                  input.rows, input.cols, results.normalized.rows,
                  results.normalized.cols);

    auto inner_node_outputs
        = inner_node->forward(std::span(&results.normalized, 1));
    inner_node_outputs.outputs[0].add(input);

    LOG_DEBUG("  LayerNorm Forward:");
    LOG_DEBUG("    input norm: %f",
                input.norm());
    LOG_DEBUG("    normalized norm: %f",
                results.normalized.norm());
    LOG_DEBUG("    mean norm: %f",
                results.mean.norm());
    LOG_DEBUG("    inv_variance norm: %f",
                results.inv_variance.norm());
    LOG_DEBUG("    post_residual_connection norm: %f",
                inner_node_outputs.outputs[0].norm());

    std::vector<matrix> return_vec = std::move(inner_node_outputs.outputs);
    return_vec.emplace_back(std::move(results.normalized));
    return_vec.emplace_back(std::move(results.mean));
    return_vec.emplace_back(std::move(results.inv_variance));
    return standardResult(std::move(return_vec));
}

std::vector<matrix> LayerNorm::backpropogate(const ForwardingResult& result,
                                             std::span<const matrix> inputs,
                                             std::span<const matrix> gradients,
                                             float learning_rate) {
    const matrix& layer_input = inputs[0];
    const matrix& normalized_input = result.outputs.rbegin()[2];
    const matrix& mean = result.outputs.rbegin()[1];
    const matrix& inv_variance = result.outputs.rbegin()[0];

    // The gradient dL/dy splits at the residual connection y = x + z
    // The gradient for the residual path (x) is grad_output.
    // The gradient for the main path (z) is also grad_output.
    std::vector<matrix> inner_backprop_outputs = inner_node->backpropogate(
        result, std::span<const matrix>{ &normalized_input, 1 }, gradients,
        learning_rate);
    matrix& grad_normalized = inner_backprop_outputs[0];

    LOG_DEBUG("  LayerNorm Inputs: ");
    LOG_DEBUG("    layer_input norm: %f",
                layer_input.norm());
    LOG_DEBUG("    normalized_input norm: %f",
                normalized_input.norm());
    LOG_DEBUG("    mean norm: %f",
                mean.norm());
    LOG_DEBUG("    inv_variance norm: %f",
                inv_variance.norm());
    LOG_DEBUG("    grad_normalized norm: %f",
                grad_normalized.norm());

    kernel::layer_norm::LayerNormGradients results
        = kernel::layer_norm::layer_normalization_backward(
            layer_input, gamma, beta, mean, inv_variance, grad_normalized,
            epsilon);

    LOG_DEBUG("  LayerNorm Layer Gradients:");
    LOG_DEBUG("    grad_gamma norm: %f",
                results.grad_gamma.norm());
    LOG_DEBUG("    grad_beta norm: %f",
                results.grad_beta.norm());
    LOG_DEBUG("    grad_input norm: %f",
                results.grad_input.norm());

    kernel::optimizer::adjust_parameter_matrix(gamma, results.grad_gamma,
                                               learning_rate);
    kernel::matrix::check_errors("pre adjust beta");
    kernel::optimizer::adjust_parameter_matrix(beta, results.grad_beta,
                                               learning_rate);
    kernel::matrix::check_errors("post adjust beta");

    // Add the gradient from the residual path
    results.grad_input.add(gradients[0]);
    return matrix::construct_vec(results.grad_input);
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
    in.read(reinterpret_cast<char*>(&layer.dimensions),
            sizeof(layer.dimensions));
    in.read(reinterpret_cast<char*>(&layer.epsilon), sizeof(layer.epsilon));
    layer.gamma = matrix::load(in);
    layer.beta = matrix::load(in);
    layer.inner_node = load_node(in);
    return layer;
}
