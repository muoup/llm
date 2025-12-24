#include "layer_normalize.hpp"

#include <chrono>
#include <cstddef>
#include <inference/inference.hpp>
#include <inference/network_node.hpp>
#include <iomanip>
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
      inner_node(std::move(inner_node)) {
    if (this->inner_node) {
        this->context_name = node_type_to_string(this->inner_node->getType());
    } else {
        this->context_name = "LayerNorm";
    }
}

size_t LayerNorm::parameterCount() const {
    size_t inner_parameters = [&]() -> size_t {
        if (inner_node) {
            return inner_node->parameterCount();
        } else {
            return 0;
        }
    }();

    return inner_parameters + (gamma.rows * gamma.cols)
           + (beta.rows * beta.cols);
}

void LayerNorm::randomize(float min, float max) {
    gamma.set_all(1.0f);
    beta.set_all(0.0f);

    if (inner_node)
        inner_node->randomize(min, max);
}

NodeType LayerNorm::getType() const {
    return NodeType::LayerNorm;
}

ForwardingResult LayerNorm::forward(std::span<const matrix> inputs,
                                    bool perf) const {
    const matrix& input = inputs[0];

    auto start_norm = std::chrono::high_resolution_clock::now();
    auto results
        = kernel::layer_norm::layer_normalization(input, gamma, beta, epsilon);

    if (!inner_node) {
        auto fr = standardResult(matrix::construct_vec(
            results.normalized, results.mean, results.inv_variance));
        return fr;
    }

    auto start_inner = std::chrono::high_resolution_clock::now();
    auto inner_node_outputs
        = inner_node->forward(std::span(&results.normalized, 1), perf);
    inner_node_outputs.outputs[0].add(input);

    if (perf) {
        kernel::optimizer::wait_for_operations();
        auto end_inner = std::chrono::high_resolution_clock::now();

        auto end_norm = start_inner;
        std::chrono::duration<double, std::milli> norm_dur
            = end_norm - start_norm;
        std::chrono::duration<double, std::milli> inner_dur
            = end_inner - start_inner;

        std::cout << "[PERF] " << context_name
                  << " norm forward: " << std::fixed << std::setprecision(3)
                  << norm_dur.count() << " ms" << std::endl;
        std::cout << "[PERF] " << context_name << " inner ("
                  << node_type_to_string(inner_node->getType())
                  << ") forward: " << inner_dur.count() << " ms" << std::endl;
    }

    LOG_DEBUG("  LayerNorm Forward:");
    LOG_DEBUG("    input norm: %f", input.norm());
    LOG_DEBUG("    normalized norm: %f", results.normalized.norm());
    LOG_DEBUG("    mean norm: %f", results.mean.norm());
    LOG_DEBUG("    inv_variance norm: %f", results.inv_variance.norm());
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
                                             float learning_rate,
                                             bool perf) {
    const matrix& layer_input = inputs[0];
    const matrix& mean = result.outputs.rbegin()[1];
    const matrix& inv_variance = result.outputs.rbegin()[0];

    // Maybe unused if no inner node
    std::vector<matrix> inner_backprop_outputs;

    auto start_inner = std::chrono::high_resolution_clock::now();
    // The gradient dL/dy splits at the residual connection y = x + z
    // The gradient for the residual path (x) is grad_output.
    // The gradient for the main path (z) is also grad_output.
    const matrix& grad_normalized = [&]() -> const matrix& {
        const matrix& normalized_input = result.outputs.rbegin()[2];

        if (inner_node) {
            inner_backprop_outputs = inner_node->backpropogate(
                result, std::span<const matrix>{ &normalized_input, 1 },
                gradients, learning_rate);
            return inner_backprop_outputs[0];
        } else {
            return gradients[0];
        }
    }();

    if (perf) {
        kernel::optimizer::wait_for_operations();
        auto end_inner = std::chrono::high_resolution_clock::now();
        if (inner_node) {
            std::chrono::duration<double, std::milli> duration
                = end_inner - start_inner;
            std::cout << "[PERF] " << context_name << " inner ("
                      << node_type_to_string(inner_node->getType())
                      << ") backprop: " << std::fixed << std::setprecision(3)
                      << duration.count() << " ms" << std::endl;
        }
    }

    auto start_norm = std::chrono::high_resolution_clock::now();
    LOG_DEBUG("  LayerNorm Inputs: ");
    LOG_DEBUG("    layer_input norm: %f", layer_input.norm());
    LOG_DEBUG("    mean norm: %f", mean.norm());
    LOG_DEBUG("    inv_variance norm: %f", inv_variance.norm());
    LOG_DEBUG("    grad_normalized norm: %f", grad_normalized.norm());

    kernel::layer_norm::LayerNormGradients results
        = kernel::layer_norm::layer_normalization_backward(
            layer_input, gamma, beta, mean, inv_variance, grad_normalized,
            epsilon);

    if (perf) {
        kernel::optimizer::wait_for_operations();
        auto end_norm = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration
            = end_norm - start_norm;
        std::cout << "[PERF] " << context_name
                  << " norm backprop: " << duration.count() << " ms"
                  << std::endl;
    }

    LOG_DEBUG("  LayerNorm Layer Gradients:");
    LOG_DEBUG("    grad_gamma norm: %f", results.grad_gamma.norm());
    LOG_DEBUG("    grad_beta norm: %f", results.grad_beta.norm());
    LOG_DEBUG("    grad_input norm: %f", results.grad_input.norm());

    kernel::optimizer::adjust_parameter_matrix(gamma, results.grad_gamma,
                                               learning_rate);
    kernel::matrix::check_errors("pre adjust beta");
    kernel::optimizer::adjust_parameter_matrix(beta, results.grad_beta,
                                               learning_rate);
    kernel::matrix::check_errors("post adjust beta");

    // Add the gradient from the residual path
    if (inner_node) {
        results.grad_input.add(gradients[0]);
    }
    return matrix::construct_vec(results.grad_input);
}

void LayerNorm::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    gamma.save(out);
    beta.save(out);

    bool has_inner_node = (inner_node != nullptr);
    out.write(reinterpret_cast<const char*>(&has_inner_node),
              sizeof(has_inner_node));

    if (!inner_node)
        return;

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

    bool has_inner_node;
    in.read(reinterpret_cast<char*>(&has_inner_node), sizeof(has_inner_node));
    if (has_inner_node) {
        layer.inner_node = load_node(in);
        layer.context_name = node_type_to_string(layer.inner_node->getType());
    } else {
        layer.context_name = "LayerNorm";
    }

    return layer;
}
