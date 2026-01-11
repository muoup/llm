#pragma once

#include <cstdint>
#include <inference/network_node.hpp>
#include <istream>
#include <util/matrix.hpp>
#include <vector>

enum class ActivationFunction : uint8_t { LeakyReLU, GeLU, SwiGLU };

// Note: The INode forward pass is pure and does not modify layer state.
// It returns a vector of matrices containing the actual output followed by
// any intermediate values needed for backpropagation.
//
// For FeedForwardLayer with LeakyReLU/GeLU, the forward output vector is:
// [0] -> output matrix
// [1] -> activation_input (result of input * w1 + b1)
// [2] -> activation_output (result of activate(activation_input))
//
// For SwiGLU, the forward output vector is:
// [0] -> output matrix
// [1] -> gate_input (result of input * w1 + b1)
// [2] -> value_input (result of input * w2 + b2)
// [3] -> gate_output (result of SiLU(gate_input))

class FeedForwardLayer final : public INode {
   public:
    FeedForwardLayer(size_t dimensions,
                     size_t projection_size,
                     ActivationFunction activation
                     = ActivationFunction::LeakyReLU);

    NodeType getType() const override;

    size_t parameterCount() const override;

    ForwardingResult forward(std::span<const matrix> inputs,
                             bool perf = false) const override;

    std::vector<matrix> backpropogate(const ForwardingResult& result,
                                      std::span<const matrix> inputs,
                                      std::span<const matrix> gradients,
                                      CentralOptimizer& optimizer,
                                      bool perf = false) override;

    void randomize(float min, float max) override;
    void save(std::ostream& out) const override;
    static FeedForwardLayer load(std::istream& in);

    matrix w1, b1;
    matrix w2, b2;
    matrix w3, b3;
    ActivationFunction activation;
};
