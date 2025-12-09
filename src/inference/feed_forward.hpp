#pragma once

#include <util/matrix.hpp>
#include <inference/network_node.hpp>
#include <vector>
#include <istream>

// Note: The INode forward pass is pure and does not modify layer state.
// It returns a vector of matrices containing the actual output followed by
// any intermediate values needed for backpropagation.
//
// For FeedForwardLayer, the forward output vector is:
// [0] -> output matrix
// [1] -> activation_input (result of input * w1 + b1)
// [2] -> activation_output (result of activate(activation_input))

class FeedForwardLayer final : public INode {
public:
    FeedForwardLayer(size_t dimensions, size_t projection_size);

    NodeType getType() const override;
    
    size_t parameterCount() const override;
    
    ForwardingResult forward(std::span<const matrix> inputs) const override;
    std::vector<matrix> backpropogate(
        const ForwardingResult& result,
        std::span<const matrix> inputs,
        std::span<const matrix> gradients,
        float learning_rate) override;

    void randomize(float min, float max) override;
    
    void save(std::ostream& out) const override;
    static FeedForwardLayer load(std::istream& in);

private:
    matrix w1, b1;
    matrix w2, b2;
};
