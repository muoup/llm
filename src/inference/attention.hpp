#pragma once

#include <util/matrix.hpp>
#include <inference/network_node.hpp>
#include <vector>
#include <istream>


// Note: The INode forward pass is pure and does not modify layer state.
// It returns a vector of matrices containing the actual output followed by
// any intermediate values needed for backpropagation.
//
// For AttentionLayer, the forward output vector is:
// [0] -> output matrix
// [1] -> q (queries)
// [2] -> k (keys)
// [3] -> v (values)
// [4] -> scores (attention scores after softmax)

struct AttentionHead {
    matrix wq, wk, wv;
};

class AttentionLayer final : public INode {
public:
    AttentionLayer(size_t dimensions, size_t head_size, size_t head_count);

    // INode interface implementation
    NodeType getType() const override;
    std::vector<matrix> forward(std::span<const matrix> inputs) override;
    std::vector<matrix> backpropogate(
        std::span<const matrix> inputs,
        std::span<const matrix> outputs,
        std::span<const matrix> gradients,
        float learning_rate) override;

    void randomize(float min, float max) override;
    void save(std::ostream& out) const override;

    // Static load function for deserialization via a factory
    static AttentionLayer load(std::istream& in);

private:
    AttentionLayer()
        : dimensions(0), head_size(0), head_count(0), wo() {}

    size_t dimensions, head_size, head_count;

    std::vector<AttentionHead> heads;
    matrix wo;
};
