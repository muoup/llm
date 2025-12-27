#pragma once

#include <kernels/scheduling.hpp>
#include <inference/network_node.hpp>
#include <util/matrix.hpp>

#include <istream>
#include <vector>

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
    AttentionLayer(size_t dimensions, size_t head_count, bool masked);

    size_t parameterCount() const override;
    NodeType getType() const override;
    size_t headCount() const { return head_count; }
    
    ForwardingResult forward(std::span<const matrix> inputs,
                             bool perf = false) const override;
    std::vector<matrix> backpropogate(const ForwardingResult& result,
                                      std::span<const matrix> inputs,
                                      std::span<const matrix> gradients,
                                      CentralOptimizer& optimizer,
                                      bool perf = false) override;

    void randomize(float min, float max) override;
    void save(std::ostream& out) const override;

    // Static load function for deserialization via a factory
    static AttentionLayer load(std::istream& in);

    AttentionLayer()
        : dimensions(0), head_size(0), head_count(0), masked(false), wo(), streams(0) {}

    size_t dimensions, head_size, head_count;
    bool masked;

    std::vector<AttentionHead> heads;
    kernel::FixedStreamList streams;
    matrix wo;
};
