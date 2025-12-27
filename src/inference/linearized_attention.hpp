#pragma once

#include <inference/network_node.hpp>
#include <inference/node_type.hpp>

// Mostly equivalent to AttentionLayer but when calculating Q @ K^T @ V, we swap
// the order of multiplication from (Q @ K^T) @ V to Q @ (K^T @ V) to reduce
// computational complexity.
//
// This reduces the time complexity, but also the accuracy of the attention
// mechanism, which could be useful for models with the goal of size and speed
// over accuracy.

struct LinearAttentionHead {
    matrix wq, wk, wv;
};

class LinearizedAttention : public INode {
   public:
    LinearizedAttention(size_t dimensions, size_t head_count);

    size_t parameterCount() const override;
    NodeType getType() const override { return NodeType::LinearizedAttention; }

    ForwardingResult forward(std::span<const matrix> inputs,
                             bool perf = false) const override;
    std::vector<matrix> backpropogate(const ForwardingResult& result,
                                      std::span<const matrix> inputs,
                                      std::span<const matrix> gradients,
                                      CentralOptimizer& optimizer,
                                      float learning_rate,
                                      bool perf = false) override;

    void randomize(float min, float max) override;
    void save(std::ostream& out) const override;
    static LinearizedAttention load(std::istream& in);

   private:
    LinearizedAttention();
    size_t dimensions, head_size, head_count;

    std::vector<LinearAttentionHead> heads;
    matrix wo;
};
