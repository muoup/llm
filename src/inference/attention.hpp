#pragma once

#include <kernels/matrix_kernels.hpp>
#include <inference/network_node.hpp>
#include <util/matrix.hpp>

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
    AttentionLayer(size_t dimensions, size_t head_count, bool masked);
    AttentionLayer(AttentionLayer&&) noexcept;
    ~AttentionLayer();

    size_t parameterCount() const override;
    NodeType getType() const override;
    ForwardingResult forward(std::span<const matrix> inputs,
                             bool perf = false) const override;
    std::vector<matrix> backpropogate(const ForwardingResult& result,
                                      std::span<const matrix> inputs,
                                      std::span<const matrix> gradients,
                                      float learning_rate,
                                      bool perf = false) override;

    void randomize(float min, float max) override;
    void save(std::ostream& out) const override;

    // Static load function for deserialization via a factory
    static AttentionLayer load(std::istream& in);

   private:
    AttentionLayer();

    size_t dimensions, head_size, head_count;
    bool masked;

    std::vector<AttentionHead> heads;
    matrix wo;
    
    kernel::matrix::matmul_stream_t streams[4];
};
