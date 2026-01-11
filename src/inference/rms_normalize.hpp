#pragma once

#include <inference/network_node.hpp>
#include <util/matrix.hpp>
#include "kernels/scheduling.hpp"

class RMSNorm final : public INode {
   public:
    RMSNorm(std::unique_ptr<INode> inner,
            size_t dimensions,
            float epsilon = 1e-5f);
    RMSNorm();

    size_t parameterCount() const override;
    NodeType getType() const override;

    void randomize(float min, float max) override;

    ForwardingResult forward(std::span<const matrix> inputs,
                             bool perf = false) const override;
    std::vector<matrix> backpropogate(const ForwardingResult& result,
                                      std::span<const matrix> outputs,
                                      std::span<const matrix> gradients,
                                      CentralOptimizer& optimizer,
                                      bool perf = false) override;

    void save(std::ostream& out) const override;
    static RMSNorm load(std::istream& in);

    std::unique_ptr<INode> inner_node;
    matrix gamma;
    size_t dimensions;
    float epsilon;
    std::string context_name = "RMSNorm";

    kernel::FixedStreamList streams;
};
