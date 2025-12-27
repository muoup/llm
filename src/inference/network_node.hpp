#pragma once

#include <inference/node_type.hpp>
#include <inference/optimizer.hpp>
#include <span>
#include <util/matrix.hpp>
#include <vector>

#include <memory>

struct INodeData {
    virtual ~INodeData() = default;
};

struct ForwardingResult {
    std::unique_ptr<INodeData> data;
    std::vector<matrix> outputs;
};

struct INode {
    virtual ~INode() = default;

    virtual NodeType getType() const = 0;
    virtual size_t parameterCount() const = 0;

    virtual ForwardingResult forward(std::span<const matrix> inputs,
                                     bool perf = false) const
        = 0;
    virtual std::vector<matrix> backpropogate(const ForwardingResult& result,
                                              std::span<const matrix> inputs,
                                              std::span<const matrix> gradients,
                                              CentralOptimizer& optimizer,
                                              float learning_rate,
                                              bool perf = false)
        = 0;

    virtual void randomize(float min, float max) = 0;
    virtual void save(std::ostream& out) const = 0;

    static ForwardingResult standardResult(std::vector<matrix>&& outputs) {
        return ForwardingResult{ .data = nullptr,
                                 .outputs = std::move(outputs) };
    }
};
