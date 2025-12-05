#pragma once

#include <inference/node_type.hpp>
#include <span>
#include <util/matrix.hpp>
#include <vector>

struct INode {
    virtual ~INode() = default;

    virtual NodeType getType() const = 0;

    virtual size_t parameterCount() const = 0;

    virtual std::vector<matrix> forward(std::span<const matrix> inputs) const
        = 0;
    virtual std::vector<matrix> backpropogate(std::span<const matrix> inputs,
                                              std::span<const matrix> outputs,
                                              std::span<const matrix> gradients,
                                              float learning_rate)
        = 0;

    virtual void randomize(float min, float max) = 0;
    virtual void save(std::ostream& out) const = 0;
};
