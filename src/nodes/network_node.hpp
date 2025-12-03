#pragma once

#include <util/matrix.hpp>
#include <vector>

struct INode {
    virtual ~INode() = default;

    virtual std::vector<matrix> forward(const std::span<matrix>& inputs) = 0;
    virtual std::vector<matrix> backpropagate(
        const std::span<matrix>& inputs, const std::span<matrix>& gradients,
        float learning_rate)
        = 0;

    virtual void randomize(float min, float max) = 0;

    virtual void save(std::ostream& out) const = 0;
};
