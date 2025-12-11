#pragma once

#include <inference/network_node.hpp>
#include <util/matrix.hpp>

class LayerNorm final : public INode {
public:
    LayerNorm(std::unique_ptr<INode> inner, size_t dimensions, float epsilon = 1e-5f);
    LayerNorm() = default;

    size_t parameterCount() const override;
    NodeType getType() const override;
    
    void randomize(float min, float max) override;

    ForwardingResult forward(std::span<const matrix> inputs) const override;
    std::vector<matrix> backpropogate(const ForwardingResult& result,
                                      std::span<const matrix> outputs,
                                      std::span<const matrix> gradients,
                                      float learning_rate) override;

    void save(std::ostream& out) const override;
    static LayerNorm load(std::istream& in);
    
    std::unique_ptr<INode> inner_node;
    matrix gamma;
    matrix beta;
    size_t dimensions;
    float epsilon;
};
