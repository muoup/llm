#pragma once

// NODE DESCRIPTION: A fork in a recursive structure. The node looks at its
// current output, and depending on the value, determines if it is satisfied or
// if it should recurse back to a previous node.

// Expected input: A single [S x D] matrix where S is the sequence length and D
// is the dimensionality.

#include <inference/network_node.hpp>

struct RecursionData : INodeData {
    size_t recursionCount;
    std::vector<float> loopProbabilities;
    std::vector<matrix> presigmoidValues;
    std::vector<std::vector<ForwardingResult>> loopNodeOutputs;

    RecursionData() : recursionCount(0) {}
    RecursionData(RecursionData&&) = default;
    ~RecursionData() override = default;
};

class RecursionNode final : public INode {
   public:
    RecursionNode(size_t dimensions,
                  size_t max_recursion_depth,
                  std::vector<std::unique_ptr<INode>> loop_nodes)
        : dimensions(dimensions),
          max_recursion_depth(max_recursion_depth),
          w(dimensions, 1),
          b(1, 1),
          loop(std::move(loop_nodes)) {}

    NodeType getType() const override { return NodeType::Recursion; }

    size_t parameterCount() const override {
        return (w.rows * w.cols) + (b.rows * b.cols);
    }

    ForwardingResult forward(std::span<const matrix> inputs) const override;
    std::vector<matrix> backpropogate(const ForwardingResult& results,
                                      std::span<const matrix> inputs,
                                      std::span<const matrix> gradients,
                                      float learning_rate) override;

    void randomize(float min, float max) override;
    void save(std::ostream& out) const override;
    static RecursionNode load(std::istream& in);

   private:
    RecursionNode() : dimensions(0), max_recursion_depth(0), w(0, 0), b(0, 0) {}

    size_t dimensions, max_recursion_depth;

    matrix w, b;
    std::vector<std::unique_ptr<INode>> loop;
};
