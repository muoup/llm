#include "inference/recursion_node.hpp"

#include <inference/inference.hpp>
#include <inference/network_node.hpp>
#include <kernels/optimizer.hpp>

ForwardingResult RecursionNode::forward(std::span<const matrix> inputs,
                                        bool perf) const {
    RecursionData recursion_data;

    float budget = 0.0f;

    matrix final_output;
    size_t recursion_count = 0;

    for (recursion_count = 0; recursion_count < max_recursion_depth;
         recursion_count++) {
        recursion_data.loopNodeOutputs.emplace_back();

        std::span<const matrix> loop_input = inputs;

        for (size_t j = 0; j < loop.size(); j++) {
            auto loop_node = loop[j].get();
            auto loop_forward = loop_node->forward(loop_input);

            recursion_data.loopNodeOutputs[recursion_count].emplace_back(
                std::move(loop_forward));
            loop_input = recursion_data.loopNodeOutputs[recursion_count]
                             .back()
                             .outputs;
        }

        // We don't for sure know the size of the final output of the loop, so
        // we need to lazy initialize it here.
        if (final_output.rows == 0) {
            final_output = matrix(loop_input[0].rows, loop_input[0].cols, loop_input[0].type);
        }

        auto p_n = loop_input[0].cross_multiplied(w);
        for (size_t r = 0; r < p_n.rows; r++) {
            p_n.add_row_vector(r, b);
        }
        recursion_data.presigmoidValues.emplace_back(p_n.clone());

        // Sigmoid activation
        // TODO:
        // p_n = p_n.mapped([](float x) { return 1.0f / (1.0f + std::exp(-x));
        // });

        auto probability = p_n.col_sum(0) / static_cast<float>(p_n.rows);
        budget += probability;
        recursion_data.loopProbabilities.emplace_back(probability);

        final_output.add_scaled(loop_input[0], probability);

        if (budget > 1.0f) {
            break;
        }
    }

    return ForwardingResult{ .data = std::make_unique<RecursionData>(
                                 std::move(recursion_data)),
                             .outputs = matrix::construct_vec(final_output) };
}

std::span<const matrix> determine_node_inputs(
    size_t node_index,
    size_t loop_index,
    const std::span<const matrix>& initial_inputs,
    const RecursionData& recursion_data) {
    if (node_index == 0) {
        if (loop_index == 0) {
            return initial_inputs;
        } else {
            return std::span(
                recursion_data.loopNodeOutputs[loop_index - 1].back().outputs);
        }
    } else {
        return std::span(
            recursion_data.loopNodeOutputs[loop_index][node_index - 1].outputs);
    }
}

std::vector<matrix> RecursionNode::backpropogate(
    const ForwardingResult& results,
    std::span<const matrix> inputs,
    std::span<const matrix> gradients,
    CentralOptimizer& optimizer,
    bool perf) {
    constexpr auto TIME_PENALTY = 0.01f;

    auto* rec_data = dynamic_cast<RecursionData*>(results.data.get());
    if (!rec_data) {
        throw std::runtime_error(
            "RecursionNode::backpropogate: Invalid node data");
    }
    RecursionData& recursion_data = *rec_data;

    std::vector<matrix> output_gradient_storage;
    float chance_acc = 1.0f;

    for (int loop_index = recursion_data.recursionCount; loop_index >= 0;
         loop_index--) {
        std::span<const matrix> output_gradient_span = std::span(gradients);
        chance_acc -= recursion_data.loopProbabilities.at(loop_index);

        const auto& y_gradient = gradients[0];
        const auto& y_n
            = recursion_data.loopNodeOutputs.at(loop_index).back().outputs[0];
        const auto& p_n = recursion_data.loopProbabilities[loop_index];

        const auto dp_n_ponder = (loop_index * TIME_PENALTY) * (1 - p_n);
        const auto dp_n
            = y_gradient.t_cross_multiplied(y_n).sum() + dp_n_ponder;

        matrix dP_n = matrix(y_n.rows, 1, y_n.type);
        dP_n.set_all(dp_n * chance_acc);

        auto dw = y_n.t_cross_multiplied(dP_n);
        optimizer.update(w, dw);

        auto db = matrix(1, 1, b.type);
        for (size_t r = 0; r < dP_n.cols; r++) {
            float col_sum = dP_n.col_sum(r);
            db.set(0, 0, db.get(0, 0) + col_sum);
        }

        optimizer.update(b, db);

        auto dy_n = y_gradient.scaled(p_n);
        output_gradient_span = std::span(&dy_n, 1);

        for (int node_index = loop.size() - 1; node_index >= 0; node_index--) {
            auto node_inputs = determine_node_inputs(node_index, loop_index,
                                                     inputs, recursion_data);
            auto& node_forwarding_result
                = recursion_data.loopNodeOutputs.at(loop_index).at(node_index);
            auto& node = loop[node_index];

            if (node == nullptr) {
                throw std::runtime_error(
                    "RecursionNode::backpropogate: Null node in loop");
            }

            output_gradient_storage = node->backpropogate(
                node_forwarding_result, node_inputs, output_gradient_span,
                optimizer);
            output_gradient_span = std::span(output_gradient_storage);
        }
    }

    return std::move(output_gradient_storage);
}

void RecursionNode::randomize(float min, float max) {
    w.randomize(min, max);
    b.randomize(min, max);

    for (auto& node : loop) {
        node->randomize(min, max);
    }
}

void RecursionNode::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&max_recursion_depth),
              sizeof(max_recursion_depth));
    w.save(out);
    b.save(out);

    size_t loop_size = loop.size();
    out.write(reinterpret_cast<const char*>(&loop_size), sizeof(loop_size));
    for (const auto& node : loop) {
        NodeType node_type = node->getType();
        out.write(reinterpret_cast<const char*>(&node_type), sizeof(node_type));
        node->save(out);
    }
}

RecursionNode RecursionNode::load(std::istream& in) {
    size_t dimensions, max_recursion_depth;
    in.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    in.read(reinterpret_cast<char*>(&max_recursion_depth),
            sizeof(max_recursion_depth));
    
    matrix w = matrix::load(in);
    matrix b = matrix::load(in);

    size_t loop_size;
    in.read(reinterpret_cast<char*>(&loop_size), sizeof(loop_size));

    std::vector<std::unique_ptr<INode>> loop;
    for (size_t i = 0; i < loop_size; i++) {
        loop.emplace_back(load_node(in));
    }

    RecursionNode recursion_node;
    recursion_node.dimensions = dimensions;
    recursion_node.max_recursion_depth = max_recursion_depth;
    recursion_node.w = std::move(w);
    recursion_node.b = std::move(b);

    return recursion_node;
}
