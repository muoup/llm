#include "inference/recursion_node.hpp"

#include <inference/inference.hpp>
#include <inference/network_node.hpp>
#include <training/optimizer.hpp>

ForwardingResult RecursionNode::forward(std::span<const matrix> inputs) const {
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
            loop_input = recursion_data.loopNodeOutputs[recursion_count].back().outputs;
        }

        // We don't for sure know the size of the final output of the loop, so
        // we need to lazy initialize it here.
        if (final_output.rows == 0) {
            final_output = matrix(loop_input[0].rows, loop_input[0].cols);
        }

        auto p_n = loop_input[0].cross_multiplied(w);
        for (size_t r = 0; r < p_n.rows; r++) {
            p_n.add_row_vector(r, b);
        }

        // Sigmoid activation
        p_n = p_n.mapped([](float x) { return 1.0f / (1.0f + std::exp(-x)); });

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

std::vector<matrix> RecursionNode::backpropogate(
    const ForwardingResult& results,
    std::span<const matrix> inputs,
    std::span<const matrix> gradients,
    float learning_rate) {
        
    std::cout << "RecursionNode::backpropogate called" << std::endl;
        
    constexpr auto TIME_PENALTY = 0.01f;

    auto* rec_data = dynamic_cast<RecursionData*>(results.data.get());
    if (!rec_data) {
        throw std::runtime_error(
            "RecursionNode::backpropogate: Invalid node data");
    }
    RecursionData& recursion_data = *rec_data;

    std::vector<matrix> output_gradient_storage;
    std::span<const matrix> output_gradient_span = std::span(gradients);

    float chance_acc = 1.0f;

    for (int loop_index = recursion_data.recursionCount; loop_index >= 0;
         loop_index--) {
        chance_acc -= recursion_data.loopProbabilities[loop_index];

        // Before we backpropogate through our contained nodes, we have to
        const auto& y_n
            = recursion_data.loopNodeOutputs[loop_index].back().outputs[0];
        const auto& y_n_gradient = output_gradient_span[0];

        auto gradient_p_n = y_n_gradient.cross_t_multiplied(y_n);

        auto d_w = y_n.t_cross_multiplied(gradient_p_n);
        regularize_weight_gradient(d_w, w);
        adjust_parameter_matrix(w, d_w, learning_rate);

        auto d_b = matrix(1, 1);
        for (size_t r = 0; r < d_b.rows; r++) {
            float row_sum = gradient_p_n.col_sum(r);
            d_b.set(0, 0, d_b.get(0, 0) + row_sum);
        }

        regularize_weight_gradient(d_b, b);
        adjust_parameter_matrix(b, d_b, learning_rate);

        for (int node_index = loop.size(); node_index >= 0; node_index--) {
            auto [loop_output, loop_output_gradient] = [&]()
                -> std::pair<const ForwardingResult*, std::span<const matrix>> {
                if (loop_index == recursion_data.recursionCount) {
                    return std::make_pair(&results, gradients);
                } else {
                    const auto& results
                        = recursion_data
                              .loopNodeOutputs[loop_index + 1][node_index];

                    return std::make_pair(&results, output_gradient_span);
                }
            }();

            auto loop_inputs = [&]() -> std::span<const matrix> {
                if (node_index == 0) {
                    if (loop_index == 0) {
                        return std::span(inputs);
                    } else {
                        return std::span(
                            recursion_data.loopNodeOutputs[loop_index - 1]
                                .back()
                                .outputs);
                    }
                } else {
                    return std::span(
                        recursion_data
                            .loopNodeOutputs[loop_index][node_index - 1]
                            .outputs);
                }
            }();

            auto& node = loop[node_index];

            output_gradient_storage = node->backpropogate(
                *loop_output, loop_inputs, loop_output_gradient, learning_rate);
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
    size_t dimensions;
    size_t max_recursion_depth;
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

    RecursionNode recursion_node(dimensions, max_recursion_depth, std::move(loop));
    recursion_node.w = std::move(w);
    recursion_node.b = std::move(b);

    return recursion_node;
}
