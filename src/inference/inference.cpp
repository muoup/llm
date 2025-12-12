#include "inference.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <queue>

#include <inference/attention.hpp>
#include <inference/feed_forward.hpp>
#include <inference/layer_normalize.hpp>
#include <inference/linearized_attention.hpp>
#include <inference/network_node.hpp>

#include <kernels/matrix_kernels.hpp>

std::unique_ptr<INode> load_node(std::istream& in) {
    NodeType type;
    in.read(reinterpret_cast<char*>(&type), sizeof(type));

    switch (type) {
        case NodeType::Attention:
            return std::make_unique<AttentionLayer>(AttentionLayer::load(in));
        case NodeType::FeedForward:
            return std::make_unique<FeedForwardLayer>(
                FeedForwardLayer::load(in));
        case NodeType::LayerNorm:
            return std::make_unique<LayerNorm>(LayerNorm::load(in));
        case NodeType::LinearizedAttention:
            return std::make_unique<LinearizedAttention>(
                LinearizedAttention::load(in));
        default:
            // Handle error: unknown node type
            std::cerr << "Error: Unknown node type during loading: "
                      << static_cast<uint32_t>(type) << std::endl;
            return nullptr;
    }
}

void InferenceModel::save(std::ostream& out) const {
    // Magic number and version
    uint32_t magic = 0x67676d6c;  // "ggml"
    uint32_t version = 2;         // Version 2 for polymorphic nodes
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Model parameters
    uint32_t dimensions = this->m_dimensions;
    uint32_t vocab_size = this->vocab_size();

    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

    // Layers
    this->m_embedding_layer.save(out);

    size_t connection_count = this->m_connections.size();
    out.write(reinterpret_cast<const char*>(&connection_count),
              sizeof(connection_count));
    for (const auto& conn : this->m_connections) {
        out.write(reinterpret_cast<const char*>(&conn.from_idx),
                  sizeof(conn.from_idx));
        out.write(reinterpret_cast<const char*>(&conn.to_idx),
                  sizeof(conn.to_idx));
    }

    size_t layer_count = this->m_layers.size();
    out.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));
    for (const auto& layer : this->m_layers) {
        NodeType type = layer->getType();
        out << std::string_view{ reinterpret_cast<const char*>(&type),
                                 sizeof(type) };
        layer->save(out);
    }

    this->m_logit_layer.save(out);
}

size_t InferenceModel::parameter_count() const {
    size_t count = 0;
    count += m_embedding_layer.parameterCount();
    for (const auto& layer : m_layers) {
        count += layer->parameterCount();
    }
    count += m_logit_layer.parameterCount();
    return count;
}

InferenceModel InferenceModel::load(std::istream& in) {
    uint32_t magic, version, dimensions, vocab_size;

    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x67676d6c || version != 2)
        throw std::runtime_error("Invalid model file format or version.");

    in.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

    std::cout << "Instantiating model with vocab size " << vocab_size
              << " and dimensions " << dimensions << "." << std::endl;

    InferenceModel model = InferenceModel();
    model.m_dimensions = dimensions;

    model.m_embedding_layer = EmbeddingLayer::load(in);

    size_t connection_count;
    in.read(reinterpret_cast<char*>(&connection_count),
            sizeof(connection_count));
    model.m_connections.reserve(connection_count);

    for (size_t i = 0; i < connection_count; ++i) {
        NodeConnection conn;
        in.read(reinterpret_cast<char*>(&conn.from_idx), sizeof(conn.from_idx));
        in.read(reinterpret_cast<char*>(&conn.to_idx), sizeof(conn.to_idx));
        model.m_connections.push_back(conn);
    }

    size_t layer_count;
    in.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
    model.m_layers.reserve(layer_count);

    for (size_t i = 0; i < layer_count; ++i) {
        auto node = load_node(in);
        if (node) {
            model.m_layers.emplace_back(std::move(node));
        } else {
            throw std::runtime_error("Failed to load model layer.");
        }
    }

    model.m_logit_layer = LogitLayer::load(in);
    model.finalize();

    return model;
}

void InferenceModel::randomize() {
    constexpr auto min = -0.5f;
    constexpr auto max = 0.5f;

    m_embedding_layer.randomize(min, max);
    for (auto& layer : m_layers) {
        // Attention layers are more sensitive, use smaller weights
        if (layer->getType() == NodeType::Attention) {
            layer->randomize(min / 10, max / 10);
        } else {
            layer->randomize(min, max);
        }
    }
    m_logit_layer.randomize(min, max);
}

void InferenceModel::generate_path() {
    // Topological sort to generate a valid forward pass order
    size_t node_count = m_layers.size();
    std::vector<size_t> in_degree(node_count, 0);

    for (const auto& conn : m_connections) {
        in_degree[conn.to_idx]++;
    }

    std::queue<size_t> zero_in_degree;
    for (size_t i = 0; i < node_count; ++i) {
        if (in_degree[i] == 0) {
            zero_in_degree.push(i);
        }
    }

    execution_order.clear();

    while (!zero_in_degree.empty()) {
        size_t node_idx = zero_in_degree.front();
        zero_in_degree.pop();
        execution_order.push_back(node_idx);

        for (const auto& conn : m_connections) {
            if (conn.from_idx == node_idx) {
                in_degree[conn.to_idx]--;
                if (in_degree[conn.to_idx] == 0) {
                    zero_in_degree.push(conn.to_idx);
                }
            }
        }
    }

    if (execution_order.size() != node_count) {
        throw std::runtime_error(
            "Cycle detected in model graph; invalid architecture.");
    }
}

void InferenceModel::finalize() {
    this->finalized = true;
    this->generate_path();
}

std::vector<ForwardingResult> InferenceModel::forwarding_results(
    const std::span<const token_id_t> tokens) const {
    if (!finalized) {
        throw std::runtime_error("Model must be finalized before prediction.");
    }

    std::vector<ForwardingResult> results;

    matrix embeddings = m_embedding_layer.forward(tokens);
    kernel::matrix::check_errors("Forwarding embeddings...");
    results.emplace_back(
        INode::standardResult(matrix::construct_vec(embeddings)));

    for (size_t node_idx : execution_order) {
        auto forward_result
            = this->m_layers.at(node_idx)->forward(results.back().outputs);
        results.emplace_back(std::move(forward_result));
        std::string msg
            = std::string("Forwarding layer ") + std::to_string(node_idx + 1);
        kernel::matrix::check_errors(msg.data());
    }

    auto logits = m_logit_layer.forward(results.back().outputs[0]);
    kernel::matrix::check_errors("Forwarding logits...");
    results.emplace_back(INode::standardResult(matrix::construct_vec(logits)));
    return results;
}

token_id_t InferenceModel::predict(const std::span<const token_id_t> tokens,
                                   float temperature) const {
    auto results = this->forwarding_results(tokens);
    matrix logits = std::move(results.back().outputs[0]);

    logits.scale(temperature);
    logits.softmax();

    // Find the index of the maximum logit
    const size_t last_row = logits.rows - 1;
    float random = static_cast<float>(rand()) / RAND_MAX;

    for (size_t i = 0; i < logits.cols; ++i) {
        random -= logits.get(last_row, i);

        if (random <= 0.0f) {
            return i;
        }
    }

    throw std::runtime_error("Failed to sample next token.");
}

float InferenceModel::train_on(const std::span<const token_id_t> tokens,
                               const std::span<const token_id_t> actual,
                               float learning_rate) {
    if (!finalized) {
        throw std::runtime_error("Model must be finalized before training.");
    }

    auto results = this->forwarding_results(tokens);

    // Backprop through logit layer
    auto [logit_gradients, loss] = m_logit_layer.backpropogate(
        results[results.size() - 2].outputs[0], results.back().outputs[0],
        actual, learning_rate);
    kernel::matrix::check_errors("Backpropogating logits...");
    std::vector<std::vector<matrix>> gradients;
    gradients.emplace_back(matrix::construct_vec(logit_gradients));

    // Backprop through layers in reverse order
    for (size_t i = execution_order.size(); i-- > 0;) {
        size_t node_idx = execution_order[i];
        gradients.emplace_back(m_layers[node_idx]->backpropogate(
            results[i + 1], results[i].outputs, gradients.back(),
            learning_rate));
        std::string msg = std::string("Backpropogating layer ")
                          + std::to_string(i + 1) + "...";
        kernel::matrix::check_errors(msg.data());
    }

    this->m_embedding_layer.backpropogate(tokens, gradients.back()[0],
                                          learning_rate);
    kernel::matrix::check_errors("Backpropogating embeddings...");
    return loss;
}
