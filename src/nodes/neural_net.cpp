#include "neural_net.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>

// ---[ Serialization Helpers ]---

// Helper to write a matrix to a stream
static void write_matrix(std::ostream& out, const matrix& m) {
    uint64_t dims[] = { m.rows, m.cols };
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(m.data_ptr()), m.buffer_size());
}

// Helper to read a matrix from a stream (used by non-node layers)
static void read_matrix(std::ifstream& file, matrix& m) {
    uint64_t dims[2];
    file.read(reinterpret_cast<char*>(dims), sizeof(dims));
    m = matrix(dims[0], dims[1]);
    file.read(reinterpret_cast<char*>(m.data_ptr()), m.buffer_size());
}

// ---[ Node Factory for Deserialization ]---
static std::unique_ptr<INode> load_node(std::istream& in) {
    NodeType type;
    in.read(reinterpret_cast<char*>(&type), sizeof(type));

    switch (type) {
        case NodeType::Attention:
            return std::make_unique<AttentionLayer>(AttentionLayer::load(in));
        case NodeType::FeedForward:
            return std::make_unique<FeedForwardLayer>(FeedForwardLayer::load(in));
        default:
            // Handle error: unknown node type
            std::cerr << "Error: Unknown node type during loading: " << static_cast<uint32_t>(type) << std::endl;
            return nullptr;
    }
}


// ---[ Serialization Functions ]---
void save_llm(const llm& model, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return;

    // Magic number and version
    uint32_t magic = 0x67676d6c; // "ggml"
    uint32_t version = 2; // Version 2 for polymorphic nodes
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Model parameters
    uint32_t dimensions = model.m_dimensions;
    uint32_t layer_count = model.m_layer_count;
    uint32_t vocab_size = model.vocab_size();
    file.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    file.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

    // Layers
    model.m_embedding_layer.save(file);
    for (const auto& layer : model.m_layers) {
        NodeType type = layer->getType();
        file.write(reinterpret_cast<const char*>(&type), sizeof(type));
        layer->save(file);
    }
    model.m_logit_layer.save(file);
    
    std::cout << "Model saved: " << path << std::endl;
}

std::optional<llm> load_llm(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return std::nullopt;

    uint32_t magic, version, dimensions, layer_count, vocab_size;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != 0x67676d6c || (version != 1 && version != 2)) return std::nullopt;

    file.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    file.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

    llm model(vocab_size, layer_count, dimensions);
    
    model.m_embedding_layer = embedding_layer::load(file, vocab_size, dimensions);
    
    // Clear the layers created by the constructor and load fresh ones
    model.m_layers.clear();
    for (size_t i = 0; i < layer_count * 2; ++i) { // 2 nodes per "layer" (Attn + FF)
        model.m_layers.push_back(load_node(file));
    }
    
    model.m_logit_layer = logit_layer::load(file, dimensions, vocab_size);
    
    std::cout << "Model loaded: " << path << std::endl;
    return model;
}

// ---[ Model Operations ]---

void llm::randomize() {
    constexpr auto min = -0.5f;
    constexpr auto max = 0.5f;

    m_embedding_layer.randomize(min, max);
    for (auto &layer : m_layers) {
        // Attention layers are more sensitive, use smaller weights
        if (layer->getType() == NodeType::Attention) {
            layer->randomize(min / 10, max / 10);
        } else {
            layer->randomize(min, max);
        }
    }
    m_logit_layer.randomize(min, max);
}

matrix llm::prediction_matrix(const std::span<const token_id_t> tokens) const {
    matrix acc = m_embedding_layer.forward(tokens);
    
    for (const auto& layer : m_layers) {
        auto outputs = layer->forward({acc});
        acc = std::move(outputs[0]);
    }
    
    matrix logits = m_logit_layer.apply(acc);
    logits.softmax();

    return logits;
}

token_id_t llm::predict(const std::span<const token_id_t> tokens) const {
    const auto predictions = prediction_matrix(tokens);
    auto max_idx = 0;
    const size_t last_row = predictions.rows - 1;
    
    for (size_t j = 1; j < predictions.cols; ++j) {
        if (predictions.get(last_row, j) > predictions.get(last_row, max_idx)) {
            max_idx = j;
        }
    }
    
    return static_cast<token_id_t>(max_idx);
}

std::string llm::to_string() const {
    std::stringstream ss;
    ss << "LLM with " << m_embedding_layer.get_vocab_size() << " embeddings and " << m_layer_count << " layers.\n";
    return ss.str();
}