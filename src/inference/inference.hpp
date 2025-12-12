#pragma once

#include <vector>
#include <memory>

#include <tokenizer/token.hpp>
#include <util/matrix.hpp>

#include <inference/network_node.hpp>
#include <inference/embedding.hpp>
#include <inference/logit_layer.hpp>

struct InferenceModel;

struct NodeConnection {
    size_t from_idx;
    size_t to_idx;
};

std::unique_ptr<INode> load_node(std::istream& in);

struct InferenceModel {
    void randomize();

    std::vector<ForwardingResult> forwarding_results(std::span<const token_id_t> tokens) const;
    
    token_id_t predict(std::span<const token_id_t> tokens, float temperature = 1.0f) const;
    float train_on(std::span<const token_id_t> tokens, std::span<const token_id_t> actual, float learning_rate);
    
    size_t parameter_count() const;
    
    InferenceModel(size_t dimensions, size_t vocab_size)
        : m_dimensions(dimensions), m_embedding_layer(dimensions, vocab_size),
          m_logit_layer(dimensions, vocab_size) {}

    size_t add_layer(std::unique_ptr<INode> layer) {
        if (finalized) {
            throw std::runtime_error("Cannot add layer to finalized model.");
        }
        
        if (layer.get() == nullptr) {
            throw std::runtime_error("Cannot add null layer to model.");
        }
        
        m_layers.emplace_back(std::move(layer));
        return m_layers.size() - 1;
    }
    
    size_t add_connection(size_t from_idx, size_t to_idx) {
        if (finalized) {
            throw std::runtime_error("Cannot add connection to finalized model.");
        }
    
        m_connections.emplace_back(from_idx, to_idx);
        return m_connections.size() - 1;
    }
    
    void finalize();
    size_t vocab_size() const { return m_logit_layer.get_vocab_size(); }

    void save(std::ostream &out) const;
    static InferenceModel load(std::istream &in);
    
private:
    InferenceModel()
        : finalized(false), m_dimensions(0),
          m_embedding_layer(0, 0), m_logit_layer(0, 0) {}

    void generate_path();

    bool finalized = false;
    size_t m_dimensions;
    
    EmbeddingLayer m_embedding_layer;
    LogitLayer m_logit_layer;
    
    std::vector<size_t> execution_order;
    std::vector<std::unique_ptr<INode>> m_layers;
    std::vector<NodeConnection> m_connections;
};
