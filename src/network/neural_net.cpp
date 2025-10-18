#include "neural_net.h"

#include <iostream>
#include <sstream>
#include <fstream>

// ---[ Serialization Helpers ]---
static void write_matrix(std::ofstream& file, const matrix& m) {
    uint64_t dims[] = {m.rows, m.cols};
    file.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    file.write(reinterpret_cast<const char*>(m.data_ptr()), m.rows * m.cols * sizeof(float));
}

static void read_matrix(std::ifstream& file, matrix& m) {
    uint64_t dims[2];
    file.read(reinterpret_cast<char*>(dims), sizeof(dims));
    m = matrix(dims[0], dims[1]);
    file.read(reinterpret_cast<char*>(m.data_ptr()), m.rows * m.cols * sizeof(float));
}

// ---[ Serialization Functions ]---
void save_llm(const llm& model, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return;

    uint32_t magic = 0x67676d6c; // "ggml"
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    uint32_t dimensions = model.m_dimensions;
    uint32_t layer_count = model.m_layer_count;
    uint32_t vocab_size = model.vocab_size();
    file.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    file.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

    for (const auto& embedding : model.m_embedding_layer.m_embeddings) {
        write_matrix(file, embedding.data);
    }
    for (const auto& layer : model.m_attention_layers) {
        write_matrix(file, layer.wq);
        write_matrix(file, layer.wk);
        write_matrix(file, layer.wv);
        write_matrix(file, layer.wo);
    }
    for (const auto& layer : model.m_ff_layer) {
        write_matrix(file, layer.w1);
        write_matrix(file, layer.b1);
        write_matrix(file, layer.w2);
        write_matrix(file, layer.b2);
    }
    write_matrix(file, model.m_logit_layer.w);
    write_matrix(file, model.m_logit_layer.b);
}

std::optional<llm> load_llm(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return std::nullopt;

    uint32_t magic, version, dimensions, layer_count, vocab_size;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != 0x67676d6c || version != 1) return std::nullopt;

    file.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    file.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

    llm model(dimensions, layer_count, vocab_size);
    
    for (auto& embedding : model.m_embedding_layer.m_embeddings) {
        read_matrix(file, embedding.data);
    }
    for (auto& layer : model.m_attention_layers) {
        read_matrix(file, layer.wq);
        read_matrix(file, layer.wk);
        read_matrix(file, layer.wv);
        read_matrix(file, layer.wo);
    }
    for (auto& layer : model.m_ff_layer) {
        read_matrix(file, layer.w1);
        read_matrix(file, layer.b1);
        read_matrix(file, layer.w2);
        read_matrix(file, layer.b2);
    }
    read_matrix(file, model.m_logit_layer.w);
    read_matrix(file, model.m_logit_layer.b);
    
    return model;
}

// ---[ Model Operations ]---

void llm::randomize() {
    constexpr auto min = -0.5f;
    constexpr auto max = 0.5f;

    m_embedding_layer.randomize(min, max);
    for (auto &layer : m_attention_layers) {
        layer.randomize(min / 10, max / 10);
    }
    for (auto &layer : m_ff_layer) {
        layer.randomize(min, max);
    }
    
    m_logit_layer.randomize(min, max);
}

matrix llm::prediction_matrix(const std::span<const token_id_t> tokens) const {
    matrix acc = m_embedding_layer.apply(tokens);
    
    for (size_t i = 0; i < m_ff_layer.size(); ++i) {
        matrix residual = acc;
        acc = m_attention_layers[i].apply(acc).output;
        acc.add(residual);
        acc = m_ff_layer[i].apply(acc).output;
    }
    
    return m_logit_layer.apply(acc).softmax();
}

token_id_t llm::predict(const std::span<const token_id_t> tokens) const {
    const auto predictions = prediction_matrix(tokens);
    auto max_idx = 0;
    const size_t last_row = predictions.rows - 1;
    for (size_t i = 1; i < predictions.cols; i++) {
        if (predictions.get(last_row, i) > predictions.get(last_row, max_idx)) {
            max_idx = i;
        }
    }
    return max_idx;
}

std::string llm::to_string() const {
    std::stringstream ss;
    ss << "LLM with " << m_embedding_layer.m_embeddings.size() << " embeddings and " << m_ff_layer.size() << " layers.\n";
    return ss.str();
}