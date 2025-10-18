#include "neural_net.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>

 // ---[ Serialization Helpers ]---
 static void write_matrix(std::ofstream& file, const matrix& m) {
     // Write dimensions as before
     uint64_t dims[] = {m.rows, m.cols};
     file.write(reinterpret_cast<const char*>(dims), sizeof(dims));
     file.write(reinterpret_cast<const char*>(m.data_ptr()), m.buffer_size());
 }
 
 static void read_matrix(std::ifstream& file, matrix& m) {
     uint64_t dims[2];
     file.read(reinterpret_cast<char*>(dims), sizeof(dims));
     m = matrix(dims[0], dims[1]);
     
     const auto buffer_size = m.buffer_size();
     file.read(reinterpret_cast<char*>(m.data_ptr()), buffer_size);
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
    for (const auto& layer : model.m_ff_layers) {
        write_matrix(file, layer.w1);
        write_matrix(file, layer.b1);
        write_matrix(file, layer.w2);
        write_matrix(file, layer.b2);
    }
    write_matrix(file, model.m_logit_layer.w);
    write_matrix(file, model.m_logit_layer.b);
    
    std::cout << "Model saved: " << path << std::endl;
    std::cout << "Dimensions: " << dimensions << ", Layers: " << layer_count << ", Vocab Size: " << vocab_size << std::endl;
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

    llm model(vocab_size, layer_count, dimensions);
    
    for (auto& embedding : model.m_embedding_layer.m_embeddings) {
        read_matrix(file, embedding.data);
    }
    for (auto& layer : model.m_attention_layers) {
        read_matrix(file, layer.wq);
        read_matrix(file, layer.wk);
        read_matrix(file, layer.wv);
        read_matrix(file, layer.wo);
    }
    for (auto& layer : model.m_ff_layers) {
        read_matrix(file, layer.w1);
        read_matrix(file, layer.b1);
        read_matrix(file, layer.w2);
        read_matrix(file, layer.b2);
    }
    read_matrix(file, model.m_logit_layer.w);
    read_matrix(file, model.m_logit_layer.b);
    
    std::cout << "Model loaded: " << path << std::endl;
    std::cout << "Dimensions: " << dimensions << ", Layers: " << layer_count << ", Vocab Size: " << vocab_size << std::endl;
    
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
    for (auto &layer : m_ff_layers) {
        layer.randomize(min, max);
    }
    
    m_logit_layer.randomize(min, max);
}

matrix llm::prediction_matrix(const std::span<const token_id_t> tokens) const {
    matrix acc = m_embedding_layer.apply(tokens);
    
    for (size_t i = 0; i < m_ff_layers.size(); ++i) {
        matrix residual = acc.clone();
        
        acc = m_attention_layers[i].apply(acc).output;
        acc.add(residual);
        acc = m_ff_layers[i].apply(acc).output;
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
    ss << "LLM with " << m_embedding_layer.m_embeddings.size() << " embeddings and " << m_ff_layers.size() << " layers.\n";
    return ss.str();
}

bool llm::equals(const llm& other, const float epsilon) const {
    if (m_dimensions != other.m_dimensions || m_layer_count != other.m_layer_count ||
        m_embedding_layer.m_embeddings.size() != other.m_embedding_layer.m_embeddings.size()) {
        return false;
    }

    for (size_t i = 0; i < m_embedding_layer.m_embeddings.size(); ++i) {
        const auto& emb1 = m_embedding_layer.m_embeddings[i].data;
        const auto& emb2 = other.m_embedding_layer.m_embeddings[i].data;
        for (size_t r = 0; r < emb1.rows; ++r) {
            for (size_t c = 0; c < emb1.cols; ++c) {
                if (std::abs(emb1.get(r, c) - emb2.get(r, c)) > epsilon) {
                    return false;
                }
            }
        }
    }

    for (size_t l = 0; l < m_attention_layers.size(); ++l) {
        const auto& layer1 = m_attention_layers[l];
        const auto& layer2 = other.m_attention_layers[l];
        const matrix* matrices1[] = { &layer1.wq, &layer1.wk, &layer1.wv, &layer1.wo };
        const matrix* matrices2[] = { &layer2.wq, &layer2.wk, &layer2.wv, &layer2.wo };

        for (size_t m = 0; m < 4; ++m) {
            const auto& mat1 = *matrices1[m];
            const auto& mat2 = *matrices2[m];
            for (size_t r = 0; r < mat1.rows; ++r) {
                for (size_t c = 0; c < mat1.cols; ++c) {
                    if (std::abs(mat1.get(r, c) - mat2.get(r, c)) > epsilon) {
                        return false;
                    }
                }
            }
        }
    }

    // Similar comparison can be done for ff_layer and logit_layer if needed.

    return true;
}