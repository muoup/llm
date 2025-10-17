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

    for (const auto& embedding : model.m_embeddings) {
        write_matrix(file, embedding.data);
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

void load_llm(llm& model, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return;

    uint32_t magic, version, dimensions, layer_count, vocab_size;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != 0x67676d6c || version != 1) return;

    file.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    file.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

    if (model.m_dimensions != dimensions || model.m_layer_count != layer_count || model.vocab_size() != vocab_size) {
        return; // Model architecture mismatch
    }

    for (auto& embedding : model.m_embeddings) {
        read_matrix(file, embedding.data);
    }
    for (auto& layer : model.m_ff_layer) {
        read_matrix(file, layer.w1);
        read_matrix(file, layer.b1);
        read_matrix(file, layer.w2);
        read_matrix(file, layer.b2);
    }
    read_matrix(file, model.m_logit_layer.w);
    read_matrix(file, model.m_logit_layer.b);
}

// ---[ Model Operations ]---

void llm::randomize() {
    constexpr auto min = -0.5f;
    constexpr auto max = 0.5f;

    for (auto &embedding : m_embeddings) {
        embedding.data.randomize(min, max);
    }
    for (auto &layer : m_ff_layer) {
        layer.w1.randomize(min, max);
        layer.b1.randomize(min, max);
        layer.w2.randomize(min, max);
        layer.b2.randomize(min, max);
    }
    
    m_logit_layer.w.randomize(min, max);
    m_logit_layer.b.randomize(min, max);
}

matrix llm::embed_tokens(const std::span<const token_id_t> tokens) const {
    matrix output { tokens.size(), m_dimensions };
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto &embedding = m_embeddings[tokens[i]];
        output.set_row_vector(i, embedding.data);
    }
    positional_encoding(output);
    return output;
}

void llm::positional_encoding(matrix& input) const {
    for (size_t token_i = 0; token_i < input.rows; ++token_i) {
        for (size_t encoding_i = 0; encoding_i < input.cols / 2; ++encoding_i) {
            const auto inner = token_i / std::pow(10000, 2 * encoding_i / static_cast<float>(input.cols));
            input.offset(token_i, encoding_i, std::sin(inner));
            input.offset(token_i, encoding_i + 1, std::cos(inner));
        }
    }
}

matrix llm::forward_l1(const matrix& input, const size_t layer) const {
    const auto& ff_layer = m_ff_layer.at(layer);
    matrix output = input.cross_multiply(ff_layer.w1);
    
    for (size_t i = 0; i < output.rows; ++i) {
        output.add_row_vector(i, ff_layer.b1);
    }
    
    return output;
}

matrix llm::activate(const matrix& input) const {
    constexpr static auto leaky_relu = [](const float f) { return f < 0 ? 0.01f * f : f; };
    matrix output { input };
    return output.map(leaky_relu);
}

matrix llm::forward_l2(const matrix& input, const size_t layer) const {
    const auto& ff_layer = m_ff_layer.at(layer);
    matrix output = input.cross_multiply(ff_layer.w2);
    
    for (size_t i = 0; i < output.rows; ++i) {
        output.add_row_vector(i, ff_layer.b2);
    }
    
    return output;
}

matrix llm::generate_logits(const matrix& input) const {
    matrix logits = input.cross_multiply(m_logit_layer.w);
    for (size_t i = 0; i < logits.rows; ++i) {
        logits.add_row_vector(i, m_logit_layer.b);
    }
    return logits;
}

matrix llm::feed_forward(const matrix& input, const size_t layer) const {
    const matrix l1_output = forward_l1(input, layer);
    const matrix activated = activate(l1_output);
    const matrix l2_output = forward_l2(activated, layer);
    
    return l2_output;
}

matrix llm::prediction_matrix(const std::span<const token_id_t> tokens) const {
    matrix acc = embed_tokens(tokens);
    
    for (size_t i = 0; i < m_ff_layer.size(); ++i) {
        const matrix l1_output = forward_l1(acc, i);
        const matrix activated = activate(l1_output);
        const matrix l2_output = forward_l2(activated, i);
        acc.offset(l2_output);
    }
    
    return generate_logits(acc);
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
    ss << "LLM with " << m_embeddings.size() << " embeddings and " << m_ff_layer.size() << " layers.\n";
    return ss.str();
}
