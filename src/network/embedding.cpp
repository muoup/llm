#include "embedding.h"

#include <cmath>

// ---[ Layer Operations ]---

void embedding::randomize(const float min, const float max) {
    data.randomize(min, max);
}

void embedding_layer::randomize(const float min, const float max) {
    for (auto& embedding : m_embeddings) {
        embedding.randomize(min, max);
    }
}

static void positional_encoding(matrix& input) {
    for (size_t token_i = 0; token_i < input.rows; ++token_i) {
        for (size_t encoding_i = 0; encoding_i < input.cols / 2; ++encoding_i) {
            const auto inner = token_i / std::pow(10000, 2 * encoding_i / static_cast<float>(input.cols));
            input.offset(token_i, encoding_i, std::sin(inner));
            input.offset(token_i, encoding_i + 1, std::cos(inner));
        }
    }
}

matrix embedding_layer::apply(const std::span<const token_id_t> tokens) const {
    matrix output { tokens.size(), m_dimensions };
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto &embedding = m_embeddings[tokens[i]];
        output.set_row_vector(i, embedding.data);
    }
    positional_encoding(output);
    return output;
}
