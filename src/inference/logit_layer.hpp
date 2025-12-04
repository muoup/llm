#pragma once

#include <util/matrix.hpp>
#include <tokenizer/token.hpp>

#include <iostream>

// logit_layer is not an INode. It is a special exit layer mapping from
// the final matrix back to logits over the vocabulary.
class LogitLayer {
   public:
    explicit LogitLayer(const size_t dimensions, const size_t vocab_size);

    void randomize(float min, float max);
    matrix forward(const matrix& input) const;
    std::pair<matrix, float> backpropogate(
        const matrix& input, const matrix& predictions,
        const std::span<const token_id_t> actual, float learning_rate);

    void save(std::ostream& out) const;
    static LogitLayer load(std::istream& in);

    size_t get_vocab_size() const { return vocab_size; }

   private:
    size_t dimensions;
    size_t vocab_size;
    matrix w, b;
};
