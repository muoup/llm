#pragma once

#include <util/matrix.hpp>
#include <iostream>
#include <span>

// Forward declaration
struct llm;

// logit_layer is not an INode. It is a special exit layer mapping from
// the final matrix back to logits over the vocabulary.
class logit_layer {
public:
    explicit logit_layer(const size_t dimensions, const size_t vocab_size);
    
    void randomize(float min, float max);
    matrix apply(const matrix &input) const;

    void save(std::ostream& out) const;
    static logit_layer load(std::istream& in, size_t dimensions, size_t vocab_size);
    
    size_t get_vocab_size() const { return vocab_size; }

private:
    size_t vocab_size;
    matrix w, b;

    // Grant access to the standalone backpropagation function for this layer
    friend matrix backpropogate_logit_row(llm &model, const matrix &last_ff_output,
                               const matrix &predictions,
                               const std::span<const token_id_t> actual,
                               float learning_rate);
};
