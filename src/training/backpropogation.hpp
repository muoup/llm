#pragma once

#include <span>
#include <vector>

#include <nodes/attention.hpp>
#include <nodes/feed_forward.hpp>

#include <tokenizer/token.hpp>
#include <util/matrix.hpp>

struct llm;

struct training_data {
    matrix predictions;
    std::vector<forward_result> forward_results;
    std::vector<attention_forward_result> attention_forward_results;
    std::vector<matrix> attention_inputs;
    matrix logit_input;

    std::span<const token_id_t> tokens;

    explicit training_data(const std::span<const token_id_t> token_span, const size_t dimensions)
        : predictions(token_span.size(), 1),
          logit_input(token_span.size(), dimensions),
          tokens(token_span) {}
};

struct optimization_results {
    matrix embeddings_gradient;

    std::vector<matrix> ff_weight_gradients;
    std::vector<matrix> ff_bias_gradients;

    matrix logit_weight_gradient;
    matrix logit_bias_gradient;
};

void regularize_weight_gradient(matrix &gradient, const matrix &weights);
void adjust_matrix(matrix &weights, const matrix &gradient, float learning_rate);