#pragma once

#include <span>
#include <vector>

#include "../tokenizer/token.h"
#include "../util/matrix.h"

class llm;

struct forward_result {
    matrix layer_input;
    matrix activation_input;
    matrix activation_output;
};

struct training_data {
    matrix predictions;
    std::vector<forward_result> forward_results;
    matrix logit_input;

    std::span<const token_id_t> tokens;

    explicit training_data(const std::span<const token_id_t> tokens, const size_t dimensions)
        : predictions(tokens.size(), 1),
          logit_input( tokens.size(), dimensions ),
          tokens(tokens.begin(), tokens.end()) {}
};

void backpropogate(llm& model, const training_data& data);