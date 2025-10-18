#pragma once

#include <vector>
#include <iostream>

#include <network/neural_net.h>

// tokens_span should be the original input tokens used for training
inline void check_teacher_forcing(const llm &model, const std::span<const token_id_t> tokens) {
    // Build truncated input exactly like training (exclude last token)
    std::vector<token_id_t> truncated(tokens.begin(), tokens.end() - 1);
    auto predictions = model.prediction_matrix({truncated.data(), truncated.size()}); // returns logits -> we want softmax
    predictions.softmax();

    int correct = 0;
    for (size_t i = 0; i < truncated.size(); ++i) {
        // golden next token is tokens[i+1]
        size_t max_idx = 0;
        for (size_t j = 1; j < predictions.cols; ++j) {
            if (predictions.get(i, j) > predictions.get(i, max_idx)) max_idx = j;
        }
        bool is_correct = (max_idx == tokens[i + 1]);
        std::cout << "pos " << i << " pred=" << max_idx << " gold=" << tokens[i + 1] << " correct=" << is_correct << "\n";
        if (is_correct) ++correct;
    }
    std::cout << "Teacher-forcing accuracy: " << correct << " " << truncated.size() << "\n";
}