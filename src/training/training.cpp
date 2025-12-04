#include "training.hpp"

#include <inference/inference.hpp>
#include <tokenizer/token.hpp>
#include <training/optimizer.hpp>

void train(InferenceModel& model, const std::span<const token_id_t> input, float learning_rate) {
    const auto truncated_input = std::span { input.begin(), input.end() - 1 };
    
    model.train_on(truncated_input, input, learning_rate);
}
