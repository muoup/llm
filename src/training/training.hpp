#pragma once

#include <tokenizer/token.hpp>
#include <span>

struct InferenceModel;

void train(InferenceModel& model, const std::span<const token_id_t> input, float learning_rate = 0.0001f);