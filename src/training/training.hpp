#pragma once

#include <tokenizer/token.hpp>
#include <span>

struct llm;

void train(llm& model, std::span<const token_id_t> input);