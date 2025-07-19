#pragma once

#include "../tokenizer/token.h"

struct llm;

void train(llm& model, std::span<const token_id_t> input);