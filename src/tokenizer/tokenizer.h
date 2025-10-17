#pragma once

#include "token.h"

#include <vector>
#include <span>
#include <string>
#include <string_view>

struct tokenizer {
    token_map_t token_map;
    std::vector<combo_token_t> merges;

    size_t vocab_size() const {
        return token_map.size();
    }
    
    tokenizer();
};

// Train the tokenizer on a corpus to learn the vocabulary and merge rules.
void train_tokenizer(tokenizer& tokenizer, std::string_view corpus, size_t vocab_size, size_t minimum_frequency = 3);

// Encode a piece of text into a sequence of token IDs.
std::vector<token_id_t> encode(const tokenizer& tokenizer, std::string_view text);

// Decode a sequence of token IDs back into a string.
std::string decode(const tokenizer& tokenizer, const std::span<const token_id_t> tokens);

// Save the tokenizer's state (vocabulary and merges) to a file.
void save_tokenizer(const tokenizer& tokenizer, const std::string& path);

// Load the tokenizer's state from a file.
void load_tokenizer(tokenizer& tokenizer, const std::string& path);
