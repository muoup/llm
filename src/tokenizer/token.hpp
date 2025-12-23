#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <tuple>
#include <functional>

// A token ID is just an index into the vocabulary vector.
using token_id_t = uint16_t;

// A combo token represents a merge rule.
struct combo_token_t {
    token_id_t byte1;
    token_id_t byte2;
    
    auto operator<=>(const combo_token_t& other) const {
        return std::tie(byte1, byte2) <=> std::tie(other.byte1, other.byte2);
    }

    bool operator==(const combo_token_t& other) const = default;
};

template<>
struct std::hash<combo_token_t> {
    std::size_t operator()(const combo_token_t& k) const {
        // Simple hash combining two 16-bit integers into a 32-bit (or larger) size_t
        return (static_cast<size_t>(k.byte1) << 16) | k.byte2;
    }
};

// Represents a single token in the vocabulary.
// The `text` field stores the full byte representation of the token.
struct token_t {
    std::string text;
};

// The vocabulary is a simple vector of tokens.
// The index of a token is its ID.
using token_map_t = std::vector<token_t>;
