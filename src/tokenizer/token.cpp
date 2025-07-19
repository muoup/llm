//
// Created by user on 7/15/25.
//

#include "token.h"

#include <iostream>
#include <sstream>

std::string debug_string(const combo_token_t &token) {
    std::stringstream ss;

    ss << "(" << static_cast<int>(token.byte1) << ", "
       << static_cast<int>(token.byte2) << ")";

    return ss.str();
}

std::string debug_string(token_t token) {
    if (const auto *combo = std::get_if<combo_token_t>(&token.value))
        return debug_string(*combo);

    if (const auto *ch = std::get_if<char>(&token.value))
        return std::string { *ch };

    return "Unknown token type";
}

std::string tokens_to_plaintext(const token_map_t& token_map, const std::span<const token_id_t>& tokens) {
    std::stringstream ss;

    for (const auto &token_id : tokens) {
        const auto &token = token_map[token_id];
        ss << token_to_plaintext(token_map, token);
    }

    return ss.str();
}


std::string token_to_plaintext(const token_map_t &token_map, const token_t &token) {
    if (const auto *combo = std::get_if<combo_token_t>(&token.value)) {
        const auto str1 = token_to_plaintext(token_map, token_map[combo->byte1]);
        const auto str2 = token_to_plaintext(token_map, token_map[combo->byte2]);

        return str1 + str2;
    }

    if (const auto *ch = std::get_if<char>(&token.value)) {
        return std::string { *ch };
    }

    std::cout << "Unknown token type encountered in token_to_plaintext" << std::endl;
    exit(1);
}