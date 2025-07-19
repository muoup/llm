#include "tokenizer.h"

#include <iostream>
#include <map>
#include <span>

using frequency_map_t = std::map<combo_token_t, std::size_t>;

token_list_t collect_char_data(const std::string_view input, token_map_t &token_map) {
    token_list_t tokens;

    for (int i = 0; i < 128; ++i) {
        token_map.insert(token_t { static_cast<char>(i) });
    }

    for (const char ch : input) {
        tokens.push_back(ch);
    }

    return tokens;
}

combo_token_t get_highest_frequency_pair(const token_span_t tokens) {
    frequency_map_t frequency_map;

    for (std::size_t i = 0; i < tokens.size() - 1; ++i) {
        const auto token1 = tokens[i];
        const auto token2 = tokens[i + 1];

        combo_token_t pair = { token1, token2 };
        frequency_map[pair]++;
    }

    combo_token_t highest_frequency_pair = { 0, 0 };
    std::size_t highest_frequency = 0;

    for (const auto &[pair, frequency] : frequency_map) {
        if (frequency > highest_frequency) {
            highest_frequency = frequency;
            highest_frequency_pair = pair;
        }
    }

    return highest_frequency_pair;
}

struct token_pair_seen {
    bool seen_token1 : 1 = false;
    bool seen_token2 : 1 = false;
};

token_pair_seen replace_map(std::vector<token_id_t> &tokens, const combo_token_t &pair, const token_id_t new_token) {
    auto begin = tokens.begin();
    auto cbegin = tokens.cbegin();

    token_pair_seen seen;

    while (cbegin < tokens.end() - 1) {
        if (*cbegin == pair.byte1 && *(cbegin + 1) == pair.byte2) {
            *begin = new_token;
            ++begin;
            cbegin += 2;
            continue;
        }

        if (*cbegin == pair.byte1) {
            seen.seen_token1 = true;
        } else if (*cbegin == pair.byte2) {
            seen.seen_token2 = true;
        }

        *begin = *cbegin;
        ++cbegin;
        ++begin;
    }

    tokens.erase(begin, tokens.end());
    return seen;
}

tokenize_results_t tokenize(const std::string_view input, const size_t tokenize_size) {
    token_map_t token_map;
    token_list_t tokens = collect_char_data(input, token_map);

    for (int i = 0; i < tokenize_size; ++i) {
        auto highest_frequency_pair = get_highest_frequency_pair(tokens);

        const auto token_id = token_map.insert(token_t { highest_frequency_pair });
        const auto seen = replace_map(tokens, highest_frequency_pair, token_id);

        if (!seen.seen_token1) token_map.remove(highest_frequency_pair.byte1);
        if (!seen.seen_token2) token_map.remove(highest_frequency_pair.byte2);
    }

    return tokenize_results_t { tokens, token_map };
}