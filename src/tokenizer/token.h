#pragma once

#include <queue>
#include <span>
#include <vector>
#include <string_view>
#include <unordered_map>
#include <variant>

using token_id_t = std::uint16_t;

using token_list_t = std::vector<token_id_t>;
using token_span_t = std::span<const token_id_t>;

struct combo_token_t {
    token_id_t byte1;
    token_id_t byte2;

    auto operator <=>(const combo_token_t &other) const {
        return std::tie(byte1, byte2) <=> std::tie(other.byte1, other.byte2);
    }
};

struct token_t {
    std::variant<combo_token_t, char> value;
};

struct token_map_t {
    std::vector<token_t> tokens;
    std::queue<token_id_t> free_ids;

    token_id_t insert(const token_t& token) {
        if (!free_ids.empty()) {
            const token_id_t id = free_ids.front();
            free_ids.pop();
            tokens[id] = token_t {};
            return id;
        }

        tokens.push_back(token);

        auto id = static_cast<token_id_t>(tokens.size() - 1);
        return id;
    }

    void remove(const token_id_t id) {
        if (id < tokens.size()) {
            tokens[id] = token_t {};
            free_ids.push(id);
        }
    }

    size_t size() const {
        return tokens.size();
    }

    token_t& operator[](const token_id_t id) {
        return tokens[id];
    }

    const token_t& operator[](const token_id_t id) const {
        return tokens[id];
    }

    [[nodiscard]] auto begin() { return tokens.begin(); }
    [[nodiscard]] auto end() { return tokens.end(); }
    [[nodiscard]] auto begin() const { return tokens.cbegin(); }
    [[nodiscard]] auto end() const { return tokens.cend(); }
};

std::string debug_string(const combo_token_t &token);
std::string debug_string(token_t token);

std::string token_to_plaintext(const token_map_t &token_map, const token_t &token);