#include "tokenizer.hpp"

#include <map>
#include <unordered_map>
#include <fstream>

struct pair_frequency_t {
    combo_token_t combo;
    size_t frequency;
};

constexpr token_id_t CHAR_START_RANGE = 32;
constexpr token_id_t CHAR_END_RANGE = 126;
constexpr token_id_t SUPPORTED_CHAR_COUNT = CHAR_END_RANGE - CHAR_START_RANGE;

static pair_frequency_t get_most_frequent_pair(const std::span<const token_id_t> tokens) {
    if (tokens.size() < 2) {
        return {0, 0};
    }

    std::unordered_map<combo_token_t, size_t> counts;
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        counts[{tokens[i], tokens[i+1]}]++;
    }

    combo_token_t best_pair { .byte1 = 0, .byte2 = 0 };
    size_t max_freq = 0;
    
    for (const auto& [pair, freq] : counts) {
        if (freq > max_freq) {
            max_freq = freq;
            best_pair = pair;
        }
    }
    
    return { .combo = best_pair, .frequency = max_freq };
}

static void replace_pair(std::vector<token_id_t>& tokens, const combo_token_t& pair, token_id_t new_token) {
    if (tokens.size() < 2) {
        return;
    }

    std::vector<token_id_t> new_tokens;
    new_tokens.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); ) {
        if (i + 1 < tokens.size() && tokens[i] == pair.byte1 && tokens[i+1] == pair.byte2) {
            new_tokens.push_back(new_token);
            i += 2;
        } else {
            new_tokens.push_back(tokens[i]);
            i += 1;
        }
    }
    tokens = std::move(new_tokens);
}

tokenizer::tokenizer() {
    token_map.reserve(SUPPORTED_CHAR_COUNT);
 
    for (int i = CHAR_START_RANGE; i < CHAR_END_RANGE; ++i) {
        token_map.emplace_back(std::string(1, static_cast<char>(i)));
    }
}

void train_tokenizer(tokenizer& tokenizer, std::string_view corpus, size_t max_vocab_size, size_t minimum_frequency) {
    auto tokens = encode(tokenizer, corpus); 
    
    while (tokenizer.token_map.size() < max_vocab_size) {
        pair_frequency_t most_frequent_pair = get_most_frequent_pair(tokens);
        
        if (most_frequent_pair.frequency < minimum_frequency) {
            break; // Stop if no pair is frequent enough
        }
        
        if (most_frequent_pair.combo.byte1 == 0 && most_frequent_pair.combo.byte2 == 0) {
            break; // No more pairs to merge
        }
        
        std::string new_token_text = tokenizer.token_map[most_frequent_pair.combo.byte1].text + tokenizer.token_map[most_frequent_pair.combo.byte2].text;
        token_id_t new_token_id = tokenizer.token_map.size();
        tokenizer.token_map.push_back({new_token_text});
        tokenizer.merges.push_back(most_frequent_pair.combo);
        
        if (tokenizer.token_map.size() % 100 == 0) {
            std::printf("Tokenizer training: vocabulary size %zu / %zu\r", tokenizer.token_map.size(), max_vocab_size);
            std::fflush(stdout);
        }

        replace_pair(tokens, most_frequent_pair.combo, new_token_id);
    }
}

std::vector<token_id_t> encode(const tokenizer& tokenizer, std::string_view text) {
    std::vector<token_id_t> tokens;
    tokens.reserve(text.size());
    
    for (char c : text) {
        if (c < CHAR_START_RANGE || c > CHAR_END_RANGE) {
            continue; // Skip non-printable characters
        }
        
        tokens.push_back(static_cast<unsigned char>(c - CHAR_START_RANGE));
    }

    for (size_t merge_id = 0; merge_id < tokenizer.merges.size(); ++merge_id) {
        const combo_token_t& merge = tokenizer.merges[merge_id];
        
        token_id_t new_token_id = SUPPORTED_CHAR_COUNT + merge_id;
        replace_pair(tokens, merge, new_token_id);
    }

    return tokens;
}

std::string decode(const tokenizer& tokenizer, const std::span<const token_id_t> tokens) {
    std::string decoded_text;
    for (token_id_t token_id : tokens) {
        if (token_id < tokenizer.token_map.size()) {
            decoded_text += tokenizer.token_map[token_id].text;
        }
    }
    return decoded_text;
}

void save_tokenizer(const tokenizer& tokenizer, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return;

    uint32_t magic = 0x67676d6c; // "ggml" in hex
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    uint32_t vocab_size = tokenizer.token_map.size();
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

    for (const auto& token : tokenizer.token_map) {
        uint32_t len = token.text.length();
        file.write(reinterpret_cast<const char*>(&len), sizeof(len));
        file.write(token.text.c_str(), len);
    }
}

std::optional<tokenizer> load_tokenizer(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return std::nullopt;

    uint32_t magic, version, vocab_size;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x67676d6c || version != 1) {
        return std::nullopt;
    }

    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    
    tokenizer tokenizer;

    tokenizer.token_map.clear();
    tokenizer.token_map.reserve(vocab_size);
    tokenizer.merges.clear(); 

    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint32_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string text(len, '\0');
        file.read(&text[0], len);
        tokenizer.token_map.push_back({text});
    }
    
    std::map<std::string, token_id_t> token_to_id;
    for(token_id_t i = 0; i < tokenizer.token_map.size(); ++i) {
        token_to_id[tokenizer.token_map[i].text] = i;
    }

    for (token_id_t i = SUPPORTED_CHAR_COUNT; i < tokenizer.token_map.size(); ++i) {
        const std::string& text = tokenizer.token_map[i].text;
        for (size_t j = 1; j < text.length(); ++j) {
            std::string s1 = text.substr(0, j);
            std::string s2 = text.substr(j);
            if (token_to_id.count(s1) && token_to_id.count(s2)) {
                tokenizer.merges.push_back({token_to_id[s1], token_to_id[s2]});
                break; 
            }
        }
    }
    
    return tokenizer;
}