#pragma once

#include "token.h"
#include <vector>

struct tokenize_results_t {
    std::vector<token_id_t> tokens;
    token_map_t token_map;
};

tokenize_results_t tokenize(std::string_view input, size_t tokenize_size);