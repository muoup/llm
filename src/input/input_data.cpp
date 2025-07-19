//
// Created by user on 7/18/25.
//

#include "input_data.h"

#include <fstream>
#include <sstream>

std::string get_file_data(const char* file_path) {
    std::ifstream file(file_path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + std::string(file_path));
    }

    std::stringstream ss;

    while (file) {
        std::string line;
        std::getline(file, line);
        ss << line << "\n";
    }

    return format_data(ss.str());
}

bool valid_whitespace(auto iter, const auto begin, const auto end) {
    if (end - iter == 1 || iter == begin)
        return false;

    if (std::iswspace(iter[-1]) || std::iswspace(iter[1]))
        return false;

    return true;
}

std::string format_data(std::string data) {
    std::string formatted_data;

    for (auto ptr = data.begin(); ptr < data.end(); ++ptr) {
        const auto ch = *ptr;

        if (std::iswspace(*ptr)) {
            if (valid_whitespace(ptr, data.begin(), data.end()))
                formatted_data.append(1, ' ');
        } else if (ch >= 0 && ch < 128) {
            formatted_data.append(1, ch);
        }
    }

    return formatted_data;
}