#include "dataset_factory.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <string_view>

std::unique_ptr<dataset> create_dataset(const std::string_view path, dataset_type type, size_t specified_size) {
    std::printf("Loading dataset from: %s\n", path.data());
    std::ifstream file(path.data());

    if (!file) {
        throw std::runtime_error("Could not open dataset file: " + std::string { path });
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string file_content = buffer.str();

    if (type == dataset_type::RAW) {
        auto ds = std::make_unique<raw_dataset>();
        ds->data = std::move(file_content);
        return ds;
    } else if (type == dataset_type::ROW_BASED) {
        auto ds = std::make_unique<row_dataset>();
        ds->data = std::move(file_content);

        std::string_view remaining_view(ds->data);
        const std::string_view delimiter = "<|endoftext|>";
        size_t true_size = 0;

        while (!remaining_view.empty() && true_size < specified_size) {
            size_t delimiter_pos = remaining_view.find(delimiter);

            if (delimiter_pos == std::string_view::npos) {
                if (!remaining_view.empty()) {
                    ds->rows.push_back(remaining_view);
                }
                break;
            }

            ds->rows.push_back(remaining_view.substr(0, delimiter_pos));

            size_t advance_by = delimiter_pos + delimiter.length();
            if (advance_by < remaining_view.length() && remaining_view[advance_by] == '\n') {
                advance_by++; // Also skip the newline that often follows the delimiter
            }
            remaining_view.remove_prefix(advance_by);
            true_size++;
        }
        
        return ds;
    } else if (type == dataset_type::OVERFIT) {
        constexpr size_t MAX_OVERFIT_ROWS = 1000;
        
        auto ds = std::make_unique<overfit_dataset>();
        ds->repeat_count = std::min(specified_size, MAX_OVERFIT_ROWS);
        ds->data = file_content.substr(0, 250); // Take first 250 characters
        return ds;
    }

    throw std::runtime_error("Unknown dataset type");
}

dataset_type detect_dataset_type(std::string_view type_str) {
    if (type_str == "raw") {
        return dataset_type::RAW;
    } else if (type_str == "row-based") {
        return dataset_type::ROW_BASED;
    } else if (type_str == "overfit") {
        return dataset_type::OVERFIT;
    } else {
        throw std::invalid_argument("Unknown dataset type: " + std::string(type_str));
    }
}

std::string to_string(dataset_type type) {
    switch (type) {
        case dataset_type::RAW:
            return "raw";
        case dataset_type::ROW_BASED:
            return "row-based";
        case dataset_type::OVERFIT:
            return "overfit";
        default:
            return "unknown";
    }
}
