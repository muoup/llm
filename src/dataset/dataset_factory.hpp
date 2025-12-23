#pragma once

#include <dataset/dataset.hpp>

#include <memory>
#include <string_view>
#include <string>

// Factory function to create the appropriate dataset based on file type.
enum class dataset_type {
    RAW,
    ROW_BASED,
    OVERFIT
};

std::unique_ptr<dataset> create_dataset(const std::string_view path, dataset_type type, size_t specified_size);

dataset_type detect_dataset_type(std::string_view arg);

std::string to_string(dataset_type type);