#pragma once

#include <memory>

#include <dataset/dataset.hpp>

// Factory function to create the appropriate dataset based on file type.
enum class dataset_type {
    RAW,
    ROW_BASED,
    OVERFIT
};

std::unique_ptr<dataset> create_dataset(const std::string_view path, dataset_type type);

dataset_type detect_dataset_type(std::string_view arg);

std::string to_string(dataset_type type);