#pragma once

#include <dataset/dataset.hpp>

#include <memory>
#include <string_view>
#include <string>
#include <optional>

// Factory function to create the appropriate dataset based on file type.
enum class dataset_type {
    RAW,
    ROW_BASED,
    OVERFIT
};

std::unique_ptr<dataset> create_dataset(const std::string_view path, dataset_type type, std::optional<size_t> specified_size = std::nullopt);

dataset_type detect_dataset_type(std::string_view arg);

std::string to_string(dataset_type type);