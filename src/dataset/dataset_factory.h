#pragma once

#include <memory>

#include "dataset.hpp"

// Factory function to create the appropriate dataset based on file type.
enum class dataset_type {
    RAW,
    ROW_BASED
};

std::unique_ptr<dataset> create_dataset(const std::string_view path, dataset_type type);
