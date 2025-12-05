#pragma once

#include <cstddef>
#include <inference/inference.hpp>

InferenceModel create_standard_model(size_t dimensions, size_t vocab_size,
                                     size_t ff_layers, size_t attention_heads);
