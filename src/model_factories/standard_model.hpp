#pragma once

#include <cstddef>
#include <inference/inference.hpp>

InferenceModel standard_attention_model(size_t dimensions,
                                        size_t vocab_size,
                                        size_t ff_layers,
                                        size_t attention_heads);

InferenceModel linearized_attention_model(size_t dimensions,
                                          size_t vocab_size,
                                          size_t num_blocks,
                                          size_t attention_heads);

InferenceModel standard_recursive_model(size_t dimensions,
                                        size_t vocab_size,
                                        size_t num_blocks,
                                        size_t attention_heads,
                                        size_t max_recursions);
