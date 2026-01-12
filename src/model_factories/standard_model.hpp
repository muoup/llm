#pragma once

#include <cstddef>
#include <inference/feed_forward.hpp>
#include <inference/inference.hpp>

InferenceModel minimal_model(size_t vocab_size, DataType dtype);

InferenceModel standard_attention_model(size_t dimensions,
                                        size_t vocab_size,
                                        size_t num_blocks,
                                        size_t attention_heads,
                                        DataType dtype,
                                        ActivationFunction activation
                                        = ActivationFunction::GeLU);
