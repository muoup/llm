#pragma once

#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>

namespace kernel::optimizer {

void norm_clip(::matrix& gradient,
               kernel_stream_t stream = nullptr,
               size_t normalization_count = 0);

void regularize_weight_gradient(::matrix& gradient,
                                const ::matrix& parameters,
                                kernel_stream_t stream = nullptr,
                                size_t normalization_count = 0);

void adjust_parameter_matrix(::matrix& adjust,
                             ::matrix& gradient,
                             float learning_rate,
                             kernel_stream_t stream = nullptr);

}  // namespace kernel::optimizer
