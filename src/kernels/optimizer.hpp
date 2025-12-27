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

void adamw_step(::matrix& parameter,
                const ::matrix& gradient,
                ::matrix& m,
                ::matrix& v,
                size_t t,
                float learning_rate,
                float beta1 = 0.9f,
                float beta2 = 0.999f,
                float epsilon = 1e-8f,
                float weight_decay = 0.01f,
                kernel_stream_t stream = nullptr);

}  // namespace kernel::optimizer