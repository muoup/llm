#pragma once

#include <util/matrix.hpp>

namespace kernel::optimizer {

typedef void* kernel_stream_t;
    
void norm_clip(::matrix& gradient);

void regularize_weight_gradient(::matrix& gradient, const ::matrix& parameters);

void adjust_parameter_matrix(::matrix& adjust,
                             ::matrix& gradient,
                             float learning_rate);

void wait_for_operations();

}  // namespace kernel::optimizer
