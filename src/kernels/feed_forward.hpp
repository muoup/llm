#pragma once

#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>

namespace kernel::feed_forward {

void add_bias(matrix& mat,
              const matrix& row_vec,
              kernel_stream_t stream = nullptr);
matrix sum_columns(const matrix& mat, kernel_stream_t stream = nullptr);

matrix leaky_relu_activation(const matrix& activation_input,
                             kernel_stream_t stream = nullptr);
matrix leaky_relu_activation_backprop(const ::matrix& activation_input,
                                      const ::matrix& a1_gradient,
                                      kernel_stream_t stream = nullptr);

matrix gelu_activation(const matrix& activation_input,
                       kernel_stream_t stream = nullptr);
matrix gelu_activation_backprop(const ::matrix& activation_input,
                                const ::matrix& a1_gradient,
                                kernel_stream_t stream = nullptr);

matrix silu_activation(const matrix& activation_input,
                       kernel_stream_t stream = nullptr);
matrix silu_activation_backprop(const ::matrix& activation_input,
                                const ::matrix& a1_gradient,
                                kernel_stream_t stream = nullptr);

}  // namespace kernel::feed_forward
