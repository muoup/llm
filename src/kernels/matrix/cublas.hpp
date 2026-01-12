#pragma once

#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>

namespace kernel::matrix {

// ===== Matrix Multiplication =====
::matrix cross_multiplied(const const_matrix_view& a,
                          const const_matrix_view& b,
                          kernel_stream_t stream = nullptr);
::matrix t_cross_multiplied(const const_matrix_view& a,
                            const const_matrix_view& b,
                            kernel_stream_t stream = nullptr);
::matrix cross_t_multiplied(const const_matrix_view& a,
                            const const_matrix_view& b,
                            kernel_stream_t stream = nullptr);

// ===== Reductions =====
void* sum(const ::matrix& mat, kernel_stream_t stream = nullptr);
void* sum_of_squares(const ::matrix& mat, kernel_stream_t stream = nullptr);
void* abssum(const ::matrix& mat, kernel_stream_t stream = nullptr);
void* max(const ::matrix& mat, kernel_stream_t stream = nullptr);
void* min(const ::matrix& mat, kernel_stream_t stream = nullptr);
void* absmax(const ::matrix& mat, kernel_stream_t stream = nullptr);
void* variance(const ::matrix& mat, kernel_stream_t stream = nullptr);

// ===== Activations =====
void softmax(::matrix& mat, kernel_stream_t stream = nullptr);
void backprop_softmax(::matrix& buffer,
                      const ::matrix& output,
                      const ::matrix& gradient,
                      kernel_stream_t stream = nullptr);

// ===== Comparison =====
bool is_equal(const ::matrix& a,
              const ::matrix& b,
              float epsilon,
              kernel_stream_t stream = nullptr);

}  // namespace kernel::matrix
