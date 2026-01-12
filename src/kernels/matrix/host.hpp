#pragma once

#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>

namespace kernel::matrix {

// ===== Memory Management =====
::matrix async_allocate(size_t rows,
                        size_t cols,
                        DataType type,
                        kernel_stream_t stream = nullptr);
void* allocate_buffer(size_t size,
                      DataType type,
                      kernel_stream_t stream = nullptr);
void free_buffer(void* buffer, kernel_stream_t stream = nullptr);

// ===== Data Transfer =====
void load_into(::matrix& mat,
               const void* host_data,
               kernel_stream_t stream = nullptr);
void store_from(const ::matrix& mat,
                void* host_data,
                kernel_stream_t stream = nullptr);

// ===== Element Access =====
float get(const ::matrix& mat,
          size_t row,
          size_t col,
          kernel_stream_t stream = nullptr);
void set(::matrix& mat,
         size_t row,
         size_t col,
         float value,
         kernel_stream_t stream = nullptr);
void* get_addr(::matrix& mat, size_t row, size_t col);
const void* get_addr(const ::matrix& mat, size_t row, size_t col);

// ===== Randomization =====
void randomize(::matrix& mat,
               float min,
               float max,
               kernel_stream_t stream = nullptr);

// ===== Clone =====
::matrix clone(const ::const_matrix_view mat, kernel_stream_t stream = nullptr);

// ===== Basic Operations =====
void scale(::matrix& mat, float factor, kernel_stream_t stream = nullptr);
void add(::matrix& mat,
         const ::matrix& other,
         kernel_stream_t stream = nullptr);
void atomic_add(::matrix& mat,
                const ::matrix& other,
                kernel_stream_t stream = nullptr);
void add_scaled(::matrix& mat,
                const ::matrix& other,
                float factor,
                kernel_stream_t stream = nullptr);
void add(::matrix& mat, float value, kernel_stream_t stream = nullptr);
void set_all(::matrix& mat, float value, kernel_stream_t stream = nullptr);

// ===== Row Operations =====
void transfer_row(::matrix& dest,
                  size_t dest_row,
                  const ::matrix& src,
                  size_t src_row,
                  kernel_stream_t stream = nullptr);
void set_row_vector(::matrix& mat,
                    size_t mat_row,
                    const ::matrix& vec,
                    size_t vec_row,
                    kernel_stream_t stream = nullptr);
::matrix get_row_vector(const ::matrix& mat,
                        size_t row,
                        kernel_stream_t stream = nullptr);
void add_row_vector(::matrix& mat,
                    size_t row,
                    const ::matrix& vec,
                    size_t vec_row,
                    float scale,
                    kernel_stream_t stream = nullptr);
void atomic_add_row_vector(::matrix& mat,
                           size_t row,
                           const ::matrix& vec,
                           size_t vec_row,
                           float scale,
                           kernel_stream_t stream = nullptr);

// ===== Slice Operations =====
void set_horizontal_slice(::matrix& mat,
                          size_t start_col,
                          const ::matrix& slice,
                          kernel_stream_t stream = nullptr);

// ===== Masking =====
void mask_upper_triangle(::matrix& mat,
                         float mask_value,
                         kernel_stream_t stream = nullptr);

// ===== Element-wise Multiply =====
void element_wise_multiply(::matrix& a,
                           const ::matrix& b,
                           kernel_stream_t stream = nullptr);

}  // namespace kernel::matrix
