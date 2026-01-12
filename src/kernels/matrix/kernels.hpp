#pragma once
#include <cuda_runtime_api.h>
#include <kernels/scheduling.cuh>
#include <util/matrix.hpp>
#include "device.cuh"
#include "mapping.hpp"
#include "pools.cuh"

namespace kernel::matrix {

// ===== Host-side implementations =====

::matrix async_allocate(size_t rows,
                        size_t cols,
                        DataType type,
                        kernel_stream_t stream);
void* allocate_buffer(size_t size, DataType type, kernel_stream_t stream);
void free_buffer(void* buffer, kernel_stream_t stream);

void load_into(::matrix& mat, const void* host_data, kernel_stream_t stream);
void store_from(const ::matrix& mat, void* host_data, kernel_stream_t stream);

float get(const ::matrix& mat, size_t row, size_t col, kernel_stream_t stream);
void set(::matrix& mat,
         size_t row,
         size_t col,
         float value,
         kernel_stream_t stream);
void* get_addr(::matrix& mat, size_t row, size_t col);
const void* get_addr(const ::matrix& mat, size_t row, size_t col);

void randomize(::matrix& mat, float min, float max, kernel_stream_t stream);
void check_errors(const char* step);

::matrix clone(const ::matrix& mat, kernel_stream_t stream);
void set_all(::matrix& mat, float value, kernel_stream_t stream);

void scale(::matrix& mat, float factor, kernel_stream_t stream);

void transfer_row(::matrix& dest,
                  size_t dest_row,
                  const ::matrix& src,
                  size_t src_row,
                  kernel_stream_t stream);
void set_row_vector(::matrix& mat,
                    size_t mat_row,
                    const ::matrix& vec,
                    size_t vec_row,
                    kernel_stream_t stream);
::matrix get_row_vector(const ::matrix& mat,
                        size_t row,
                        kernel_stream_t stream);
void add_row_vector(::matrix& mat,
                    size_t row,
                    const ::matrix& vec,
                    size_t vec_row,
                    float scale,
                    kernel_stream_t stream);
void atomic_add_row_vector(::matrix& mat,
                           size_t row,
                           const ::matrix& vec,
                           size_t vec_row,
                           float scale,
                           kernel_stream_t stream);

void add(::matrix& mat, const ::matrix& other, kernel_stream_t stream);
void atomic_add(::matrix& mat, const ::matrix& other, kernel_stream_t stream);
void add_scaled(::matrix& mat,
                const ::matrix& other,
                float factor,
                kernel_stream_t stream);
void add(::matrix& mat, float value, kernel_stream_t stream);
void set_horizontal_slice(::matrix& mat,
                          size_t start_col,
                          const ::matrix& slice,
                          kernel_stream_t stream);

void mask_upper_triangle(::matrix& mat,
                         float mask_value,
                         kernel_stream_t stream);

void element_wise_multiply(::matrix& a,
                           const ::matrix& b,
                           kernel_stream_t stream);

}  // namespace kernel::matrix
