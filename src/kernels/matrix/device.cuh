#pragma once

#include <cuda_runtime_api.h>
#include <kernels/scheduling.cuh>
#include <util/matrix.hpp>
#include <cuda_fp16.hpp>
#include <cuda_bf16.hpp>

namespace kernel::matrix::device {

// ===== Device Inline Helpers =====

inline __device__ float* get_addr_float(void* data, size_t stride, size_t row, size_t col);
inline __device__ half* get_addr_half(void* data, size_t stride, size_t row, size_t col);
inline __device__ __nv_bfloat16* get_addr_bf16(void* data, size_t stride, size_t row, size_t col);

inline __device__ float get_float(const void* data, size_t stride, size_t row, size_t col);
inline __device__ half get_half(const void* data, size_t stride, size_t row, size_t col);
inline __device__ __nv_bfloat16 get_bf16(const void* data, size_t stride, size_t row, size_t col);

inline __device__ void set_float(void* data, size_t stride, size_t row, size_t col, float value);
inline __device__ void set_half(void* data, size_t stride, size_t row, size_t col, half value);
inline __device__ void set_bf16(void* data, size_t stride, size_t row, size_t col, __nv_bfloat16 value);

inline __device__ void offset_elem_float(void* data, size_t stride, size_t row, size_t col, float value);
inline __device__ void offset_elem_half(void* data, size_t stride, size_t row, size_t col, half value);
inline __device__ void offset_elem_bf16(void* data, size_t stride, size_t row, size_t col, __nv_bfloat16 value);

inline __device__ void offset_elem_atomic_float(void* data, size_t stride, size_t row, size_t col, float value);
inline __device__ void offset_elem_atomic_half(void* data, size_t stride, size_t row, size_t col, half value);
inline __device__ void offset_elem_atomic_bf16(void* data, size_t stride, size_t row, size_t col, __nv_bfloat16 value);

inline __device__ float warp_reduce_sum(float val);
inline __device__ float warp_reduce_max(float val);
inline __device__ float block_reduce_sum(float val);
inline __device__ float block_reduce_max(float val);

// ===== Device Kernels =====

// Copy kernels
__global__ void copy_matrix_kernel_float(const matrix_view dest, const const_matrix_view src);
__global__ void copy_matrix_kernel_half(matrix_view dest, const const_matrix_view src);
__global__ void copy_matrix_kernel_bf16(matrix_view dest, const const_matrix_view src);

// Set all kernels
__global__ void set_all_kernel_float(matrix_view data, float value);
__global__ void set_all_kernel_half(matrix_view data, float value);
__global__ void set_all_kernel_bf16(matrix_view data, float value);

// Scale kernels
__global__ void scale_kernel_float(matrix_view data, float factor);
__global__ void scale_kernel_half(matrix_view data, float factor);
__global__ void scale_kernel_bf16(matrix_view data, float factor);

// Add matrix kernels
__global__ void add_matrix_kernel_float(matrix_view data, const const_matrix_view other);
__global__ void add_matrix_kernel_half(matrix_view data, const const_matrix_view other);
__global__ void add_matrix_kernel_bf16(matrix_view data, const const_matrix_view other);

// Atomic add matrix kernels
__global__ void add_matrix_atomic_kernel_float(matrix_view data, const const_matrix_view other);
__global__ void add_matrix_atomic_kernel_half(matrix_view data, const const_matrix_view other);
__global__ void add_matrix_atomic_kernel_bf16(matrix_view data, const const_matrix_view other);

// Add scaled kernels
__global__ void add_scaled_kernel_float(matrix_view data, const const_matrix_view other, float factor);
__global__ void add_scaled_kernel_half(matrix_view data, const const_matrix_view other, float factor);
__global__ void add_scaled_kernel_bf16(matrix_view data, const const_matrix_view other, float factor);

// Add value kernels
__global__ void add_value_kernel_float(matrix_view data, float value);
__global__ void add_value_kernel_half(matrix_view data, float value);
__global__ void add_value_kernel_bf16(matrix_view data, float value);

// Transfer row kernels
__global__ void transfer_row_kernel_float(matrix_view dest, size_t dest_row, const const_matrix_view src, size_t src_row);
__global__ void transfer_row_kernel_half(matrix_view dest, size_t dest_row, const const_matrix_view src, size_t src_row);
__global__ void transfer_row_kernel_bf16(matrix_view dest, size_t dest_row, const const_matrix_view src, size_t src_row);

// Add row vector kernels
__global__ void add_row_vector_kernel_float(matrix_view data, size_t data_row, const const_matrix_view row_vec, size_t offset_row, float scale);
__global__ void add_row_vector_kernel_half(matrix_view data, size_t data_row, const const_matrix_view row_vec, size_t offset_row, float scale);
__global__ void add_row_vector_kernel_bf16(matrix_view data, size_t data_row, const const_matrix_view row_vec, size_t offset_row, float scale);

// Atomic add row vector kernels
__global__ void atomic_add_row_vector_kernel_float(matrix_view data, size_t data_row, const const_matrix_view row_vec, size_t offset_row, float scale);
__global__ void atomic_add_row_vector_kernel_half(matrix_view data, size_t data_row, const const_matrix_view row_vec, size_t offset_row, float scale);
__global__ void atomic_add_row_vector_kernel_bf16(matrix_view data, size_t data_row, const const_matrix_view row_vec, size_t offset_row, float scale);

// Set horizontal slice kernels
__global__ void set_horizontal_slice_kernel_float(matrix_view data, size_t start_col, const const_matrix_view slice);
__global__ void set_horizontal_slice_kernel_half(matrix_view data, size_t start_col, const const_matrix_view slice);
__global__ void set_horizontal_slice_kernel_bf16(matrix_view data, size_t start_col, const const_matrix_view slice);

// Mask upper triangle kernels
__global__ void mask_upper_triangle_kernel_float(matrix_view data, float mask_value);
__global__ void mask_upper_triangle_kernel_half(matrix_view data, float mask_value);
__global__ void mask_upper_triangle_kernel_bf16(matrix_view data, float mask_value);

// Element-wise multiply kernels
__global__ void element_wise_multiply_kernel_float(matrix_view a_data, const const_matrix_view b_data);
__global__ void element_wise_multiply_kernel_half(matrix_view a_data, const const_matrix_view b_data);
__global__ void element_wise_multiply_kernel_bf16(matrix_view a_data, const const_matrix_view b_data);

// Type conversion kernels
__global__ void convert_float_to_half(float* float_data, uint16_t* half_data, size_t count);
__global__ void convert_float_to_bf16(float* float_data, uint16_t* bf16_data, size_t count);

// Softmax kernels
__global__ void softmax_kernel_float(matrix_view data);
__global__ void softmax_kernel_half(matrix_view data);
__global__ void softmax_kernel_bf16(matrix_view data);

// Backprop softmax kernels
__global__ void backprop_softmax_kernel_float(const const_matrix_view softmax_output, const const_matrix_view output_gradient, matrix_view softmax_gradient);
__global__ void backprop_softmax_kernel_half(const const_matrix_view softmax_output, const const_matrix_view output_gradient, matrix_view softmax_gradient);
__global__ void backprop_softmax_kernel_bf16(const const_matrix_view softmax_output, const const_matrix_view output_gradient, matrix_view softmax_gradient);

// Comparison kernels
__global__ void compare_kernel_float(const const_matrix_view a, const const_matrix_view b, float epsilon, bool* result);
__global__ void compare_kernel_half(const const_matrix_view a, const const_matrix_view b, float epsilon, bool* result);
__global__ void compare_kernel_bf16(const const_matrix_view a, const const_matrix_view b, float epsilon, bool* result);

// Reduction kernels
__global__ void sum_reduction_float(const const_matrix_view data, float* result, float identity);
__global__ void sum_reduction_half(const const_matrix_view data, float* result, float identity);
__global__ void sum_reduction_bf16(const const_matrix_view data, float* result, float identity);

// Variance kernels
__global__ void variance_kernel_float(const float* sum_ptr, const float* sum_sq_ptr, float* result, int total_elements);
__global__ void variance_kernel_half(const float* sum_ptr, const float* sum_sq_ptr, float* result, int total_elements);
__global__ void variance_kernel_bf16(const float* sum_ptr, const float* sum_sq_ptr, float* result, int total_elements);

} // namespace kernel::matrix::device
