#include "kernels/feed_forward.hpp"

#include <cublas_v2.h>
#include <kernels/matrix_device_kernels.cuh>

__global__ void kernel_add_bias(float* mat,
                                size_t mat_stride,
                                size_t mat_rows,
                                size_t mat_cols,
                                const float* row_vec,
                                size_t row_vec_stride) {
    // We can safely assume that:
    // row_vec_rows = 1
    // row_vec_cols = mat_cols

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < mat_rows && col < mat_cols) {
        auto val = kernel::matrix::device_get(row_vec, row_vec_stride, 0, col);
        kernel::matrix::device_offset_elem(mat, mat_stride, row, col, val);
    }
}

void kernel::feed_forward::add_bias(::matrix& mat, const ::matrix& row_vec) {
    dim3 block_size(16, 16);
    dim3 grid_size((mat.cols + block_size.x - 1) / block_size.x,
                   (mat.rows + block_size.y - 1) / block_size.y);

    kernel_add_bias<<<grid_size, block_size>>>(
        mat.data, mat.stride, mat.rows, mat.cols, row_vec.data, row_vec.stride);
}

__global__ void sum_columns_kernel(const float* base,
                                   size_t base_stride,
                                   size_t base_rows,
                                   size_t base_cols,
                                   float* result,
                                   size_t result_stride) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // We are safe to assume here that result_rows = 1 and result_columns =
    // base_columns

    if (col < base_cols) {
        float sum = 0.0f;
        for (int row = 0; row < base_rows; ++row) {
            sum += kernel::matrix::device_get(base, base_stride, row, col);
        }
        kernel::matrix::device_offset_elem(result, result_stride, 0, col, sum);
    }
}

matrix kernel::feed_forward::sum_columns(const ::matrix& mat) {
    ::matrix result(1, mat.cols);

    dim3 block_size(256, 1);
    dim3 grid_size((mat.cols + block_size.x - 1) / block_size.x, 1);

    sum_columns_kernel<<<grid_size, block_size>>>(
        mat.data, mat.stride, mat.rows, mat.cols, result.data, result.stride);
    return result;
}

__device__ float relu(float x) {
    return x > 0 ? x : 0.0f;
}

matrix kernel::feed_forward::relu_activation(const ::matrix& z1) {
    return kernel::matrix::map_matrix<relu>(z1);
}

__device__ float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

matrix kernel::feed_forward::relu_activation_backprop(
    const ::matrix& activation_input,
    const ::matrix& a1_gradient) {
    ::matrix z1_mapped
        = kernel::matrix::map_matrix<relu_derivative>(activation_input);
    z1_mapped.element_wise_multiply(a1_gradient);

    return z1_mapped;
}
