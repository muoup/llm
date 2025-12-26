#include "feed_forward.hpp"
#include "kernels/matrix_kernels.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/optimizer.hpp>
#include <kernels/scheduling.cuh>

#include <cublas_v2.h>

__global__ void kernel_add_bias(const matrix_view matrix,
                                const const_matrix_view bias) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrix.rows && col < matrix.cols) {
        auto val = kernel::matrix::device_get(bias, 0, col);
        kernel::matrix::device_offset_elem(matrix, row, col, val);
    }
}

void kernel::feed_forward::add_bias(::matrix& mat, const ::matrix& row_vec, kernel_stream_t stream) {
    dim3 block_size(16, 16);
    dim3 grid_size((mat.cols + block_size.x - 1) / block_size.x,
                   (mat.rows + block_size.y - 1) / block_size.y);

    kernel_add_bias<<<grid_size, block_size, 0, from_kernel_stream(stream)>>>(mat, row_vec);
}

__global__ void sum_columns_kernel(const const_matrix_view base,
                                   const matrix_view result) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < base.cols) {
        float sum = 0.0f;

        for (int row = 0; row < base.rows; ++row) {
            sum += kernel::matrix::device_get(base, row, col);
        }

        kernel::matrix::device_set(result, 0, col, sum);
    }
}

matrix kernel::feed_forward::sum_columns(const ::matrix& mat, kernel_stream_t stream) {
    ::matrix result(1, mat.cols);

    size_t threads_per_block = 256;
    size_t num_blocks = (mat.cols + threads_per_block - 1) / threads_per_block;

    sum_columns_kernel<<<num_blocks, threads_per_block, 0, from_kernel_stream(stream)>>>(mat, result);
    return result;
}

__device__ float leaky_relu(float x) {
    return x > 0 ? x : x * 0.01f;
}

matrix kernel::feed_forward::leaky_relu_activation(const ::matrix& z1, kernel_stream_t stream) {
    return kernel::matrix::map_matrix<leaky_relu>(z1, stream);
}

__device__ float leaky_relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.01f;
}

matrix kernel::feed_forward::leaky_relu_activation_backprop(
    const ::matrix& activation_input,
    const ::matrix& a1_gradient,
    kernel_stream_t stream) {
    ::matrix z1_mapped
        = kernel::matrix::map_matrix<leaky_relu_derivative>(activation_input, stream);
    kernel::matrix::element_wise_multiply(z1_mapped, a1_gradient, stream);

    return z1_mapped;
}
