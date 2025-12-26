#include "layer_norm.hpp"

#include <cuda_runtime_api.h>

#include <inference/layer_normalize.hpp>
#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <kernels/optimizer.hpp>

static __global__ void row_mean(const const_matrix_view input,
                         const matrix_view mean) {
    size_t row_idx = blockIdx.x;

    float* sum = kernel::matrix::device_get_addr(mean, row_idx, 0);
    *sum = 0.0f;

    for (size_t j = 0; j < input.cols; ++j) {
        *sum += kernel::matrix::device_get(input, row_idx, j);
    }

    *sum /= static_cast<float>(input.cols);
}

static __global__ void row_inv_variance(const const_matrix_view input,
                                 const const_matrix_view mean,
                                 const matrix_view inv_variance,
                                 float epsilon) {
    size_t row_idx = blockIdx.x;

    float row_mean = kernel::matrix::device_get(mean, row_idx, 0);
    float variance = 0.0f;

    for (size_t col = 0; col < input.cols; ++col) {
        float diff = kernel::matrix::device_get(input, row_idx, col) - row_mean;
        variance += diff * diff;
    }

    variance /= static_cast<float>(input.cols);
    // std::printf("Mean: %f | Variance: %f\n", row_mean, variance);
    
    kernel::matrix::device_set(inv_variance, row_idx, 0,
                               1.0f / sqrtf(variance + epsilon));
}

static __global__ void normalize_and_scale(const const_matrix_view input,
                                    const const_matrix_view mean,
                                    const const_matrix_view inv_variance,
                                    const const_matrix_view gamma,
                                    const const_matrix_view beta,
                                    matrix_view normalized_input) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < normalized_input.cols && row < normalized_input.rows) {
        float input_val = kernel::matrix::device_get(input, row, col);
        float inv_variance_val
            = kernel::matrix::device_get(inv_variance, row, 0);
        float gamma_val = kernel::matrix::device_get(gamma, 0, col);
        float beta_val = kernel::matrix::device_get(beta, 0, col);
        float row_mean = kernel::matrix::device_get(mean, row, 0);

        float normalized = (input_val - row_mean) * inv_variance_val;
        float scaled = normalized * gamma_val + beta_val;
        kernel::matrix::device_set(normalized_input, row, col, scaled);
    }
}

kernel::layer_norm::LayerNormResult kernel::layer_norm::layer_normalization(
    const ::matrix& input,
    const ::matrix& gamma,
    const ::matrix& beta,
    float epsilon) {
    ::matrix normalized_input(input.rows, input.cols);
    ::matrix mean(input.rows, 1);
    ::matrix inv_variance(input.rows, 1);

    row_mean<<<input.rows, 1>>>(input, mean);
    kernel::optimizer::wait_for_operations();

    row_inv_variance<<<input.rows, 1>>>(input, mean, inv_variance, epsilon);
    kernel::optimizer::wait_for_operations();

    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (input.cols + threads_per_block.x - 1) / threads_per_block.x,
        (input.rows + threads_per_block.y - 1) / threads_per_block.y);

    normalize_and_scale<<<blocks, threads_per_block>>>(
        input, mean, inv_variance, gamma, beta, normalized_input);
    kernel::optimizer::wait_for_operations();

    return { .normalized = std::move(normalized_input),
             .mean = std::move(mean),
             .inv_variance = std::move(inv_variance) };
}

__global__ void layer_norm_backward_optimized_kernel(
    const const_matrix_view mean,
    const const_matrix_view gamma,
    const const_matrix_view inv_variance,
    const const_matrix_view layer_input,
    const const_matrix_view grad_normalized,
    matrix_view grad_beta,
    matrix_view grad_gamma,
    matrix_view grad_input) {
    
    const size_t row = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t dimensions = layer_input.cols;

    if (row >= layer_input.rows) return;

    float row_mean = kernel::matrix::device_get(mean, row, 0);
    float row_inv_var = kernel::matrix::device_get(inv_variance, row, 0);

    float local_d_norm_sum = 0.0f;
    float local_d_norm_dot_x_norm = 0.0f;

    for (size_t col = tid; col < dimensions; col += blockDim.x) {
        float grad_norm_val = kernel::matrix::device_get(grad_normalized, row, col);
        float layer_input_val = kernel::matrix::device_get(layer_input, row, col);
        float gamma_val = kernel::matrix::device_get(gamma, 0, col);
        
        float normalized_val = (layer_input_val - row_mean) * row_inv_var;
        
        // These still need to be atomic because they are shared across all rows
        kernel::matrix::device_offset_elem_atomic(grad_beta, 0, col, grad_norm_val);
        kernel::matrix::device_offset_elem_atomic(grad_gamma, 0, col, grad_norm_val * normalized_val);

        float d_norm = grad_norm_val * gamma_val;
        local_d_norm_sum += d_norm;
        local_d_norm_dot_x_norm += d_norm * normalized_val;
    }

    float d_norm_sum = kernel::matrix::device::block_reduce_sum(local_d_norm_sum);
    float d_norm_dot_x_norm = kernel::matrix::device::block_reduce_sum(local_d_norm_dot_x_norm);

    __shared__ float s_d_norm_sum, s_d_norm_dot_x_norm;
    if (tid == 0) {
        s_d_norm_sum = d_norm_sum;
        s_d_norm_dot_x_norm = d_norm_dot_x_norm;
    }
    __syncthreads();

    for (size_t col = tid; col < dimensions; col += blockDim.x) {
        float layer_input_val = kernel::matrix::device_get(layer_input, row, col);
        float grad_norm_val = kernel::matrix::device_get(grad_normalized, row, col);
        float gamma_val = kernel::matrix::device_get(gamma, 0, col);

        float normalized_val = (layer_input_val - row_mean) * row_inv_var;
        float d_norm = grad_norm_val * gamma_val;

        float grad_in = (dimensions * d_norm) - s_d_norm_sum - (normalized_val * s_d_norm_dot_x_norm);
        grad_in *= row_inv_var / static_cast<float>(dimensions);

        kernel::matrix::device_set(grad_input, row, col, grad_in);
    }
}

kernel::layer_norm::LayerNormGradients
kernel::layer_norm::layer_normalization_backward(
    const ::matrix& layer_input,
    const ::matrix& gamma,
    const ::matrix& beta,
    const ::matrix& mean,
    const ::matrix& inv_variance,
    const ::matrix& grad_normalized,
    float epsilon) {
    ::matrix grad_input(layer_input.rows, layer_input.cols);
    ::matrix grad_gamma(gamma.rows, gamma.cols);
    ::matrix grad_beta(beta.rows, beta.cols);

    constexpr size_t threads_per_block = 256;
    layer_norm_backward_optimized_kernel<<<layer_input.rows, threads_per_block>>>(
        mean, gamma, inv_variance, layer_input, grad_normalized, grad_beta,
        grad_gamma, grad_input);

    return { .grad_input = std::move(grad_input),
             .grad_gamma = std::move(grad_gamma),
             .grad_beta = std::move(grad_beta) };
}
