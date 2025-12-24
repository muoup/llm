#include "layer_norm.hpp"

#include <cuda_runtime_api.h>

#include <inference/layer_normalize.hpp>
#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>

__global__ void layer_norm_fused_forward_kernel(
    const const_matrix_view input,
    const const_matrix_view gamma,
    const const_matrix_view beta,
    matrix_view normalized_output,
    matrix_view mean_out,
    matrix_view inv_var_out,
    float epsilon) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int cols = input.cols;

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += kernel::matrix::device_get(input, row, col);
    }
    
    float row_sum = kernel::matrix::device::block_reduce_sum(local_sum);
    __shared__ float shared_mean;
    if (tid == 0) shared_mean = row_sum / cols;
    __syncthreads();
    
    float row_mean = shared_mean;

    float local_var_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float val = kernel::matrix::device_get(input, row, col);
        float diff = val - row_mean;
        local_var_sum += diff * diff;
    }
    float row_var_sum = kernel::matrix::device::block_reduce_sum(local_var_sum);
    
    __shared__ float shared_inv_std;
    if (tid == 0) {
        float row_var = row_var_sum / cols;
        shared_inv_std = 1.0f / sqrtf(row_var + epsilon);
        kernel::matrix::device_set(mean_out, row, 0, row_mean);
        kernel::matrix::device_set(inv_var_out, row, 0, shared_inv_std);
    }
    __syncthreads();
    float inv_std = shared_inv_std;

    for (int col = tid; col < cols; col += blockDim.x) {
        float val = kernel::matrix::device_get(input, row, col);
        float g = kernel::matrix::device_get(gamma, 0, col);
        float b = kernel::matrix::device_get(beta, 0, col);
        float norm = (val - row_mean) * inv_std;
        kernel::matrix::device_set(normalized_output, row, col, norm * g + b);
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

    const size_t threads_per_block = 256;
    layer_norm_fused_forward_kernel<<<input.rows, threads_per_block>>>(
        input, gamma, beta, normalized_input, mean, inv_variance, epsilon);

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
