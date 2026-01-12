#include <cuda_runtime_api.h>
#include "kernels/scheduling.hpp"
#include "layer_norm.hpp"

#include <inference/layer_normalize.hpp>
#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix.hpp>
#include <kernels/scheduling.cuh>


__global__ void row_mean(const const_matrix_view input,
                         const matrix_view mean) {
    size_t row_idx = blockIdx.x;

    float* sum = kernel::matrix::device_get_addr(mean, row_idx, 0);
    *sum = 0.0f;

    for (size_t j = 0; j < input.cols; ++j) {
        *sum += kernel::matrix::device_get(input, row_idx, j);
    }

    *sum /= static_cast<float>(input.cols);
}

__global__ void row_inv_variance(const const_matrix_view input,
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
    
    kernel::matrix::device_set(inv_variance, row_idx, 0,
                               1.0f / sqrtf(variance + epsilon));
}

__global__ void normalize_and_scale(const const_matrix_view input,
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
    const LayerNorm &layer,
    float epsilon,
    kernel_stream_t stream) {
    ::matrix normalized_input = matrix::async_allocate(input.rows, input.cols, stream);
    ::matrix mean = matrix::async_allocate(input.rows, 1, stream);
    ::matrix inv_variance = matrix::async_allocate(input.rows, 1, stream);

    row_mean<<<input.rows, 1, 0, get_kernel_stream(stream)>>>(input, mean);
    row_inv_variance<<<input.rows, 1, 0, get_kernel_stream(stream)>>>(input, mean, inv_variance, epsilon);

    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (input.cols + threads_per_block.x - 1) / threads_per_block.x,
        (input.rows + threads_per_block.y - 1) / threads_per_block.y);

    normalize_and_scale<<<blocks, threads_per_block, 0, get_kernel_stream(stream)>>>(
        input, mean, inv_variance, layer.gamma, layer.beta, normalized_input);

    return { .normalized = std::move(normalized_input),
             .mean = std::move(mean),
             .inv_variance = std::move(inv_variance) };
}

__global__ void layer_norm_grad_input_kernel(
    const const_matrix_view mean,
    const const_matrix_view gamma,
    const const_matrix_view inv_variance,
    const const_matrix_view layer_input,
    const const_matrix_view grad_normalized,
    matrix_view grad_input) {
    
    // One block per row
    const size_t row = blockIdx.x;
    if (row >= layer_input.rows) return;

    const size_t cols = layer_input.cols;
    const float row_mean = kernel::matrix::device_get(mean, row, 0);
    const float row_inv_var = kernel::matrix::device_get(inv_variance, row, 0);

    float local_d_norm_sum = 0.0f;
    float local_d_norm_dot_x_norm = 0.0f;

    // First pass: Compute reductions
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        float grad_norm_val = kernel::matrix::device_get(grad_normalized, row, col);
        float layer_input_val = kernel::matrix::device_get(layer_input, row, col);
        float gamma_val = kernel::matrix::device_get(gamma, 0, col);
        
        float normalized_val = (layer_input_val - row_mean) * row_inv_var;
        float d_norm = grad_norm_val * gamma_val;

        local_d_norm_sum += d_norm;
        local_d_norm_dot_x_norm += d_norm * normalized_val;
    }

    // Block-wide reductions
    float d_norm_sum = kernel::matrix::device::block_reduce_sum(local_d_norm_sum);
    float d_norm_dot_x_norm = kernel::matrix::device::block_reduce_sum(local_d_norm_dot_x_norm);

    __shared__ float s_d_norm_sum;
    __shared__ float s_d_norm_dot_x_norm;

    if (threadIdx.x == 0) {
        s_d_norm_sum = d_norm_sum;
        s_d_norm_dot_x_norm = d_norm_dot_x_norm;
    }
    __syncthreads();

    d_norm_sum = s_d_norm_sum;
    d_norm_dot_x_norm = s_d_norm_dot_x_norm;

    // Second pass: Compute grad_input
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        float grad_norm_val = kernel::matrix::device_get(grad_normalized, row, col);
        float layer_input_val = kernel::matrix::device_get(layer_input, row, col);
        float gamma_val = kernel::matrix::device_get(gamma, 0, col);

        float normalized_val = (layer_input_val - row_mean) * row_inv_var;
        float d_norm = grad_norm_val * gamma_val;

        float grad_in = (static_cast<float>(cols) * d_norm) - d_norm_sum
                        - (normalized_val * d_norm_dot_x_norm);
        grad_in *= row_inv_var / static_cast<float>(cols);

        kernel::matrix::device_set(grad_input, row, col, grad_in);
    }
}

__global__ void layer_norm_grad_params_kernel(
    const const_matrix_view mean,
    const const_matrix_view inv_variance,
    const const_matrix_view layer_input,
    const const_matrix_view grad_normalized,
    matrix_view grad_beta,
    matrix_view grad_gamma) {

    // One thread per column (feature)
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= layer_input.cols) return;

    float sum_beta = 0.0f;
    float sum_gamma = 0.0f;
    const size_t rows = layer_input.rows;

    // Loop over rows - Coalesced reads because adjacent threads read adjacent columns
    for (size_t row = 0; row < rows; ++row) {
        float grad_norm_val = kernel::matrix::device_get(grad_normalized, row, col);
        float input_val = kernel::matrix::device_get(layer_input, row, col);
        float row_mean = kernel::matrix::device_get(mean, row, 0);
        float row_inv_var = kernel::matrix::device_get(inv_variance, row, 0);

        float normalized_val = (input_val - row_mean) * row_inv_var;

        sum_beta += grad_norm_val;
        sum_gamma += grad_norm_val * normalized_val;
    }

    kernel::matrix::device_set(grad_beta, 0, col, sum_beta);
    kernel::matrix::device_set(grad_gamma, 0, col, sum_gamma);
}

kernel::layer_norm::LayerNormGradients
kernel::layer_norm::layer_normalization_backward(
    const ::matrix& layer_input,
    const ::matrix& gamma,
    const ::matrix& beta,
    const ::matrix& mean,
    const ::matrix& inv_variance,
    const ::matrix& grad_normalized,
    float epsilon,
    kernel_stream_t stream) {
    ::matrix grad_input = matrix::async_allocate(layer_input.rows, layer_input.cols, stream);
    ::matrix grad_gamma = matrix::async_allocate(gamma.rows, gamma.cols, stream);
    ::matrix grad_beta = matrix::async_allocate(beta.rows, beta.cols, stream);

    constexpr size_t threads_per_block = 256;

    // 1. Compute grad_input
    // Grid size = number of rows (one block per row)
    layer_norm_grad_input_kernel<<<layer_input.rows, threads_per_block, 0, get_kernel_stream(stream)>>>(
        mean, gamma, inv_variance, layer_input, grad_normalized, grad_input);
    CHECK_ERRORS("layer_normalization_backward: grad_input kernel launch");

    // 2. Compute grad_params (beta/gamma)
    // Grid size = sufficient to cover all columns (one thread per column)
    size_t num_blocks_params = (layer_input.cols + threads_per_block - 1) / threads_per_block;
    layer_norm_grad_params_kernel<<<num_blocks_params, threads_per_block, 0, get_kernel_stream(stream)>>>(
        mean, inv_variance, layer_input, grad_normalized, grad_beta, grad_gamma);
    CHECK_ERRORS("layer_normalization_backward: grad_params kernel launch");

    return { .grad_input = std::move(grad_input),
             .grad_gamma = std::move(grad_gamma),
             .grad_beta = std::move(grad_beta) };
}

__global__ void row_rms(const const_matrix_view input,
                       const matrix_view inv_rms,
                       float epsilon) {
    size_t row_idx = blockIdx.x;

    float* inv_rms_ptr = kernel::matrix::device_get_addr(inv_rms, row_idx, 0);
    float sum_sq = 0.0f;

    for (size_t j = 0; j < input.cols; ++j) {
        float val = kernel::matrix::device_get(input, row_idx, j);
        sum_sq += val * val;
    }

    float mean_sq = sum_sq / static_cast<float>(input.cols);
    *inv_rms_ptr = 1.0f / sqrtf(mean_sq + epsilon);
}

__global__ void rms_normalize_and_scale(const const_matrix_view input,
                                       const const_matrix_view inv_rms,
                                       const const_matrix_view gamma,
                                       matrix_view normalized_input) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < normalized_input.cols && row < normalized_input.rows) {
        float input_val = kernel::matrix::device_get(input, row, col);
        float inv_rms_val = kernel::matrix::device_get(inv_rms, row, 0);
        float gamma_val = kernel::matrix::device_get(gamma, 0, col);

        float normalized = input_val * inv_rms_val;
        float scaled = normalized * gamma_val;
        kernel::matrix::device_set(normalized_input, row, col, scaled);
    }
}

kernel::layer_norm::RMSNormResult kernel::layer_norm::rms_normalization(
    const ::matrix& input,
    const ::matrix& gamma,
    float epsilon,
    kernel_stream_t stream) {
    ::matrix normalized_input = matrix::async_allocate(input.rows, input.cols, stream);
    ::matrix inv_rms = matrix::async_allocate(input.rows, 1, stream);

    row_rms<<<input.rows, 1, 0, get_kernel_stream(stream)>>>(input, inv_rms, epsilon);

    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (input.cols + threads_per_block.x - 1) / threads_per_block.x,
        (input.rows + threads_per_block.y - 1) / threads_per_block.y);

    rms_normalize_and_scale<<<blocks, threads_per_block, 0, get_kernel_stream(stream)>>>(
        input, inv_rms, gamma, normalized_input);

    return { .normalized = std::move(normalized_input),
             .inv_rms = std::move(inv_rms) };
}

__global__ void rms_norm_grad_input_kernel(
    const const_matrix_view inv_rms,
    const const_matrix_view gamma,
    const const_matrix_view layer_input,
    const const_matrix_view grad_normalized,
    matrix_view grad_input) {
    
    const size_t row = blockIdx.x;
    if (row >= layer_input.rows) return;

    const size_t cols = layer_input.cols;
    const float row_inv_rms = kernel::matrix::device_get(inv_rms, row, 0);
    const float inv_rms_sq = row_inv_rms * row_inv_rms;

    float local_d_norm_dot_x = 0.0f;

    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        float grad_norm_val = kernel::matrix::device_get(grad_normalized, row, col);
        float layer_input_val = kernel::matrix::device_get(layer_input, row, col);
        float gamma_val = kernel::matrix::device_get(gamma, 0, col);
        
        float d_norm = grad_norm_val * gamma_val;
        local_d_norm_dot_x += d_norm * layer_input_val;
    }

    float d_norm_dot_x = kernel::matrix::device::block_reduce_sum(local_d_norm_dot_x);

    __shared__ float s_d_norm_dot_x;
    if (threadIdx.x == 0) {
        s_d_norm_dot_x = d_norm_dot_x;
    }
    __syncthreads();

    d_norm_dot_x = s_d_norm_dot_x;

    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        float grad_norm_val = kernel::matrix::device_get(grad_normalized, row, col);
        float layer_input_val = kernel::matrix::device_get(layer_input, row, col);
        float gamma_val = kernel::matrix::device_get(gamma, 0, col);

        float d_norm = grad_norm_val * gamma_val;

        float grad_in = d_norm - (layer_input_val * inv_rms_sq * d_norm_dot_x);
        grad_in *= row_inv_rms / static_cast<float>(cols);

        kernel::matrix::device_set(grad_input, row, col, grad_in);
    }
}

__global__ void rms_norm_grad_gamma_kernel(
    const const_matrix_view inv_rms,
    const const_matrix_view layer_input,
    const const_matrix_view grad_normalized,
    matrix_view grad_gamma) {

    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= layer_input.cols) return;

    float sum_gamma = 0.0f;
    const size_t rows = layer_input.rows;

    for (size_t row = 0; row < rows; ++row) {
        float grad_norm_val = kernel::matrix::device_get(grad_normalized, row, col);
        float input_val = kernel::matrix::device_get(layer_input, row, col);
        float row_inv_rms = kernel::matrix::device_get(inv_rms, row, 0);

        float normalized_val = input_val * row_inv_rms;
        sum_gamma += grad_norm_val * normalized_val;
    }

    kernel::matrix::device_set(grad_gamma, 0, col, sum_gamma);
}

kernel::layer_norm::RMSNormGradients
kernel::layer_norm::rms_normalization_backward(
    const ::matrix& layer_input,
    const ::matrix& gamma,
    const ::matrix& inv_rms,
    const ::matrix& grad_normalized,
    float epsilon,
    kernel_stream_t stream) {
    ::matrix grad_input = matrix::async_allocate(layer_input.rows, layer_input.cols, stream);
    ::matrix grad_gamma = matrix::async_allocate(gamma.rows, gamma.cols, stream);

    constexpr size_t threads_per_block = 256;

    rms_norm_grad_input_kernel<<<layer_input.rows, threads_per_block, 0, get_kernel_stream(stream)>>>(
        inv_rms, gamma, layer_input, grad_normalized, grad_input);
    CHECK_ERRORS("rms_normalization_backward: grad_input kernel launch");

    size_t num_blocks_params = (layer_input.cols + threads_per_block - 1) / threads_per_block;
    rms_norm_grad_gamma_kernel<<<num_blocks_params, threads_per_block, 0, get_kernel_stream(stream)>>>(
        inv_rms, layer_input, grad_normalized, grad_gamma);
    CHECK_ERRORS("rms_normalization_backward: grad_gamma kernel launch");

    return { .grad_input = std::move(grad_input),
             .grad_gamma = std::move(grad_gamma) };
}

