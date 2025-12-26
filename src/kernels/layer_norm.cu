#include <cuda_runtime_api.h>
#include "kernels/optimizer.hpp"
#include "kernels/scheduling.hpp"
#include "layer_norm.hpp"
#include "util/logger.hpp"

#include <inference/layer_normalize.hpp>
#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <kernels/scheduling.cuh>

// matrix cpu_version(const matrix& input,
//                    const matrix& gamma,
//                    const matrix& beta,
//                    float epsilon) {
//     matrix normalized_input(input.rows, input.cols);
//     matrix mean(input.rows, 1);
//     matrix inv_variance(input.rows, 1);

//     for (size_t i = 0; i < input.rows; ++i) {
//         float row_mean = 0.0f;
//         for (size_t j = 0; j < input.cols; ++j) {
//             row_mean += input.get(i, j);
//         }
//         row_mean /= static_cast<float>(input.cols);
//         mean.set(i, 0, row_mean);

//         float variance = 0.0f;
//         for (size_t j = 0; j < input.cols; ++j) {
//             float diff = input.get(i, j) - row_mean;
//             variance += diff * diff;
//         }
//         variance /= static_cast<float>(input.cols);
//         inv_variance.set(i, 0, 1.0f / std::sqrt(variance + epsilon));

//         for (size_t j = 0; j < input.cols; ++j) {
//             float normalized
//                 = (input.get(i, j) - row_mean) * inv_variance.get(i, 0);
//             float scaled = normalized * gamma.get(0, j) + beta.get(0, j);
//             normalized_input.set(i, j, scaled);
//         }
//     }

//     return normalized_input;
// }

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
    // std::printf("Mean: %f | Variance: %f\n", row_mean, variance);
    
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
    ::matrix normalized_input = matrix::async_allocate(input.rows, input.cols, layer.streams[0]);
    ::matrix mean = matrix::async_allocate(input.rows, 1, layer.streams[1]);
    ::matrix inv_variance = matrix::async_allocate(input.rows, 1, layer.streams[2]);
    kernel::wait_for_streams(layer.streams[0], layer.streams[1], layer.streams[2]);

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

// void cpu_snippet() {
//     for (size_t i = 0; i < layer_input.rows; i++) {
//             float row_mean = mean.get(i, 0);
//             float row_inv_var = inv_variance.get(i, 0);

//             float d_norm_sum = 0.0f;
//             float d_norm_dot_x_norm = 0.0f;

//             for (size_t j = 0; j < layer_input.cols; j++) {
//                 float grad_norm_val = grad_normalized.get(i, j);
//                 float normalized_val
//                     = (layer_input.get(i, j) - row_mean) * row_inv_var;

//                 grad_beta.offset(0, j, grad_norm_val);
//                 grad_gamma.offset(0, j, grad_norm_val * normalized_val);

//                 float d_norm = grad_norm_val * gamma.get(0, j);
//                 d_norm_sum += d_norm;
//                 d_norm_dot_x_norm += d_norm * normalized_val;
//             }

//             for (size_t j = 0; j < layer_input.cols; j++) {
//                 float normalized_val
//                     = (layer_input.get(i, j) - row_mean) * row_inv_var;
//                 float d_norm = grad_normalized.get(i, j) * gamma.get(0, j);

//                 float grad_in = (dimensions * d_norm) - d_norm_sum
//                                 - (normalized_val * d_norm_dot_x_norm);
//                 grad_in *= row_inv_var / static_cast<float>(dimensions);

//                 grad_input.set(i, j, grad_in);
//             }
//         }
// }

__global__ void layer_norm_backward_kernel(
    const const_matrix_view mean,
    const const_matrix_view gamma,
    const const_matrix_view inv_variance,
    const const_matrix_view layer_input,
    const const_matrix_view grad_normalized,
    matrix_view grad_beta,
    matrix_view grad_gamma,
    matrix_view grad_input) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= layer_input.rows) {
        return;
    }

    size_t dimensions = gamma.cols;

    float row_mean = kernel::matrix::device_get(mean, row, 0);
    float row_inv_var = kernel::matrix::device_get(inv_variance, row, 0);

    float d_norm_sum = 0.0f;
    float d_norm_dot_x_norm = 0.0f;

    for (size_t col = 0; col < layer_input.cols; col++) {
        float grad_norm_val
            = kernel::matrix::device_get(grad_normalized, row, col);
        float layer_input_val
            = kernel::matrix::device_get(layer_input, row, col);
        float normalized_val = (layer_input_val - row_mean) * row_inv_var;

        kernel::matrix::device_offset_elem_atomic(grad_beta, 0, col,
                                                  grad_norm_val);
        kernel::matrix::device_offset_elem_atomic(
            grad_gamma, 0, col, grad_norm_val * normalized_val);

        float d_norm
            = grad_norm_val * kernel::matrix::device_get(gamma, 0, col);
        d_norm_sum += d_norm;
        d_norm_dot_x_norm += d_norm * normalized_val;
    }

    for (size_t col = 0; col < layer_input.cols; col++) {
        float layer_input_val
            = kernel::matrix::device_get(layer_input, row, col);
        float grad_norm_val
            = kernel::matrix::device_get(grad_normalized, row, col);
        float gamma_val = kernel::matrix::device_get(gamma, 0, col);

        float normalized_val = (layer_input_val - row_mean) * row_inv_var;
        float d_norm = grad_norm_val * gamma_val;

        float grad_in = (dimensions * d_norm) - d_norm_sum
                        - (normalized_val * d_norm_dot_x_norm);
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
    float epsilon,
    kernel_stream_t stream) {
    ::matrix grad_input = matrix::async_allocate(layer_input.rows, layer_input.cols);
    ::matrix grad_gamma = matrix::async_allocate(gamma.rows, gamma.cols);
    ::matrix grad_beta = matrix::async_allocate(beta.rows, beta.cols);

    constexpr size_t threads_per_block = 256;
    size_t num_blocks
        = (layer_input.rows + threads_per_block - 1) / threads_per_block;

    layer_norm_backward_kernel<<<num_blocks, threads_per_block, 0, get_kernel_stream(stream)>>>(
        mean, gamma, inv_variance, layer_input, grad_normalized, grad_beta,
        grad_gamma, grad_input);
    CHECK_ERRORS("layer_normalization_backward: kernel launch");

    return { .grad_input = std::move(grad_input),
             .grad_gamma = std::move(grad_gamma),
             .grad_beta = std::move(grad_beta) };
}
