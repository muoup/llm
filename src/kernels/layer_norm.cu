#include "inference/layer_normalize.hpp"
#include "layer_norm.hpp"

#include "matrix_device_kernels.cuh"

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

__global__ void row_mean(const float* input,
                         size_t rows,
                         size_t cols,
                         size_t stride,
                         float* mean,
                         size_t mean_stride) {
    size_t row_idx = blockIdx.x;

    float* sum = &mean[row_idx + 0 * mean_stride];
    *sum = 0.0f;

    for (size_t j = 0; j < cols; ++j) {
        *sum += input[row_idx + j * stride];
    }

    *sum /= static_cast<float>(cols);
}

__global__ void row_variance(const float* input,
                             size_t rows,
                             size_t cols,
                             size_t stride,
                             const float* mean,
                             size_t mean_stride,
                             float* inv_variance,
                             size_t inv_variance_stride,
                             float epsilon) {
    size_t row_idx = blockIdx.x;

    float row_mean = mean[row_idx + 0 * mean_stride];
    float variance = 0.0f;

    for (size_t col = 0; col < cols; ++col) {
        float diff = kernel::matrix::device_get(input, stride, row_idx, col)
                     - row_mean;
        variance += diff * diff;
    }

    variance /= static_cast<float>(cols);
    inv_variance[row_idx + 0 * inv_variance_stride]
        = 1.0f / sqrtf(variance + epsilon);
}

__global__ void normalize_and_scale(const float* input,
                                    size_t rows,
                                    size_t cols,
                                    size_t stride,
                                    const float* mean,
                                    size_t mean_stride,
                                    const float* inv_variance,
                                    size_t inv_variance_stride,
                                    const float* gamma,
                                    size_t gamma_stride,
                                    const float* beta,
                                    size_t beta_stride,
                                    float* output,
                                    size_t output_stride) {
    size_t row_idx = blockIdx.x;
    size_t col_idx = threadIdx.x + blockDim.x * blockIdx.y;

    if (col_idx < cols) {
        float normalized = (input[row_idx + col_idx * stride]
                            - mean[row_idx + 0 * mean_stride])
                           * inv_variance[row_idx + 0 * inv_variance_stride];
        float scaled = normalized * gamma[0 + col_idx * gamma_stride]
                       + beta[0 + col_idx * beta_stride];
        output[row_idx + col_idx * output_stride] = scaled;
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

    row_mean<<<input.rows, 1>>>(input.data, input.rows, input.cols,
                                input.stride, mean.data, mean.stride);

    row_variance<<<input.rows, 1>>>(
        input.data, input.rows, input.cols, input.stride, mean.data,
        mean.stride, inv_variance.data, inv_variance.stride, epsilon);

    const size_t threads_per_block = 256;
    const size_t blocks_y
        = (input.cols + threads_per_block - 1) / threads_per_block;
    dim3 gridSize(input.rows, blocks_y);

    normalize_and_scale<<<gridSize, threads_per_block>>>(
        input.data, input.rows, input.cols, input.stride, mean.data,
        mean.stride, inv_variance.data, inv_variance.stride, gamma.data,
        gamma.stride, beta.data, beta.stride, normalized_input.data,
        normalized_input.stride);

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

__global__ void layer_norm_backward_kernel(const LayerNorm& layer,
                                           const matrix& mean,
                                           const matrix& inv_variance,
                                           const matrix& layer_input,
                                           const matrix& grad_output,
                                           matrix& grad_beta,
                                           matrix& grad_gamma,
                                           matrix& grad_input) {
    size_t i = blockIdx.x;

    float row_mean = kernel::matrix::device_get(mean.data, mean.stride, i, 0);
    float row_inv_var = kernel::matrix::device_get(inv_variance.data,
                                                   inv_variance.stride, i, 0);

    float d_norm_sum = 0.0f;
    float d_norm_dot_x_norm = 0.0f;

    for (size_t j = 0; j < layer_input.cols; j++) {
        float grad_norm_val = kernel::matrix::device_get(
            grad_output.data, grad_output.stride, i, j);
        float layer_input_val = kernel::matrix::device_get(
            layer_input.data, layer_input.stride, i, j);
        float normalized_val = (layer_input_val - row_mean) * row_inv_var;

        kernel::matrix::device_offset_elem(grad_beta.data, grad_beta.stride, 0,
                                           j, grad_norm_val);
        kernel::matrix::device_offset_elem(grad_gamma.data, grad_gamma.stride,
                                           0, j,
                                           grad_norm_val * normalized_val);

        float d_norm = grad_norm_val
                       * kernel::matrix::device_get(layer_input.data,
                                                    layer_input.stride, 0, j);
        d_norm_sum += d_norm;
        d_norm_dot_x_norm += d_norm * normalized_val;
    }

    for (size_t j = 0; j < layer_input.cols; j++) {
        float layer_input_val = kernel::matrix::device_get(
            layer_input.data, layer_input.stride, i, j);
        float grad_norm_val = kernel::matrix::device_get(
            grad_output.data, grad_output.stride, i, j);
        float gamma_val = kernel::matrix::device_get(layer_input.data,
                                                     layer_input.stride, 0, j);

        float normalized_val = (layer_input_val - row_mean) * row_inv_var;
        float d_norm = grad_norm_val * gamma_val;

        float grad_in = (layer.dimensions * d_norm) - d_norm_sum
                        - (normalized_val * d_norm_dot_x_norm);
        grad_in *= row_inv_var / static_cast<float>(layer.dimensions);

        kernel::matrix::device_set(grad_input.data, grad_input.stride, i, j,
                                   grad_in);
    }
}

kernel::layer_norm::LayerNormGradients
kernel::layer_norm::layer_normalization_backward(const ::LayerNorm& layer,
                                                 const ::matrix& layer_input,
                                                 const ::matrix& gamma,
                                                 const ::matrix& beta,
                                                 const ::matrix& mean,
                                                 const ::matrix& inv_variance,
                                                 const ::matrix& grad_output,
                                                 float epsilon) {
    ::matrix grad_input(layer_input.rows, layer_input.cols);
    ::matrix grad_gamma(1, layer_input.cols);
    ::matrix grad_beta(1, layer_input.cols);

    layer_norm_backward_kernel<<<layer_input.rows, 1>>>(
        layer, mean, inv_variance, layer_input, grad_output, grad_beta,
        grad_gamma, grad_input);

    return { .grad_input = std::move(grad_input),
             .grad_gamma = std::move(grad_gamma),
             .grad_beta = std::move(grad_beta) };
}
