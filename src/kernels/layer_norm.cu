#include "layer_norm.hpp"

#include "matrix_global_kernels.hpp"

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

    float* sum = &mean[row_idx * mean_stride];
    *sum = 0.0f;

    for (size_t j = 0; j < cols; ++j) {
        *sum += input[row_idx * stride + j];
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

    float row_mean = mean[row_idx * mean_stride];
    float variance = 0.0f;

    for (size_t col = 0; col < cols; ++col) {
        float diff = kernel::matrix::device_get(input, stride, rows, cols, row_idx, col) - row_mean;
        variance += diff * diff;
    }

    variance /= static_cast<float>(cols);
    inv_variance[row_idx * inv_variance_stride]
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
    size_t col_idx = threadIdx.x;

    if (col_idx < cols) {
        float normalized
            = (input[row_idx * stride + col_idx] - mean[row_idx * mean_stride])
              * inv_variance[row_idx * inv_variance_stride];
        float scaled = normalized * gamma[col_idx * gamma_stride]
                       + beta[col_idx * beta_stride];
        output[row_idx * output_stride + col_idx] = scaled;
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

    normalize_and_scale<<<input.rows, input.cols>>>(
        input.data, input.rows, input.cols, input.stride, mean.data,
        mean.stride, inv_variance.data, inv_variance.stride, gamma.data,
        gamma.stride, beta.data, beta.stride, normalized_input.data,
        normalized_input.stride);

    return { .normalized = std::move(normalized_input),
             .mean = std::move(mean),
             .inv_variance = std::move(inv_variance) };
}
