#include "optimizer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <kernels/scheduling.cuh>
#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>

constexpr auto NORM_CLIP_MAX_MAG = 2.5f;

__global__ void norm_clip_kernel(matrix_view gradient,
                                 float* sum_sq_ptr,
                                 size_t total_elements) {
    float sum_of_squares = *sum_sq_ptr;
    float mag = sum_of_squares / (float)total_elements;

    if (mag > NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG) {
        float scale = sqrtf((NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG) / mag);

        size_t row = blockIdx.x * blockDim.x + threadIdx.x;
        size_t col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < gradient.rows && col < gradient.cols) {
            float val = kernel::matrix::device_get(gradient, row, col);
            kernel::matrix::device_set(gradient, row, col, val * scale);
        }
    }
}

void kernel::optimizer::norm_clip(::matrix& gradient, kernel_stream_t stream) {
    float_device_ptr_t sum_sq_ptr
        = kernel::matrix::sum_of_squares(gradient, stream);

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (gradient.rows + threads_per_block.x - 1) / threads_per_block.x,
        (gradient.cols + threads_per_block.y - 1) / threads_per_block.y);

    norm_clip_kernel<<<blocks, threads_per_block, 0,
                       get_kernel_stream(stream)>>>(
        gradient, (float*)sum_sq_ptr, gradient.size());
    CHECK_ERRORS("After norm_clip_kernel");
}

static __global__ void regularize_gradient(const matrix_view gradient,
                                           const const_matrix_view parameters,
                                           float* sum_sq_ptr,
                                           size_t total_elements) {
    constexpr float regularization_strength = 0.0001f;
    float sum_of_squares = *sum_sq_ptr;
    float mag = sum_of_squares / (float)total_elements;

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= gradient.rows || col >= gradient.cols) {
        return;
    }

    float device_value = kernel::matrix::device_get(gradient, row, col);

    if (mag > NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG) {
        device_value *= (NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG) / mag;
    }

    float param_value = kernel::matrix::device_get(parameters, row, col);
    float regularization = 2 * regularization_strength * param_value;
    kernel::matrix::device_set(gradient, row, col,
                               device_value + regularization);
}

void kernel::optimizer::regularize_weight_gradient(::matrix& gradient,
                                                   const ::matrix& parameters,
                                                   kernel_stream_t stream) {
    MATRIX_ASSERT(
        gradient.rows == parameters.rows && gradient.cols == parameters.cols,
        "Dimension mismatch in regularize_gradient");

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (gradient.rows + threads_per_block.x - 1) / threads_per_block.x,
        (gradient.cols + threads_per_block.y - 1) / threads_per_block.y);

    norm_clip(gradient, stream);
    float_device_ptr_t sum_sq_ptr
        = kernel::matrix::sum_of_squares(gradient, stream);
    regularize_gradient<<<blocks, threads_per_block, 0,
                          get_kernel_stream(stream)>>>(
        gradient, parameters, (float*)sum_sq_ptr, gradient.size());
    CHECK_ERRORS("After regularize_gradient");
}

void kernel::optimizer::adjust_parameter_matrix(::matrix& adjust,
                                                ::matrix& gradient,
                                                float learning_rate,
                                                kernel_stream_t stream) {
    MATRIX_ASSERT(adjust.rows == gradient.rows && adjust.cols == gradient.cols,
                  "Dimension mismatch in adjust_parameter_matrix");

    dim3 threads_per_block(16, 16);
    dim3 blocks((adjust.rows + threads_per_block.x - 1) / threads_per_block.x,
                (adjust.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel::matrix::add_scaled(adjust, gradient, -learning_rate, stream);
    CHECK_ERRORS("After adjust_parameter_matrix");
}