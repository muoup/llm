#include "optimizer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix.hpp>
#include <kernels/scheduling.cuh>
#include <kernels/scheduling.hpp>

#include <util/matrix.hpp>

constexpr auto NORM_CLIP_MAX_MAG = 2.5f;

__global__ void norm_clip_kernel(matrix_view gradient,
                                 float* sum_sq_ptr,
                                 size_t total_elements,
                                 size_t normalization_count) {
    float sum_of_squares = *sum_sq_ptr;
    size_t div = normalization_count > 0 ? normalization_count : total_elements;
    float mag = sum_of_squares / (float)div;

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

void kernel::optimizer::norm_clip(::matrix& gradient, kernel_stream_t stream, size_t normalization_count) {
    float_device_ptr_t sum_sq_ptr
        = kernel::matrix::sum_of_squares(gradient, stream);

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (gradient.rows + threads_per_block.x - 1) / threads_per_block.x,
        (gradient.cols + threads_per_block.y - 1) / threads_per_block.y);

    norm_clip_kernel<<<blocks, threads_per_block, 0,
                       get_kernel_stream(stream)>>>(
        gradient, (float*)sum_sq_ptr, gradient.size(), normalization_count);
    CHECK_ERRORS("After norm_clip_kernel");
}

static __global__ void regularize_gradient(const matrix_view gradient,
                                           const const_matrix_view parameters,
                                           float* sum_sq_ptr,
                                           size_t total_elements,
                                           size_t normalization_count) {
    constexpr float regularization_strength = 0.0001f;
    float sum_of_squares = *sum_sq_ptr;
    size_t div = normalization_count > 0 ? normalization_count : total_elements;
    float mag = sum_of_squares / (float)div;

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= gradient.rows || col >= gradient.cols) {
        return;
    }

    float device_value = kernel::matrix::device_get(gradient, row, col);

    if (mag > NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG) {
        device_value *= sqrtf(NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG / mag);
    }

    float param_value = kernel::matrix::device_get(parameters, row, col);
    float regularization = 2 * regularization_strength * param_value;
    kernel::matrix::device_set(gradient, row, col,
                               device_value + regularization);
}

void kernel::optimizer::regularize_weight_gradient(::matrix& gradient,
                                                   const ::matrix& parameters,
                                                   kernel_stream_t stream,
                                                   size_t normalization_count) {
    MATRIX_ASSERT(
        gradient.rows == parameters.rows && gradient.cols == parameters.cols,
        "Dimension mismatch in regularize_gradient");

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (gradient.rows + threads_per_block.x - 1) / threads_per_block.x,
        (gradient.cols + threads_per_block.y - 1) / threads_per_block.y);

    float_device_ptr_t sum_sq_ptr
        = kernel::matrix::sum_of_squares(gradient, stream);
    regularize_gradient<<<blocks, threads_per_block, 0,
                          get_kernel_stream(stream)>>>(
        gradient, parameters, (float*)sum_sq_ptr, gradient.size(), normalization_count);
    CHECK_ERRORS("After regularize_gradient");
}

static __global__ void adamw_step_kernel(matrix_view parameter,
                                         const const_matrix_view gradient,
                                         matrix_view m,
                                         matrix_view v,
                                         size_t t,
                                         float learning_rate,
                                         float beta1,
                                         float beta2,
                                         float epsilon,
                                         float weight_decay) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= parameter.rows || col >= parameter.cols) {
        return;
    }

    float theta = kernel::matrix::device_get(parameter, row, col);
    float grad = kernel::matrix::device_get(gradient, row, col);
    float m_t_minus_1 = kernel::matrix::device_get(m, row, col);
    float v_t_minus_1 = kernel::matrix::device_get(v, row, col);

    theta -= learning_rate * weight_decay * theta;

    float m_t = beta1 * m_t_minus_1 + (1.0f - beta1) * grad;
    float v_t = beta2 * v_t_minus_1 + (1.0f - beta2) * grad * grad;

    kernel::matrix::device_set(m, row, col, m_t);
    kernel::matrix::device_set(v, row, col, v_t);

    float m_hat = m_t / (1.0f - powf(beta1, (float)t));
    float v_hat = v_t / (1.0f - powf(beta2, (float)t));

    float next_theta = theta - learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    
    kernel::matrix::device_set(parameter, row, col, next_theta);
}

void kernel::optimizer::adamw_step(::matrix& parameter,
                                   const ::matrix& gradient,
                                   ::matrix& m,
                                   ::matrix& v,
                                   size_t t,
                                   float learning_rate,
                                   float beta1,
                                   float beta2,
                                   float epsilon,
                                   float weight_decay,
                                   kernel_stream_t stream) {
    MATRIX_ASSERT(parameter.rows == gradient.rows && parameter.cols == gradient.cols,
                  "Dimension mismatch in adamw_step: parameter vs gradient");
    MATRIX_ASSERT(parameter.rows == m.rows && parameter.cols == m.cols,
                  "Dimension mismatch in adamw_step: parameter vs m");
    MATRIX_ASSERT(parameter.rows == v.rows && parameter.cols == v.cols,
                  "Dimension mismatch in adamw_step: parameter vs v");

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (parameter.rows + threads_per_block.x - 1) / threads_per_block.x,
        (parameter.cols + threads_per_block.y - 1) / threads_per_block.y);

    adamw_step_kernel<<<blocks, threads_per_block, 0,
                        get_kernel_stream(stream)>>>(
        parameter, gradient, m, v, t, learning_rate, beta1, beta2, epsilon, weight_decay);
    CHECK_ERRORS("After adamw_step_kernel");
}
