#include "optimizer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <util/matrix.hpp>

void norm_clip(matrix& gradient) {
    constexpr auto max_magnitude = 5.0f;
    const auto max = gradient.absmax();
    kernel::matrix::check_errors("After absmax in norm_clip");

    if (max > max_magnitude) {
        float factor = max_magnitude / max;
        gradient.scale(factor);
    }
}

__global__ void _test_output() {
    printf("Optimizer kernel loaded successfully.\n");
}

__global__ void regularize_gradient(const matrix_view gradient,
                                    const const_matrix_view parameters) {
    constexpr float regularization_term = 0.01f;

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= gradient.rows || col >= gradient.cols) {
        return;
    }

    float param_value = kernel::matrix::device_get(parameters, row, col);
    if (param_value != 0.0f) {
        float grad_value = kernel::matrix::device_get(gradient, row, col);
        grad_value += 2 * regularization_term * param_value;
        kernel::matrix::device_set(gradient, row, col, grad_value);
    }
}

void adjust_parameter_matrix(matrix& adjust,
                             matrix& gradient,
                             float learning_rate) {
    MATRIX_ASSERT(adjust.rows == gradient.rows && adjust.cols == gradient.cols,
                  "Dimension mismatch in adjust_parameter_matrix");
    
    dim3 threads_per_block(16, 16);
    dim3 blocks((adjust.rows + threads_per_block.x - 1) / threads_per_block.x,
                (adjust.cols + threads_per_block.y - 1) / threads_per_block.y);

    // regularize_gradient<<<blocks, threads_per_block>>>(gradient, adjust);
    // kernel::matrix::check_errors("After regularize_gradient");
    norm_clip(gradient);
    kernel::matrix::add_scaled(adjust, gradient, -learning_rate);
    kernel::matrix::check_errors("After adjust_parameter_matrix");
}
