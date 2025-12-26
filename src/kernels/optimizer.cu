#include "optimizer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <util/matrix.hpp>

void kernel::optimizer::norm_clip(::matrix& gradient) {
    constexpr auto max_magnitude = 5.0f;
    const auto mag = kernel::matrix::sum_of_squares(gradient) / gradient.size();
    CHECK_ERRORS("After absmax in norm_clip");

    if (mag > max_magnitude * max_magnitude) {
        const float scale = sqrtf((max_magnitude * max_magnitude) / mag);
        kernel::matrix::scale(gradient, scale);
        CHECK_ERRORS("After scaling in norm_clip");
    }
}

__global__ void _test_output() {
    printf("Optimizer kernel loaded successfully.\n");
}

static __global__ void regularize_gradient(const matrix_view gradient,
                                           const const_matrix_view parameters) {
    constexpr float regularization_strength = 0.0001f;

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= gradient.rows || col >= gradient.cols) {
        return;
    }

    float param_value = kernel::matrix::device_get(parameters, row, col);
    float regularization = 2 * regularization_strength * param_value;
    kernel::matrix::device_offset_elem(gradient, row, col, regularization);
}

void kernel::optimizer::regularize_weight_gradient(::matrix& gradient,
                                                   const ::matrix& parameters) {
    MATRIX_ASSERT(
        gradient.rows == parameters.rows && gradient.cols == parameters.cols,
        "Dimension mismatch in regularize_gradient");

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (gradient.rows + threads_per_block.x - 1) / threads_per_block.x,
        (gradient.cols + threads_per_block.y - 1) / threads_per_block.y);

    norm_clip(gradient);
    wait_for_operations();
    regularize_gradient<<<blocks, threads_per_block>>>(gradient, parameters);
    CHECK_ERRORS("After regularize_gradient");
}

void kernel::optimizer::adjust_parameter_matrix(::matrix& adjust,
                                                ::matrix& gradient,
                                                float learning_rate) {
    MATRIX_ASSERT(adjust.rows == gradient.rows && adjust.cols == gradient.cols,
                  "Dimension mismatch in adjust_parameter_matrix");

    dim3 threads_per_block(16, 16);
    dim3 blocks((adjust.rows + threads_per_block.x - 1) / threads_per_block.x,
                (adjust.cols + threads_per_block.y - 1) / threads_per_block.y);

    // kernel::optimizer::norm_clip(gradient);
    // kernel::optimizer::wait_for_operations();
    kernel::matrix::add_scaled(adjust, gradient, -learning_rate);
    CHECK_ERRORS("After adjust_parameter_matrix");
}

void kernel::optimizer::wait_for_operations() {
    cudaDeviceSynchronize();
}