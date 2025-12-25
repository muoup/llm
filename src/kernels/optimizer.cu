#include "optimizer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <mutex>
#include <util/matrix.hpp>

#include <cublas_api.h>
#include <cuda_runtime.h>

void kernel::optimizer::norm_clip(::matrix& gradient, kernel_stream_t stream) {
    if (stream) {
        wait_for_stream(stream);
    } else {
        wait_for_operations();
    }

    constexpr auto max_magnitude = 5.0f;
    const float sum_sq = kernel::matrix::sum_of_squares(gradient);
    const float l2_norm = std::sqrt(sum_sq);

    CHECK_ERRORS("After sum_of_squares in norm_clip");

    if (l2_norm > max_magnitude) {
        const float scale = max_magnitude / l2_norm;
        kernel::matrix::scale(gradient, scale, stream);
        CHECK_ERRORS("After scaling in norm_clip");
    }
}

__global__ void _test_output() {
    printf("Optimizer kernel loaded successfully.\n");
}

static __global__ void gradient_regularize_and_adjust(
    const matrix_view gradient,
    const matrix_view parameters,
    const float learning_rate) {
    constexpr float regularization_strength = 0.001f;
    constexpr float clip_value = 1.0f;

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= gradient.rows || col >= gradient.cols) {
        return;
    }

    float param_value = kernel::matrix::device_get(parameters, row, col);
    float regularization = 2 * regularization_strength * param_value;
    float gradient_val = kernel::matrix::device_get(gradient, row, col);

    // Value clipping
    gradient_val = fmaxf(fminf(gradient_val, clip_value), -clip_value);

    float updated_gradient = gradient_val + regularization;
    float updated_parameter = param_value - learning_rate * updated_gradient;

    kernel::matrix::device_set(gradient, row, col, updated_gradient);
    kernel::matrix::device_set(parameters, row, col, updated_parameter);
}

static __global__ void gradient_adjust_clipped(
    const matrix_view parameters,
    const const_matrix_view gradient,
    const float learning_rate) {
    constexpr float clip_value = 1.0f;

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= parameters.rows || col >= parameters.cols) {
        return;
    }

    float gradient_val = kernel::matrix::device_get(gradient, row, col);
    
    // Value clipping
    gradient_val = fmaxf(fminf(gradient_val, clip_value), -clip_value);

    kernel::matrix::device_offset_elem(parameters, row, col, -learning_rate * gradient_val);
}

static kernel::optimizer::kernel_stream_pool adjustment_pool(16);

void kernel::optimizer::adjust_regularize_parameter_matrix(
    ::matrix& gradient,
    ::matrix& parameters,
    float learning_rate,
    kernel_stream_t stream) {
    if (stream == nullptr) {
        stream = adjustment_pool.get_next_stream();
    }
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    MATRIX_ASSERT(
        gradient.rows == parameters.rows && gradient.cols == parameters.cols,
        "Dimension mismatch in regularize_gradient");

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (gradient.rows + threads_per_block.x - 1) / threads_per_block.x,
        (gradient.cols + threads_per_block.y - 1) / threads_per_block.y);

    gradient_regularize_and_adjust<<<blocks, threads_per_block, 0,
                                     cuda_stream>>>(gradient, parameters,
                                                    learning_rate);

    CHECK_ERRORS("After regularize_gradient");
}

void kernel::optimizer::adjust_parameter_matrix(::matrix& adjust,
                                                ::matrix& gradient,
                                                float learning_rate) {
    MATRIX_ASSERT(adjust.rows == gradient.rows && adjust.cols == gradient.cols,
                  "Dimension mismatch in adjust_parameter_matrix");
    
    auto stream = adjustment_pool.get_next_stream();
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (adjust.rows + threads_per_block.x - 1) / threads_per_block.x,
        (adjust.cols + threads_per_block.y - 1) / threads_per_block.y);

    gradient_adjust_clipped<<<blocks, threads_per_block, 0, cuda_stream>>>(
        adjust, gradient, learning_rate);

    CHECK_ERRORS("After adjust_parameter_matrix");
}

void kernel::optimizer::wait_for_operations() {
    cudaDeviceSynchronize();
}

void kernel::optimizer::wait_for_stream(kernel_stream_t stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaStreamSynchronize(cuda_stream);
}

kernel::optimizer::kernel_stream_pool::kernel_stream_pool(size_t pool_size) {
    for (size_t i = 0; i < pool_size; ++i) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        streams.push_back(stream);
    }
}

kernel::optimizer::kernel_stream_pool::~kernel_stream_pool() {
    for (auto& stream : streams) {
        cudaStream_t stream_ptr = static_cast<cudaStream_t>(stream);

        cudaStreamDestroy(stream_ptr);
    }
}

kernel::optimizer::kernel_stream_t
kernel::optimizer::kernel_stream_pool::get_next_stream() {
    this->next_stream_lock.lock();

    kernel_stream_t stream = streams[this->next_stream];
    this->next_stream = (this->next_stream + 1) % streams.size();

    this->next_stream_lock.unlock();

    return stream;
}
