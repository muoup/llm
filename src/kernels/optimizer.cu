#include "optimizer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <mutex>
#include <util/matrix.hpp>

#include <cublas_api.h>
#include <cuda_runtime.h>

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

static __global__ void gradient_regularize_and_adjust(
    const matrix_view gradient,
    const const_matrix_view parameters,
    const float learning_rate) {
    constexpr float regularization_strength = 0.0001f;

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= gradient.rows || col >= gradient.cols) {
        return;
    }

    float param_value = kernel::matrix::device_get(parameters, row, col);
    float regularization = 2 * regularization_strength * param_value;
    float gradient_val = kernel::matrix::device_get(gradient, row, col);
    float updated_parameter
        = param_value - learning_rate * (gradient_val + regularization);

    kernel::matrix::device_set(gradient, row, col, updated_parameter);
}

static kernel::optimizer::kernel_stream_pool adjustment_pool(16);

void kernel::optimizer::adjust_regularize_parameter_matrix(
    ::matrix& gradient,
    const ::matrix& parameters,
    float learning_rate) {
    auto stream = adjustment_pool.get_next_stream();
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

    dim3 threads_per_block(16, 16);
    dim3 blocks((adjust.rows + threads_per_block.x - 1) / threads_per_block.x,
                (adjust.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel::matrix::add_scaled(adjust, gradient, -learning_rate);
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
