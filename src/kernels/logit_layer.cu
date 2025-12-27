#include "logit_layer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/optimizer.hpp>
#include <kernels/scheduling.cuh>

static __global__ void compute_loss_gradient_kernel(
    const token_id_t actual[],
    const const_matrix_view predictions,
    const matrix_view loss_gradient,
    const matrix_view bias_gradient,
    float* loss) {
    std::uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= predictions.rows || col >= predictions.cols)
        return;

    const token_id_t actual_token = actual[row];

    float pred_value = kernel::matrix::device_get(predictions, row, col);
    float delta_loss = pred_value - (col == actual[row] ? 1.0f : 0.0f);

    kernel::matrix::device_set(loss_gradient, row, col, delta_loss);
    kernel::matrix::device_offset_elem_atomic(bias_gradient, 0, col,
                                              delta_loss);

    if (col == actual_token) {
        atomicAdd(loss, -std::log(pred_value + 1e-10f) / predictions.rows);
    }
}

kernel::logit_layer::LossResult kernel::logit_layer::compute_loss_gradient(
    const ::matrix& predictions,
    const std::span<const token_id_t> actual,
    size_t vocab_size,
    kernel_stream_t stream) {
    ::matrix logit_loss_gradient(predictions.rows, predictions.cols);
    ::matrix logit_bias_gradient(1, vocab_size);
    
    float* device_average_loss;
    cudaMalloc(&device_average_loss, sizeof(float));
    cudaMemsetAsync(device_average_loss, 0, sizeof(float), get_kernel_stream(stream));

    token_id_t* d_actual;
    cudaMalloc(&d_actual, actual.size() * sizeof(token_id_t));
    cudaMemcpyAsync(d_actual, actual.data(), actual.size() * sizeof(token_id_t),
               cudaMemcpyHostToDevice, get_kernel_stream(stream));

    dim3 block_size(16, 16);
    dim3 grid_size((predictions.cols + block_size.x - 1) / block_size.x,
                   (predictions.rows + block_size.y - 1) / block_size.y);

    compute_loss_gradient_kernel<<<grid_size, block_size, 0, get_kernel_stream(stream)>>>(
        d_actual, predictions, logit_loss_gradient, logit_bias_gradient,
        device_average_loss);

    float host_loss;
    cudaMemcpyAsync(&host_loss, device_average_loss, sizeof(float), cudaMemcpyDeviceToHost, get_kernel_stream(stream));
    cudaStreamSynchronize(get_kernel_stream(stream));
    cudaFree(device_average_loss);
    cudaFree(d_actual);

    return LossResult{
        .logit_loss_gradient = std::move(logit_loss_gradient),
        .logit_bias_gradient = std::move(logit_bias_gradient),
        .average_loss = host_loss,
    };
}
