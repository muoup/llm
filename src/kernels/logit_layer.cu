#include "logit_layer.hpp"

#include <kernels/matrix_device_kernels.cuh>

// for (size_t i = 0; i < predictions.rows; ++i) {
//         for (size_t j = 0; j < predictions.cols; ++j) {
//             const auto delta_loss = predictions.get(i, j) -
//                  (j == actual[i + 1] ? 1.0f : 0.0f);
//             logit_loss_gradient.set(i, j, delta_loss);
//             logit_bias_gradient.offset(0, j, delta_loss);
//             if (j == actual[i + 1]) {
//                 average_loss -= (std::log(predictions.get(i, j) + 1e-10f)) /
//                 predictions.rows;
//             }
//         }
//     }

__global__ void compute_loss_gradient_kernel(
    const token_id_t actual[],
    const const_matrix_view predictions,
    const matrix_view loss_gradient,
    const matrix_view bias_gradient,
    float* loss) {
    std::uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= predictions.rows || col >= predictions.cols)
        return;

    const token_id_t actual_token = actual[row + 1];

    float pred_value = kernel::matrix::device_get(predictions, row, col);
    float delta_loss = pred_value - (col == actual_token ? 1.0f : 0.0f);

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
    size_t vocab_size) {
    ::matrix logit_loss_gradient(predictions.rows, predictions.cols);
    ::matrix logit_bias_gradient(1, vocab_size);
    float* average_loss;
    cudaMalloc(&average_loss, sizeof(float));

    token_id_t* d_actual;
    cudaMalloc(&d_actual, actual.size() * sizeof(token_id_t));
    cudaMemcpy(d_actual, actual.data(), actual.size() * sizeof(token_id_t),
               cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((predictions.cols + block_size.x - 1) / block_size.x,
                   (predictions.rows + block_size.y - 1) / block_size.y);

    compute_loss_gradient_kernel<<<grid_size, block_size>>>(
        d_actual, predictions, logit_loss_gradient, logit_bias_gradient,
        average_loss);

    float host_loss;
    cudaMemcpy(&host_loss, average_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(average_loss);
    cudaFree(d_actual);

    return LossResult{
        .logit_loss_gradient = std::move(logit_loss_gradient),
        .logit_bias_gradient = std::move(logit_bias_gradient),
        .average_loss = host_loss,
    };
}
