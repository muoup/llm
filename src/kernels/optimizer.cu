#include "optimizer.hpp"
#include "util/matrix.hpp"

void norm_clip(matrix &gradient) {
    constexpr auto max_magnitude = 5.0f;
    const auto max = gradient.absmax();

    if (max > max_magnitude) {
        float factor = max_magnitude / max;
        gradient.scale(factor);
    }
}

__global__ void adjust_parameter_kernel(float *adjust_data, const float *gradient_data,
                                        size_t rows, size_t cols, float learning_rate) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = rows * cols;

    if (idx < total_elements) {
        float grad_value = gradient_data[idx];
        // Norm clipping
        if (fabsf(grad_value) > 5.0f) {
            grad_value = (grad_value / fabsf(grad_value)) * 5.0f;
        }
        float delta = grad_value * learning_rate;
        adjust_data[idx] -= delta;
    }
}

void adjust_parameter_matrix(matrix &adjust, const matrix &gradient, float learning_rate) {
    MATRIX_ASSERT(adjust.rows == gradient.rows && adjust.cols == gradient.cols,
                  "Dimension mismatch in adjust_parameter_matrix");

    matrix clipped_gradient = gradient.clone();
    norm_clip(clipped_gradient);
    
    size_t total_elements = adjust.rows * adjust.cols;
    size_t threads_per_block = 256;
    size_t blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    adjust_parameter_kernel<<<blocks, threads_per_block>>>(adjust.data, clipped_gradient.data,
                                                           adjust.rows, adjust.cols, learning_rate);
}