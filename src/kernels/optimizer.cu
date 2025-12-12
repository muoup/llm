#include "optimizer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <util/matrix.hpp>

void norm_clip(matrix& gradient) {
    constexpr auto max_magnitude = 5.0f;
    const auto max = gradient.absmax();

    if (max > max_magnitude) {
        float factor = max_magnitude / max;
        gradient.scale(factor);
    }
}

void adjust_parameter_matrix(matrix& adjust,
                             const matrix& gradient,
                             float learning_rate) {
    MATRIX_ASSERT(adjust.rows == gradient.rows && adjust.cols == gradient.cols,
                  "Dimension mismatch in adjust_parameter_matrix");
    
    kernel::matrix::add_scaled(adjust, gradient, -learning_rate);
}
