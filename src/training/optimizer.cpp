#include "optimizer.hpp"
#include "util/matrix.hpp"

#include <cmath>

void norm_clip(matrix &gradient) {
    constexpr auto max_magnitude = 5.0f;
    const auto max = gradient.absmax();
    if (max > max_magnitude) {
        float factor = max_magnitude / max;
        if (std::isinf(factor) || std::isnan(factor) || factor > 1.0f) {
            factor = 1.0f;
        }
        gradient.scale(factor);
    }
}

void adjust_parameter_matrix(matrix &adjust, const matrix &gradient, float learning_rate) {
    MATRIX_ASSERT(adjust.rows == gradient.rows && adjust.cols == gradient.cols,
                  "Dimension mismatch in adjust_parameter_matrix");
    
    matrix clipped_gradient = gradient.clone();
    norm_clip(clipped_gradient);

    for (size_t i = 0; i < adjust.rows; ++i) {
        for (size_t j = 0; j < adjust.cols; ++j) {
            const auto delta = clipped_gradient.get(i, j) * learning_rate;
            adjust.offset(i, j, -delta);
        }
    }
}

void regularize_weight_gradient(matrix &gradient, const matrix &weights, float regularization_strength) {
    MATRIX_ASSERT(gradient.rows == weights.rows && gradient.cols == weights.cols,
                  "Dimension mismatch in regularize_weight_gradient\n"
                  "[%d, %d] != [%d, %d]",
                  gradient.rows, gradient.cols, weights.rows, weights.cols);
    
    for (size_t i = 0; i < gradient.rows; ++i) {
        for (size_t j = 0; j < gradient.cols; ++j) {
            const auto weight_value = weights.get(i, j);
            const auto regularization_term = 2 * regularization_strength * weight_value;
            gradient.offset(i, j, regularization_term);
        }
    }
}