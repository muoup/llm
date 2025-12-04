#pragma once

#include <util/matrix.hpp>

void norm_clip(matrix &gradient);

void regularize_weight_gradient(matrix &gradient, const matrix &weights, float regularization_rate = 0.01f);

void adjust_matrix(matrix &adjust, const matrix &gradient, float learning_rate);
