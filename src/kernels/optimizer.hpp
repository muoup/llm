#pragma once

#include <util/matrix.hpp>

void norm_clip(matrix &gradient);

void adjust_parameter_matrix(matrix &adjust, matrix &gradient, float learning_rate);