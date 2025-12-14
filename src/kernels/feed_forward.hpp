#pragma once

#include <util/matrix.hpp>

namespace kernel::feed_forward {
    
void add_bias(matrix& mat, const matrix& row_vec);
matrix sum_columns(const matrix& mat);

matrix leaky_relu_activation(const matrix& activation_input);
matrix leaky_relu_activation_backprop(const ::matrix& activation_input, const ::matrix& a1_gradient);

}
