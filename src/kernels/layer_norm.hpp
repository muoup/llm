#pragma once

#include <util/matrix.hpp>

namespace kernel::layer_norm {

struct LayerNormResult {
    matrix normalized;
    matrix mean;
    matrix inv_variance;
};
    
LayerNormResult layer_normalization(const matrix& input,
                                          const matrix& gamma,
                                          const matrix& beta,
                                          float epsilon);

}
