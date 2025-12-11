#pragma once

#include <util/matrix.hpp>

class LayerNorm;

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

struct LayerNormGradients {
    matrix grad_input;
    matrix grad_gamma;
    matrix grad_beta;
};

LayerNormGradients layer_normalization_backward(const ::LayerNorm& layer,
                                                const matrix& input,
                                                const matrix& gamma,
                                                const matrix& beta,
                                                const matrix& mean,
                                                const matrix& inv_variance,
                                                const matrix& grad_output,
                                                float epsilon);

}  // namespace kernel::layer_norm
