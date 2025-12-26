#pragma once

#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>

class LayerNorm;

namespace kernel::layer_norm {

struct LayerNormResult {
    ::matrix normalized;
    ::matrix mean;
    ::matrix inv_variance;
};

LayerNormResult layer_normalization(const ::matrix& input,
                                    const ::matrix& gamma,
                                    const ::matrix& beta,
                                    float epsilon,
                                    kernel_stream_t stream = nullptr);

struct LayerNormGradients {
    ::matrix grad_input;
    ::matrix grad_gamma;
    ::matrix grad_beta;
};

LayerNormGradients layer_normalization_backward(const ::matrix& input,
                                                const ::matrix& gamma,
                                                const ::matrix& beta,
                                                const ::matrix& mean,
                                                const ::matrix& inv_variance,
                                                const ::matrix& grad_normalized,
                                                float epsilon,
                                                kernel_stream_t stream = nullptr);

}  // namespace kernel::layer_norm
