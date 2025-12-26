#pragma once

#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>
#include <tokenizer/token.hpp>

#include <span>

namespace kernel::logit_layer {
    struct LossResult {
        matrix logit_loss_gradient;
        matrix logit_bias_gradient;
        float average_loss;
    };
    
    LossResult compute_loss_gradient(const matrix& predictions,
                                     const std::span<const token_id_t> actual,
                                     size_t vocab_size,
                                     kernel_stream_t stream = nullptr);
}