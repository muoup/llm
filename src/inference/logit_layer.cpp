#include "logit_layer.hpp"

#include <kernels/layers/feed_forward.hpp>
#include <kernels/layers/logit_layer.hpp>
#include <kernels/optimizer.hpp>
#include <tokenizer/token.hpp>
#include <util/logger.hpp>

LogitLayer::LogitLayer(const size_t dimensions,
                       const size_t vocab_size,
                       DataType dtype)
    : dimensions(dimensions),
      vocab_size(vocab_size),
      w(dimensions, vocab_size, dtype),
      b(1, vocab_size, dtype) {}

size_t LogitLayer::parameterCount() const {
    return (w.rows * w.cols) + (b.rows * b.cols);
}

void LogitLayer::randomize(const float min, const float max) {
    w.xavier_randomize();
    b.xavier_randomize();
}

matrix LogitLayer::forward(const matrix& input) const {
    matrix logits = input.cross_multiplied(w);

    LOG_DEBUG("  Logit Layer Forward:");
    LOG_DEBUG("    input norm: %f", input.norm());

    kernel::feed_forward::add_bias(logits, b);

    LOG_DEBUG("    logits norm: %f", logits.norm());

    return logits;
}

std::pair<matrix, float> LogitLayer::backpropogate(
    const matrix& input,
    const matrix& predictions,
    const std::span<const token_id_t> actual,
    CentralOptimizer& optimizer) {
    kernel::logit_layer::LossResult loss_result
        = kernel::logit_layer::compute_loss_gradient(predictions, actual,
                                                     vocab_size);
    kernel::wait_for_all_streams();

    matrix h_final_gradient
        = loss_result.logit_loss_gradient.cross_t_multiplied(w);
    matrix logit_weight_gradient
        = input.t_cross_multiplied(loss_result.logit_loss_gradient);
    kernel::optimizer::norm_clip(h_final_gradient);

    // Regularize weight gradient is now handled by AdamW implicitly via weight
    // decay (decoupled) But if we want to keep explicit regularization logic
    // separate: AdamW implements weight decay directly. The old
    // regularize_weight_gradient was doing 2*strength*param. The new AdamW step
    // handles weight decay. However, the old implementation might have been
    // doing L2 regularization on top of gradient. AdamW decouples weight decay
    // from gradient update. If the old one was L2 loss added to gradient, then
    // AdamW weight decay parameter covers it. I will remove explicit
    // regularization call here and let AdamW handle it.

    kernel::wait_for_all_streams();

    LOG_DEBUG("  Logit Layer Gradients:");
    LOG_DEBUG("    logit_weight_gradient norm: %f",
              logit_weight_gradient.norm());
    LOG_DEBUG("    logit_bias_gradient norm: %f",
              loss_result.logit_bias_gradient.norm());

    optimizer.update(b, loss_result.logit_bias_gradient);
    optimizer.update(w, logit_weight_gradient);
    kernel::wait_for_all_streams();

    return { std::move(h_final_gradient), loss_result.average_loss };
}

void LogitLayer::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    w.save(out);
    b.save(out);
}

LogitLayer LogitLayer::load(std::istream& in) {
    size_t dimensions, vocab_size;
    in.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

    LogitLayer layer;
    layer.dimensions = dimensions;
    layer.vocab_size = vocab_size;
    layer.w = matrix::load(in);
    layer.b = matrix::load(in);
    return layer;
}
