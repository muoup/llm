#include "logit_layer.hpp"

#include <kernels/feed_forward.hpp>
#include <kernels/logit_layer.hpp>
#include <kernels/optimizer.hpp>
#include <tokenizer/token.hpp>

LogitLayer::LogitLayer(const size_t dimensions, const size_t vocab_size)
    : dimensions(dimensions),
      vocab_size(vocab_size),
      w(dimensions, vocab_size),
      b(1, vocab_size) {}

size_t LogitLayer::parameterCount() const {
    return (w.rows * w.cols) + (b.rows * b.cols);
}

void LogitLayer::randomize(const float min, const float max) {
    w.randomize(min, max);
    b.randomize(min, max);
}

matrix LogitLayer::forward(const matrix& input) const {
    matrix logits = input.cross_multiplied(w);
    kernel::optimizer::wait_for_operations();
    
    kernel::feed_forward::add_bias(logits, b);
    kernel::optimizer::wait_for_operations();

    logits.softmax();
    kernel::optimizer::wait_for_operations();
    return logits;
}

std::pair<matrix, float> LogitLayer::backpropogate(
    const matrix& input,
    const matrix& predictions,
    const std::span<const token_id_t> actual,
    float learning_rate) {
    kernel::logit_layer::LossResult loss_result
        = kernel::logit_layer::compute_loss_gradient(predictions, actual,
                                                     vocab_size);
    kernel::optimizer::wait_for_operations();

    matrix h_final_gradient
        = loss_result.logit_loss_gradient.cross_t_multiplied(w);
    matrix logit_weight_gradient
        = input.t_cross_multiplied(loss_result.logit_loss_gradient);
    kernel::optimizer::norm_clip(h_final_gradient);
    kernel::optimizer::regularize_weight_gradient(logit_weight_gradient, w);
    kernel::optimizer::wait_for_operations();

    kernel::optimizer::adjust_parameter_matrix(
        b, loss_result.logit_bias_gradient, learning_rate);
    kernel::optimizer::adjust_parameter_matrix(w, logit_weight_gradient,
                                               learning_rate);
    kernel::optimizer::wait_for_operations();

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

    LogitLayer layer(0, 0);
    layer.dimensions = dimensions;
    layer.vocab_size = vocab_size;
    layer.w = matrix::load(in);
    layer.b = matrix::load(in);
    return layer;
}
