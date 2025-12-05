#include "logit_layer.hpp"

#include <training/optimizer.hpp>
#include <tokenizer/token.hpp>

LogitLayer::LogitLayer(const size_t dimensions, const size_t vocab_size)
    : dimensions(dimensions), vocab_size(vocab_size), w(dimensions, vocab_size), b(1, vocab_size) {}

size_t LogitLayer::parameterCount() const {
    return (w.rows * w.cols) + (b.rows * b.cols);
}
    
void LogitLayer::randomize(const float min, const float max) {
    w.randomize(min, max);
    b.randomize(min, max);
}

matrix LogitLayer::forward(const matrix& input) const {
    matrix logits = input.cross_multiplied(w);
    
    for (size_t i = 0; i < logits.rows; ++i) {
        logits.add_row_vector(i, b);
    }
    
    logits.softmax();   
    return logits;
}

std::pair<matrix, float> LogitLayer::backpropogate(const matrix& input, const matrix& predictions, const std::span<const token_id_t> actual, float learning_rate) {
    matrix logit_loss_gradient({ predictions.rows, vocab_size });
    matrix logit_bias_gradient({ 1, vocab_size });
    float average_loss = 0.0f;

    for (size_t i = 0; i < predictions.rows; ++i) {
        for (size_t j = 0; j < predictions.cols; ++j) {
            const auto delta_loss = predictions.get(i, j) - (j == actual[i + 1] ? 1.0f : 0.0f);
            logit_loss_gradient.set(i, j, delta_loss);
            logit_bias_gradient.offset(0, j, delta_loss);
            if (j == actual[i + 1]) {
                average_loss -= (std::log(predictions.get(i, j) + 1e-10f)) / predictions.rows;
            }
        }
    }

    adjust_matrix(b, logit_bias_gradient, learning_rate);

    matrix h_final_gradient = logit_loss_gradient.cross_t_multiplied(w);
    norm_clip(h_final_gradient);
    matrix logit_weight_gradient = input.t_cross_multiplied(logit_loss_gradient);

    regularize_weight_gradient(logit_weight_gradient, w);
    adjust_matrix(w, logit_weight_gradient, learning_rate);

    return { std::move(h_final_gradient), average_loss };
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
