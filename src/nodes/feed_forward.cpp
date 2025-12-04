#include "feed_forward.hpp"
#include <iostream>

// ---[ Backpropagation Helpers (from backpropogation.cpp) ]---

static float norm_clip_factor(const matrix &gradient) {
    constexpr auto max_magnitude = 5.0f;
    const auto max = gradient.absmax();
    if (max > max_magnitude) {
        return max_magnitude / max;
    }
    return 1.0f;
}

static void adjust_matrix(matrix &adjust, const matrix &gradient, float learning_rate) {
    const float factor = norm_clip_factor(gradient);
    for (size_t i = 0; i < adjust.rows; ++i) {
        for (size_t j = 0; j < adjust.cols; ++j) {
            const auto delta = gradient.get(i, j) * factor * learning_rate;
            adjust.offset(i, j, -delta);
        }
    }
}

static void regularize_weight_gradient(matrix &gradient, const matrix &weights, float regularization_strength) {
    for (size_t i = 0; i < gradient.rows; ++i) {
        for (size_t j = 0; j < gradient.cols; ++j) {
            const auto weight_value = weights.get(i, j);
            const auto regularization_term = 2 * regularization_strength * weight_value;
            gradient.offset(i, j, regularization_term);
        }
    }
}

// ---[ Activation Function ]---

static matrix activate(const matrix& input) {
    constexpr static auto leaky_relu = [](const float f) { return f < 0 ? 0.01f * f : f; };
    return input.clone().map(leaky_relu);
}

// ---[ Construction ]---

NodeType FeedForwardLayer::getType() const {
    return NodeType::FeedForward;
}

FeedForwardLayer::FeedForwardLayer(size_t dimensions, size_t projection_size)
    : w1({ dimensions, projection_size }), b1({ 1, projection_size }),
      w2({ projection_size, dimensions }), b2({ 1, dimensions }) {}

// ---[ Layer Operations ]---

void FeedForwardLayer::randomize(const float min, const float max) {
    w1.randomize(min, max);
    b1.randomize(min, max);
    w2.randomize(min, max);
    b2.randomize(min, max);
}

std::vector<matrix> FeedForwardLayer::forward(std::span<const matrix> inputs) {
    const matrix& input = inputs[0];

    matrix activation_input = input.cross_multiplied(w1);
    for (size_t i = 0; i < activation_input.rows; ++i) {
        activation_input.add_row_vector(i, b1);
    }

    matrix activation_output = activate(activation_input);
    
    matrix final_output = activation_output.cross_multiplied(w2);
    for (size_t i = 0; i < final_output.rows; ++i) {
        final_output.add_row_vector(i, b2);
    }

    return {
        std::move(final_output),
        std::move(activation_input),
        std::move(activation_output)
    };
}

std::vector<matrix> FeedForwardLayer::backpropagate(
    std::span<const matrix> inputs,
    std::span<const matrix> outputs,
    std::span<const matrix> gradients,
    float learning_rate) {

    constexpr float regularization_strength = 0.01f;

    const matrix& layer_input = inputs[0];
    const matrix& activation_input = outputs[1];
    const matrix& activation_output = outputs[2];
    const matrix& post_layer_gradient = gradients[0];
    
    matrix b2_gradient({ 1, post_layer_gradient.cols });
    for (size_t i = 0; i < b2_gradient.cols; ++i) {
        b2_gradient.set(0, i, post_layer_gradient.col_sum(i));
    }
    adjust_matrix(b2, b2_gradient, learning_rate);

    matrix w2_gradient = activation_output.transposed().cross_multiplied(post_layer_gradient);
    regularize_weight_gradient(w2_gradient, w2, regularization_strength);
    adjust_matrix(w2, w2_gradient, learning_rate);

    const matrix a1_gradient = post_layer_gradient.cross_multiplied(w2.transposed());
    matrix z1_gradient({ a1_gradient.rows, a1_gradient.cols });

    for (size_t i = 0; i < z1_gradient.rows; i++) {
        for (size_t j = 0; j < z1_gradient.cols; j++) {
            const auto z1_value = activation_input.get(i, j);
            const auto self_value = a1_gradient.get(i, j);
            z1_gradient.set(i, j, self_value * (z1_value > 0 ? 1.0f : 0.01f));
        }
    }

    matrix b1_gradient({ 1, z1_gradient.cols });
    for (size_t i = 0; i < z1_gradient.cols; ++i) {
        b1_gradient.set(0, i, z1_gradient.col_sum(i));
    }
    adjust_matrix(b1, b1_gradient, learning_rate);

    matrix w1_gradient = layer_input.transposed().cross_multiplied(z1_gradient);
    regularize_weight_gradient(w1_gradient, w1, regularization_strength);
    adjust_matrix(w1, w1_gradient, learning_rate);

    return { z1_gradient.cross_multiplied(w1.transposed()) };
}

// ---[ Serialization ]---

static void write_matrix(std::ostream& out, const matrix& m) {
    uint64_t dims[] = { m.rows, m.cols };
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(m.data_ptr()), m.buffer_size());
}

static matrix read_matrix(std::istream& in) {
    uint64_t dims[2];
    in.read(reinterpret_cast<char*>(dims), sizeof(dims));
    matrix m(dims[0], dims[1]);
    in.read(reinterpret_cast<char*>(m.data_ptr()), m.buffer_size());
    return m;
}

void FeedForwardLayer::save(std::ostream& out) const {
    write_matrix(out, w1);
    write_matrix(out, b1);
    write_matrix(out, w2);
    write_matrix(out, b2);
}

FeedForwardLayer FeedForwardLayer::load(std::istream& in) {
    const auto w1 = read_matrix(in);
    const auto b1 = read_matrix(in);
    const auto w2 = read_matrix(in);
    const auto b2 = read_matrix(in);
    
    FeedForwardLayer layer(w1.rows, w1.cols);
    layer.w1 = w1;
    layer.b1 = b1;
    layer.w2 = w2;
    layer.b2 = b2;
    
    return layer;
}
