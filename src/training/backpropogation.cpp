//
// Created by user on 7/17/25.
//

#include "backpropogation.h"

#include <iostream>

#include "../network/neural_net.h"
#include "../tokenizer/token.h"

constexpr auto learning_rate = 0.001f;

float adjustments = 0.0f;

void norm_clip(matrix& gradient) {
    constexpr auto max_magnitude = 2.5f;

    auto max = gradient.max();
    auto min = gradient.min();

    if (max > max_magnitude) {
        gradient.map([max](const float value) {
            return value / max * max_magnitude;
        });
    }

    if (min < -max_magnitude) {
        gradient.map([min](const float value) {
            return value / -min * max_magnitude;
        });
    }
}

void adjust_value(matrix& adjust, const size_t row, const size_t col, const float delta) {
    const auto val = adjust.get(row, col) - delta * learning_rate;

    adjustments += std::abs(delta * learning_rate);

    adjust.set(row, col, val);//std::clamp(val, -1.0f, 1.0f));
}

void adjust_row_vector(matrix& adjust, const size_t row, matrix& gradient) {
    if (gradient.rows != 1 || gradient.cols != adjust.cols) {
        throw std::runtime_error("Gradient must be a row vector with the same number of columns as the adjust matrix.");
    }

    norm_clip(gradient);

    for (size_t j = 0; j < adjust.cols; ++j) {
        adjust_value(adjust, row, j, gradient.get(0, j));
    }
}

void adjust_matrix(matrix& adjust, matrix& gradient) {
    norm_clip(gradient);

    for (size_t i = 0; i < adjust.rows; ++i) {
        for (size_t j = 0; j < adjust.cols; ++j) {
            adjust_value(adjust, i, j, gradient.get(i, j));
        }
    }
}

matrix backpropogate_logit_row(
    llm& model,
    const matrix& last_ff_output, // h_final is the output of the last ff layer
    const matrix& predictions,
    const std::span<const token_id_t> actual
) {
    matrix logit_loss_gradient { actual.size() - 1, model.vocab_size() };

    for (size_t i = 0; i < predictions.rows; ++i) {
        for (size_t j = 0; j < predictions.cols; ++j) {
            const auto delta_loss = predictions.get(i, j) - (j == actual[i] ? 1.0f : 0.0f);
            logit_loss_gradient.set(i, j, delta_loss);
        }
    }

    const matrix weight_output_gradient = last_ff_output.transposed().cross_multiply(logit_loss_gradient);

    for (size_t i = 0; i < weight_output_gradient.rows; ++i) {
        for (size_t j = 0; j < weight_output_gradient.cols; ++j) {
            const auto gradient_val = weight_output_gradient.get(i, j);

            adjust_value(model.m_logit_layer.w, i, j, gradient_val);
        }
    }

    matrix h_final_gradient = logit_loss_gradient.cross_multiply(weight_output_gradient.transposed());

    // Return dJ/dH_final or the gradient of the final ff layer
    return h_final_gradient;
}

matrix backpropogate_ff_layer(
    ff_layer& layer,
    const matrix& layer_input, // pre-ff input
    const matrix& activation_input, // activation input
    const matrix& activation_output, // activation output
    const matrix& post_layer_gradient
) {
    matrix b2_gradient { 1, post_layer_gradient.size() };
    for (size_t i = 0; i < post_layer_gradient.cols; ++i) {
        const auto row_sum = post_layer_gradient.col_sum(i);
        b2_gradient.set(0, i, row_sum);
    }
    adjust_matrix(layer.b2, b2_gradient);

    matrix w2_gradient = activation_output.transposed().cross_multiply(post_layer_gradient);
    adjust_matrix(layer.w2, w2_gradient);

    const matrix a1_gradient = post_layer_gradient.cross_multiply(layer.w2.transposed());
    matrix z1_gradient = a1_gradient;

    for (size_t i = 0; i < a1_gradient.rows; i++) {
        for (size_t j = 0; j < a1_gradient.cols; j++) {
            const auto z1_value = activation_input.get(i, j);
            const auto self_value = z1_gradient.get(i, j);

            z1_gradient.set(i, j, self_value * (z1_value > 0 ? 1.0f : 0.0f));
        }
    }

    matrix b1_gradient { 1, z1_gradient.size() };
    for (size_t i = 0; i < post_layer_gradient.cols; ++i) {
        const auto row_sum = z1_gradient.col_sum(i);

        b1_gradient.set(0, i, row_sum);
    }
    adjust_matrix(layer.b1, b1_gradient);

    matrix w1_gradient = layer_input.transposed().cross_multiply(z1_gradient);
    adjust_matrix(layer.w1, w1_gradient);

    return z1_gradient.cross_multiply(w1_gradient.transposed());
}

void backpropogate_embedding(llm& model, const std::span<const token_id_t> tokens, const matrix& x_gradient) {
    for (size_t t = 0; t < tokens.size() - 1; t++) {
        const auto &token = tokens[t];
        auto &embedding = model.m_embeddings[token];

        matrix embedding_gradient_row { 1, embedding.data.cols };

        for (size_t i = 0; i < embedding.data.cols; i++) {
            embedding_gradient_row.set(0, i, x_gradient.get(t, i));
        }

        adjust_matrix(embedding.data, embedding_gradient_row);
    }
}

void backpropogate(llm& model, const training_data& data) {
    adjustments = 0.0f;

    matrix previous_final_gradient = backpropogate_logit_row(
        model,
        data.logit_input,
        data.predictions,
        data.tokens
    );

    for (int i = model.m_ff_layer.size() - 1; i >= 0; i--) {
        const auto& layer_result = data.forward_results[i];

        previous_final_gradient = backpropogate_ff_layer(
            model.m_ff_layer.at(i),
            layer_result.layer_input,
            layer_result.activation_input,
            layer_result.activation_output,
            previous_final_gradient
        );
    }

    backpropogate_embedding(model, data.tokens, previous_final_gradient);

    std::cout << "Adjustments: " << adjustments << std::endl;
}