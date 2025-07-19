//
// Created by user on 7/17/25.
//

#include "backpropogation.h"

#include <iomanip>
#include <iostream>

#include "../network/neural_net.h"
#include "../tokenizer/token.h"

constexpr auto learning_rate = 0.001f;

float adjustments = 0.0f;
float total_loss = 0.0f;

float norm_clip_factor(const matrix& gradient) {
    constexpr auto max_magnitude = 1.0f;

    const auto max = gradient.min();
    const auto min = gradient.max();

    const auto max_abs = std::max(std::abs(max), std::abs(min));
    float factor = 1.0f;

    if (max_abs > max_magnitude)
        factor = max_magnitude / max_abs;

#ifdef MATRIX_CHECKS
    if (std::isinf(factor) || std::isnan(factor) || factor > 1.0f) {
        std::cout << "Warning: norm_clip_factor resulted in an invalid factor: " << factor << '\n';
        std::cout << "Gradient max: " << max << ", min: " << min << '\n';
        std::cout << "Gradient min: " << gradient.min() << ", max: " << gradient.max() << '\n';
        std::cout << "Gradient:\n" << gradient.to_string(4) << std::endl;
        exit(1);
    }
#endif

    return factor;
}

void adjust_matrix(matrix& adjust, const matrix& gradient) {
    const float factor = norm_clip_factor(gradient);

    for (size_t i = 0; i < adjust.rows; ++i) {
        for (size_t j = 0; j < adjust.cols; ++j) {
            const auto delta = gradient.get(i, j) * factor * learning_rate;

            // if (std::abs(delta) > 5) {
                // throw;
            // }

            adjustments += std::abs(delta);
            adjust.offset(i, j, -delta);
        }
    }
}

matrix backpropogate_logit_row(
    llm& model,
    const matrix& last_ff_output, const matrix& predictions,
    const std::span<const token_id_t> actual
) {
    matrix logit_loss_gradient { actual.size() - 1, model.vocab_size() };

    for (size_t i = 0; i < predictions.rows; ++i) {
        for (size_t j = 0; j < predictions.cols; ++j) {
            const auto delta_loss = predictions.get(i, j) - (j == actual[i] ? 1.0f : 0.0f);
            logit_loss_gradient.set(i, j, delta_loss);
            total_loss += std::pow(delta_loss, 2);
        }
    }

    const matrix weight_output_gradient = last_ff_output.transposed().cross_multiply(logit_loss_gradient);
    adjust_matrix(model.m_logit_layer.w, weight_output_gradient);

    // Return dJ/dH_final or the gradient of the final ff layer
    return logit_loss_gradient.cross_multiply(weight_output_gradient.transposed());
}

matrix backpropogate_ff_layer(
    ff_layer& layer,
    const forward_result &result,
    const matrix& post_layer_gradient
) {
    const matrix& layer_input = result.layer_input;
    const matrix& activation_input = result.activation_input;
    const matrix& activation_output = result.activation_output;

    matrix b2_gradient { 1, post_layer_gradient.size() };
    for (size_t i = 0; i < post_layer_gradient.cols; ++i) {
        const auto col_sum = post_layer_gradient.col_sum(i);
        b2_gradient.set(0, i, col_sum);
    }
    adjust_matrix(layer.b2, b2_gradient);

    const matrix a1_t = activation_output.transposed();
    const matrix w2_gradient = a1_t.cross_multiply(post_layer_gradient);

    adjust_matrix(layer.w2, w2_gradient.scaled(-1));

    const matrix a1_gradient = post_layer_gradient.cross_multiply(layer.w2.transposed());
    matrix z1_gradient = a1_gradient;

    for (size_t i = 0; i < a1_gradient.rows; i++) {
        for (size_t j = 0; j < a1_gradient.cols; j++) {
            const auto z1_value = activation_input.get(i, j);
            const auto self_value = z1_gradient.get(i, j);

            z1_gradient.set(i, j, self_value * (z1_value > 0 ? 1.0f : 0.0f));
        }
    }

    matrix b1_gradient{1, z1_gradient.size()};
    for (size_t i = 0; i < z1_gradient.cols; ++i) {
        const auto row_sum = z1_gradient.col_sum(i);
        b1_gradient.set(0, i, row_sum);
    }
    adjust_matrix(layer.b1, b1_gradient.scaled(-1));

    const matrix w1_gradient = layer_input.transposed().cross_multiply(z1_gradient).scaled(-1);
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
    total_loss = 0.0f;

    matrix previous_final_gradient = backpropogate_logit_row(
        model,
        data.logit_input,
        data.predictions,
        data.tokens
    );

    for (int i = model.m_ff_layer.size() - 1; i >= 0; i--) {
        previous_final_gradient = backpropogate_ff_layer(
            model.m_ff_layer.at(i),
            data.forward_results[i],
            previous_final_gradient
        );
    }
    //
    // backpropogate_embedding(model, data.tokens, previous_final_gradient);

    std::cout << "Adjustments: " << adjustments << '\n';

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Total Loss: " << total_loss << '\n';
}