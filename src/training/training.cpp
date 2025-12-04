#include "training.hpp"

#include <iostream>
#include <vector>
#include <nodes/neural_net.hpp>
#include <tokenizer/token.hpp>

// ---[ Backpropagation Helpers (from former backpropogation.cpp) ]---

static double total_loss = 0.0;

static void norm_clip(matrix &gradient) {
    constexpr auto max_magnitude = 5.0f;
    const auto max = gradient.absmax();
    if (max > max_magnitude) {
        gradient.scale(max_magnitude / max);
    }
}

static void adjust_matrix(matrix &adjust, const matrix &gradient, float learning_rate) {
    constexpr auto max_magnitude = 5.0f;
    const auto max = gradient.absmax();
    float factor = 1.0f;
    if (max > max_magnitude) {
        factor = max_magnitude / max;
    }

    for (size_t i = 0; i < adjust.rows; ++i) {
        for (size_t j = 0; j < adjust.cols; ++j) {
            const auto delta = gradient.get(i, j) * factor * learning_rate;
            adjust.offset(i, j, -delta);
        }
    }
}

static void regularize_weight_gradient(matrix &gradient, const matrix &weights) {
    constexpr float regularization_strength = 0.01f;
    for (size_t i = 0; i < gradient.rows; ++i) {
        for (size_t j = 0; j < gradient.cols; ++j) {
            const auto weight_value = weights.get(i, j);
            const auto regularization_term = 2 * regularization_strength * weight_value;
            gradient.offset(i, j, regularization_term);
        }
    }
}

// Backpropagation for the final (non-node) layer
static matrix backpropogate_logit_row(llm &model, const matrix &last_ff_output,
                               const matrix &predictions,
                               const std::span<const token_id_t> actual,
                               float learning_rate) {
    matrix logit_loss_gradient{ actual.size() - 1, model.vocab_size() };
    matrix logit_bias_gradient{ 1, model.vocab_size() };

    for (size_t i = 0; i < predictions.rows; ++i) {
        for (size_t j = 0; j < predictions.cols; ++j) {
            const auto delta_loss = predictions.get(i, j) - (j == actual[i + 1] ? 1.0f : 0.0f);
            logit_loss_gradient.set(i, j, delta_loss);
            logit_bias_gradient.offset(0, j, delta_loss);
            if (j == actual[i + 1]) {
                total_loss -= std::log(predictions.get(i, j) + 1e-10f);
            }
        }
    }

    adjust_matrix(model.m_logit_layer.b, logit_bias_gradient, learning_rate);

    matrix h_final_gradient = logit_loss_gradient.cross_multiplied(model.m_logit_layer.w.transposed());
    matrix logit_weight_gradient = last_ff_output.transposed().cross_multiplied(logit_loss_gradient);

    regularize_weight_gradient(logit_weight_gradient, model.m_logit_layer.w);
    adjust_matrix(model.m_logit_layer.w, logit_weight_gradient, learning_rate);

    return h_final_gradient;
}

// Backpropagation for the first (non-node) layer
static void backpropogate_embedding(llm &model,
                             const std::span<const token_id_t> tokens,
                             const matrix &x_gradient,
                             float learning_rate) {
    #pragma omp parallel for
    for (size_t t = 0; t < tokens.size() - 1; t++) {
        const auto &token = tokens[t];
        auto &embedding = model.m_embedding_layer.m_embeddings[token];
        matrix embedding_gradient_row({ 1, embedding.data.cols });
        for (size_t i = 0; i < embedding.data.cols; i++) {
            embedding_gradient_row.set(0, i, x_gradient.get(t, i));
        }
        regularize_weight_gradient(embedding_gradient_row, embedding.data);
        adjust_matrix(embedding.data, embedding_gradient_row, learning_rate);
    }
}


// ---[ Main Training Function ]---

void train(llm& model, const std::span<const token_id_t> input, float learning_rate) {
    total_loss = 0.0;
    const auto truncated_input = std::span { input.begin(), input.end() - 1 };

    // --- FORWARD PASS ---
    matrix acc = model.m_embedding_layer.forward(truncated_input);

    std::vector<matrix> layer_inputs;
    std::vector<std::vector<matrix>> layer_outputs;
    layer_inputs.reserve(model.m_layers.size());
    layer_outputs.reserve(model.m_layers.size());

    for (const auto& layer : model.m_layers) {
        layer_inputs.push_back(acc.clone());
        auto outputs = layer->forward({acc});
        acc = std::move(outputs[0]);
        layer_outputs.push_back(std::move(outputs));
    }

    matrix logit_input = acc.clone();
    matrix predictions = model.m_logit_layer.apply(logit_input).softmax();

    // --- BACKWARD PASS ---
    matrix current_grad = backpropogate_logit_row(model, logit_input, predictions, input, learning_rate);
    norm_clip(current_grad);

    for (int i = model.m_layers.size() - 1; i >= 0; --i) {
        const auto& layer = model.m_layers[i];
        
        // Prepare spans for the backpropagate call
        std::span<const matrix> inputs_span = { &layer_inputs[i], 1 };
        std::span<const matrix> outputs_span = layer_outputs[i];
        std::span<const matrix> gradients_span = { &current_grad, 1 };

        auto input_grad_vec = layer->backpropagate(inputs_span, outputs_span, gradients_span, learning_rate);
        
        current_grad = std::move(input_grad_vec[0]);
        norm_clip(current_grad);
    }
    
    backpropogate_embedding(model, truncated_input, current_grad, learning_rate);

    std::cout << "Loss per token: " << (total_loss / static_cast<double>(truncated_input.size())) << '\n';
}
