#include "backpropogation.h"

#include <iostream>

#include <network/neural_net.h>
#include <tokenizer/token.h>

constexpr float learning_rate = 0.0001f;
constexpr float regularization_strength = 0.01f;

float adjustments = 0.0f;
double total_loss = 0.0f;

float norm_clip_factor(const matrix& gradient) {
    constexpr auto max_magnitude = 5.0f;

    const auto max = gradient.absmax();
    float factor = 1;

    if (max > max_magnitude)
        factor = max_magnitude / max;

#ifdef MATRIX_CHECKS
    if (std::isinf(factor) || std::isnan(factor) || factor > 1.0f) {
        std::cout << "Warning: norm_clip_factor resulted in an invalid factor: " << factor << '\n';
        std::cout << "Gradient max: " << max << '\n';
        std::cout << "Gradient:\n" << gradient.to_string(4) << std::endl;
        exit(1);
    }
#endif

    return factor;
}

void norm_clip(matrix& gradient) {
    const float factor = norm_clip_factor(gradient);
    gradient.scale(factor);
}

void regularize_weight_gradient(matrix& gradient, const matrix& weights) {
    norm_clip(gradient);

    for (size_t i = 0; i < gradient.rows; ++i) {
        for (size_t j = 0; j < gradient.cols; ++j) {
            const auto weight_value = weights.get(i, j);
            const auto regularization_term = 2 * regularization_strength * weight_value;

            gradient.set(i, j, gradient.get(i, j) + regularization_term);
        }
    }
}

void adjust_matrix(matrix& adjust, const matrix& gradient) {
    const float factor = norm_clip_factor(gradient);

    for (size_t i = 0; i < adjust.rows; ++i) {
        for (size_t j = 0; j < adjust.cols; ++j) {
            const auto delta = gradient.get(i, j) * factor * learning_rate;

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
    matrix logit_bias_gradient { 1, model.vocab_size() };

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

    adjust_matrix(model.m_logit_layer.b, logit_bias_gradient);

    const matrix h_final_gradient = logit_loss_gradient.cross_multiply(model.m_logit_layer.w.transposed());
    matrix logit_weight_gradient = last_ff_output.transposed().cross_multiply(logit_loss_gradient);

    regularize_weight_gradient(logit_weight_gradient, model.m_logit_layer.w);
    adjust_matrix(model.m_logit_layer.w, logit_weight_gradient);

    // Return dJ/dH_final or the gradient of the final ff layer
    return h_final_gradient;
}

matrix backpropogate_ff_layer(
    ff_layer& layer,
    const forward_result &result,
    const matrix& post_layer_gradient
) {
    const matrix& layer_input = result.layer_input;
    const matrix& activation_input = result.activation_input;
    const matrix& activation_output = result.activation_output;

    matrix b2_gradient { 1, post_layer_gradient.cols };
    for (size_t i = 0; i < b2_gradient.cols; ++i) {
        const auto col_sum = post_layer_gradient.col_sum(i);
        b2_gradient.set(0, i, col_sum);
    }
    adjust_matrix(layer.b2, b2_gradient);

    const matrix a1_t = activation_output.transposed();
    matrix w2_gradient = a1_t.cross_multiply(post_layer_gradient);

    regularize_weight_gradient(w2_gradient, layer.w2);
    adjust_matrix(layer.w2, w2_gradient);

    const matrix a1_gradient = post_layer_gradient.cross_multiply(layer.w2.transposed());
    matrix z1_gradient { a1_gradient.rows, a1_gradient.cols };

    for (size_t i = 0; i < z1_gradient.rows; i++) {
        for (size_t j = 0; j < z1_gradient.cols; j++) {
            const auto z1_value = activation_input.get(i, j);
            const auto self_value = a1_gradient.get(i, j);

            z1_gradient.set(i, j, self_value * (z1_value > 0 ? 1.0f : 0.01f));
        }
    }

    matrix b1_gradient{1, z1_gradient.cols };
    for (size_t i = 0; i < z1_gradient.cols; ++i) {
        const auto row_sum = z1_gradient.col_sum(i);
        b1_gradient.set(0, i, row_sum);
    }
    adjust_matrix(layer.b1, b1_gradient);

    matrix w1_gradient = layer_input.transposed().cross_multiply(z1_gradient);
    regularize_weight_gradient(w1_gradient, layer.w1);

    adjust_matrix(layer.w1, w1_gradient);

    auto input_gradient = z1_gradient
        .cross_multiply(layer.w1.transposed());
    return input_gradient;
}

void backpropogate_embedding(llm& model, const std::span<const token_id_t> tokens, const matrix& x_gradient) {
#pragma omp parallel for
    for (size_t t = 0; t < tokens.size() - 1; t++) {
        const auto &token = tokens[t];
        auto &embedding = model.m_embedding_layer.m_embeddings[token];

        matrix embedding_gradient_row { 1, embedding.data.cols };
        for (size_t i = 0; i < embedding.data.cols; i++) {
            embedding_gradient_row.set(0, i, x_gradient.get(t, i));
        }

        regularize_weight_gradient(embedding_gradient_row, embedding.data);
        adjust_matrix(embedding.data, embedding_gradient_row);
    }
}

matrix backpropagate_attention_layer(
    attention_layer& layer,
    const matrix& layer_input,
    const attention_forward_result &result,
    const matrix& post_layer_gradient
) {
    // Unpack forward pass results (including stored pre-projection output)
    const auto& q = result.q;
    const auto& k = result.k;
    const auto& v = result.v;
    const auto& scores = result.scores;
    const auto& output = result.output;

    // Backprop through output projection (wo)
    matrix output_t = output.transposed();
    matrix wo_gradient = output_t.cross_multiply(post_layer_gradient);
    matrix scores_gradient = post_layer_gradient.cross_multiply(layer.wo.transposed());
    regularize_weight_gradient(wo_gradient, layer.wo);
    adjust_matrix(layer.wo, wo_gradient);

    // Backprop through weighted sum: output = scores * v
    matrix scores_t = scores.transposed();
    matrix v_gradient = scores_t.cross_multiply(scores_gradient);
    matrix scores_gradient_v = scores_gradient.cross_multiply(v.transposed());

    // Backprop through softmax (correct Jacobian)
    matrix softmax_gradient { scores_gradient_v.rows, scores_gradient_v.cols };

#pragma omp parallel for
    for (size_t i = 0; i < scores.rows; ++i) {
        // Vectorized softmax backprop per-row:
        // For a row: s = softmax(scores_row), g = dL/ds (scores_gradient_v row)
        // s_dot = dot(g, s)
        // dL/dz = s * (g - s_dot)
        float s_dot = 0.0f;
        for (size_t l = 0; l < scores.cols; ++l) {
            s_dot += scores_gradient_v.get(i, l) * scores.get(i, l);
        }

#ifdef ATTENTION_DEBUG
            // Optional per-row diagnostics: max probability and entropy
            float row_max = scores.get(i, 0);
            float entropy = 0.0f;
            for (size_t l = 0; l < scores.cols; ++l) {
                const float s = scores.get(i, l);
                if (s > row_max) row_max = s;
                entropy -= s * std::log(s + 1e-12f);
            }
            std::cout << "[ATTN_SOFTMAX_ROW] row=" << i << " max=" << row_max
                      << " entropy=" << entropy << " s_dot=" << s_dot << std::endl;
#endif

        for (size_t j = 0; j < scores.cols; ++j) {
            const float s_j = scores.get(i, j);
            const float g_j = scores_gradient_v.get(i, j);
            softmax_gradient.set(i, j, s_j * (g_j - s_dot));
        }
    }

    // Backprop through scaling (scale was applied before softmax)
    const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
    softmax_gradient.scale(scale);

    // Backprop through attention scores to Q,K
    matrix q_gradient = softmax_gradient.cross_multiply(k);
    matrix k_gradient = softmax_gradient.transposed().cross_multiply(q);

    // Backprop through Q, K, V projections to their weight matrices
    matrix layer_input_t = layer_input.transposed();

    matrix wq_gradient = layer_input_t.cross_multiply(q_gradient);
    matrix wk_gradient = layer_input_t.cross_multiply(k_gradient);
    matrix wv_gradient = layer_input_t.cross_multiply(v_gradient);

    regularize_weight_gradient(wq_gradient, layer.wq);
    adjust_matrix(layer.wq, wq_gradient);
    regularize_weight_gradient(wk_gradient, layer.wk);
    adjust_matrix(layer.wk, wk_gradient);
    regularize_weight_gradient(wv_gradient, layer.wv);
    adjust_matrix(layer.wv, wv_gradient);

#ifdef ATTENTION_DEBUG
    // Optional debug logging for attention backpropagation gradients.
    // Define ATTENTION_DEBUG at compile time to enable these prints.
    std::cout << "[ATTN_BACKPROP] wo_grad.absmax=" << wo_gradient.absmax()
              << " softmax_grad.absmax=" << softmax_gradient.absmax()
              << " v_act_grad.absmax=" << v_gradient.absmax()
              << " wq_grad.absmax=" << wq_gradient.absmax()
              << " wk_grad.absmax=" << wk_gradient.absmax()
              << " wv_grad.absmax=" << wv_gradient.absmax()
              << " post_grad.absmax=" << post_layer_gradient.absmax()
              << std::endl;
#endif

    // Gradient of the input: sum of contributions via each projection
    matrix input_gradient = q_gradient.cross_multiply(layer.wq.transposed());
    input_gradient.add(k_gradient.cross_multiply(layer.wk.transposed()));
    input_gradient.add(v_gradient.cross_multiply(layer.wv.transposed()));

    // Account for the residual (skip) connection: the layer input was
    // X, the attention produced O, and the forward did Y = O + X (residual).
    // The gradient flowing into this function (post_layer_gradient) is dL/dY.
    // Part of that gradient flows directly back to the input via the identity
    // skip connection and must be added to the gradient coming through
    // the attention projections.
    input_gradient.add(post_layer_gradient);

    return input_gradient;
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

    norm_clip(previous_final_gradient);

    for (int i = model.m_ff_layer.size() - 1; i >= 0; i--) {
        previous_final_gradient = backpropogate_ff_layer(
            model.m_ff_layer.at(i),
            data.forward_results.at(i),
            previous_final_gradient
        );
        norm_clip(previous_final_gradient);

        previous_final_gradient = backpropagate_attention_layer(
            model.m_attention_layers.at(i),
            data.attention_inputs.at(i),
            data.attention_forward_results.at(i),
            previous_final_gradient
        );
        norm_clip(previous_final_gradient);
    }

    backpropogate_embedding(model, data.tokens, previous_final_gradient);

    std::cout << "Adjustments: " << adjustments << '\n';
    std::cout << "Loss per token: " << (total_loss / static_cast<double>(data.tokens.size() - 1)) << '\n';
}
