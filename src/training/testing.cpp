//
// Created by user on 7/19/25.
//

#include "testing.h"

#include <iostream>

#include "training.h"
#include "../network/neural_net.h"


void log_neuron_maxes(const llm& model) {
    auto embedding_max = 0.0f;

    for (const auto& embedding : model.m_embeddings) {
        embedding_max = std::max(embedding_max, embedding.data.absmax());
    }

    std::cout << "Embedding max: " << embedding_max << "\n";

    for (const auto& layer : model.m_ff_layer) {
        const auto w1_max = layer.w1.absmax();
        const auto w2_max = layer.w2.absmax();
        const auto b1_max = layer.b1.absmax();
        const auto b2_max = layer.b2.absmax();

        std::cout << "Layer maxes: "
                  << "W1: " << w1_max << ", "
                  << "W2: " << w2_max << ", "
                  << "B1: " << b1_max << ", "
                  << "B2: " << b2_max << '\n';
    }

    const auto logit_w_max = model.m_logit_layer.w.absmax();
    const auto logit_b_max = model.m_logit_layer.b.absmax();

    std::cout << "Logit layer maxes: "
              << "W: " << logit_w_max << ", "
              << "B: " << logit_b_max << '\n';
}

void print_prediction_process(const llm& model, const std::span<const token_id_t> tokens) {
    const auto embed = model.embed_tokens(tokens);
    const auto l1_forward = model.forward_l1(embed, 0);
    const auto activated = model.activate(l1_forward);
    const auto l2_forward = model.forward_l2(activated, 0);
    const auto prediction = model.generate_logits(l2_forward).softmax();

    std::cout << "embed:\n" << embed.to_string(4) << '\n';
    std::cout << "l1_forward:\n" << l1_forward.to_string(4) << '\n';
    std::cout << "activated:\n" << activated.to_string(4) << '\n';
    std::cout << "l2_forward:\n" << l2_forward.to_string(4) << '\n';
    std::cout << "prediction:\n" << prediction.to_string(4) << '\n';
}

void test_fixed_llm() {
    constexpr auto seed = 1752983838; // time(nullptr);
    // std::cout << "Seed: " << seed << '\n';

    std::vector tokens { static_cast<token_id_t>(0), static_cast<token_id_t>(1) };

    llm model { 2, 1, 1, 1 };

    model.m_embeddings[0].data.set(0, 0, 0);

    model.m_ff_layer[0].w1.set(0, 0, 0.5);
    model.m_ff_layer[0].b1.set(0, 0, -0.5);

    model.m_ff_layer[0].w2.set(0, 0, -0.5);
    model.m_ff_layer[0].b2.set(0, 0, -0.5);

    model.m_logit_layer.w.set(0, 0, 0.5);
    model.m_logit_layer.w.set(0, 1, -0.5);
    model.m_logit_layer.b.set(0, 0, -0.25);
    model.m_logit_layer.b.set(0, 1, 0.5);

    train(model, tokens);
}

void test_minimal_llm() {
    std::vector tokens { static_cast<token_id_t>(0), static_cast<token_id_t>(1) };

    llm model { 2, 2, 2, 2 };
    model.randomize();

    for (int i = 0; i < 50000; i++) {
        train(model, tokens);
        log_neuron_maxes(model);
    }

    std::cout << "Final Model: \n" << model.to_string() << '\n';
}