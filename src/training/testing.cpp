//
// Created by user on 7/19/25.
//

#include "testing.h"

#include <iostream>

#include "training.h"
#include "../network/neural_net.h"


void log_neuron_maxes(const llm& model) {
    auto embedding_max = 0.0f;

    for (const auto& embedding : model.m_embedding_layer.m_embeddings) {
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