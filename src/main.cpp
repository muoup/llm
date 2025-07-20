#include <iostream>
#include <chrono>

#include "input/input_data.h"
#include "tokenizer/tokenizer.h"
#include "network/neural_net.h"
#include "training/testing.h"
#include "training/training.h"

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

int main() {
    srand(123);

    test_minimal_llm();
    return 0;

    const auto data = get_file_data("../data/talking_heads.txt").substr(0, 25);
    const auto [tokens, token_map] = tokenize(data, 512 - 128);//235 - 128);

    std::cout << "String length: " << data.size() << "\n";
    std::cout << "Token length: " << tokens.size() << "\n";

    llm model { token_map.size(), 4, 32 };
    model.randomize();

    const auto prompt_span = std::span { tokens.begin(), 10 };
    auto prompt = std::vector<token_id_t> { prompt_span.begin(), prompt_span.end() };

    for (size_t i = 0; i < 200; i++) {
        train(model, tokens);

        std::cout << "Iteration " << i + 1 << " complete.\n";
        log_neuron_maxes(model);
        std::cout << '\n';
    }

    std::cout << "Prompt: " << tokens_to_plaintext(token_map, prompt) << '\n';

    for (auto i = 0; i < 10; i++) {
        const auto prediction = model.predict(prompt);
        prompt.push_back(prediction);

        std::cout << "Predicted token: " << prediction << " (" << token_to_plaintext(token_map, token_map[prediction]) << ")\n";
    }

    std::cout << "\n\nPrediction tokens: ";
    std::cout << "\n\nPrediction: " << tokens_to_plaintext(token_map, prompt) << "\n";
}