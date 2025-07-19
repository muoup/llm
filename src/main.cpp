#include <iostream>
#include <chrono>

#include "input/input_data.h"
#include "tokenizer/tokenizer.h"
#include "network/neural_net.h"
#include "training/testing.h"
#include "training/training.h"

int main() {
    srand(123);

    // test_minimal_llm();
    //
    // return 0;

    const auto data = get_file_data("../data/talking_heads.txt").substr(0, 100);
    const auto [tokens, token_map] = tokenize(data, 512 - 128);//235 - 128);

    std::cout << "String length: " << data.size() << "\n";
    std::cout << "Token length: " << tokens.size() << "\n";

    llm model { token_map.size(), 4, 32 };
    model.randomize();

    const auto prompt_span = std::span { tokens.begin(), 10 };
    auto prompt = std::vector<token_id_t> { prompt_span.begin(), prompt_span.end() };

    for (size_t i = 0; i < 250; i++) {
        train(model, tokens);

        auto max_neuron = std::max(model.m_logit_layer.w.max(), model.m_logit_layer.b.max());
        auto min_neuron = std::min(model.m_logit_layer.w.min(), model.m_logit_layer.b.min());

        for (size_t j = 0; j < model.m_embeddings.size(); j++) {
            auto& embedding = model.m_embeddings[j];

            max_neuron = std::max(max_neuron, embedding.data.max());
            min_neuron = std::min(min_neuron, embedding.data.min());
            max_neuron = std::max(max_neuron, embedding.data.min());
            min_neuron = std::min(min_neuron, embedding.data.max());
        }

        for (size_t j = 0; j < model.m_ff_layer.size(); j++) {
            auto& layer = model.m_ff_layer[j];

            max_neuron = std::max(max_neuron, layer.w1.max());
            min_neuron = std::min(min_neuron, layer.w1.min());
            max_neuron = std::max(max_neuron, layer.b1.max());
            min_neuron = std::min(min_neuron, layer.b1.min());
            max_neuron = std::max(max_neuron, layer.w2.max());
            min_neuron = std::min(min_neuron, layer.w2.min());
            max_neuron = std::max(max_neuron, layer.b2.max());
            min_neuron = std::min(min_neuron, layer.b2.min());
        }

        std::cout << "Max neuron value: " << max_neuron << '\n';
        std::cout << "Min neuron value: " << min_neuron << '\n';
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