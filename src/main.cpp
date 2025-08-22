#include <iostream>
#include <chrono>

#include "input/input_data.h"
#include "tokenizer/tokenizer.h"
#include "network/neural_net.h"
#include "training/testing.h"
#include "training/training.h"

int main() {
    srand(123);

    const auto data = get_file_data("../data/talking_heads.txt").substr(0, 250);
    const auto [tokens, token_map] = tokenize(data, 512 - 128);//235 - 128);

    std::cout << "String length: " << data.size() << "\n";
    std::cout << "Token length: " << tokens.size() << "\n";

    llm model { token_map.size(), 2, 128 };
    model.randomize();

    const auto prompt_span = std::span { tokens.begin(), 5 };
    auto prompt = std::vector<token_id_t> { prompt_span.begin(), prompt_span.end() };

    for (size_t i = 0; i < 1000; i++) {
        train(model, tokens);

        std::cout << "Iteration " << i + 1 << " complete.\n";
    }

    log_neuron_maxes(model);
    std::cout << "Prompt: " << tokens_to_plaintext(token_map, prompt) << '\n';
    std::cout << "Prediction: ";

    for (auto i = 0; i < 1000; i++) {
        const auto prediction = model.predict(prompt);
        const auto as_str = token_to_plaintext(token_map, token_map[prediction]);
        prompt.push_back(prediction);

        // std::cout << "Predicted token: " << prediction << " (" << as_str << ")\n";
        std::cout << as_str;

        if (i % 100 == 99) {
            std::cout << '\n';
        }
    }

    // std::cout << "\n\nPrediction tokens: ";
    // std::cout << "\n\nPrediction: " << ss.str() << '\n';
}