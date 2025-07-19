#include <iostream>
#include <chrono>

#include "input/input_data.h"
#include "tokenizer/tokenizer.h"
#include "network/neural_net.h"
#include "training/training.h"

int main() {
    srand(time(nullptr));

    const auto data = get_file_data("../data/talking_heads.txt").substr(0, 25);
    const auto [tokens, token_map] = tokenize(data, 256 - 128);//235 - 128);

    std::cout << "String length: " << data.size() << "\n";
    std::cout << "Token length: " << tokens.size() << "\n";

    llm model { token_map.size(), 2, 32 };
    model.randomize();

    const auto prompt_span = std::span { tokens.begin(), 10 };
    auto prompt = std::vector<token_id_t> { prompt_span.begin(), prompt_span.end() };

    for (size_t i = 0; i < 1000; i++) {
        train(model, tokens);
    }

    std::cout << "Prompt: " << tokens_to_plaintext(token_map, prompt) << '\n';

    for (auto i = 0; i < 2; i++) {
        const auto prediction = model.predict(prompt);
        prompt.push_back(prediction);

        std::cout << "Predicted token: " << prediction << " (" << token_to_plaintext(token_map, token_map[prediction]) << ")\n";
    }

    std::cout << "\n\nPrediction tokens: ";
    std::cout << "\n\nPrediction: " << tokens_to_plaintext(token_map, prompt) << "\n";
}