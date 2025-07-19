#include <iostream>
#include <chrono>

#include "input/input_data.h"
#include "tokenizer/tokenizer.h"
#include "network/neural_net.h"
#include "training/training.h"

int main() {
    srand(time(nullptr));

    const auto data = get_file_data("../data/talking_heads.txt");

    std::cout << "Input length: " << data.size() << "\n";

    const auto [tokens, token_map] = tokenize(data, 1000);

    std::cout << "Token length: " << tokens.size() << "\n";

    llm model { token_map.size(), 8, 8 };
    model.randomize();

    const auto shortened_tokens = std::span { tokens.begin(), 10 };
    const auto expected = tokens.at(shortened_tokens.size());

    for (size_t i = 0; i < 100; i++) {
        auto time = std::chrono::high_resolution_clock::now();
        train(model, tokens);
        auto finished = std::chrono::high_resolution_clock::now();

        const auto prediction_matrix = model.prediction_matrix(shortened_tokens).softmax();

        std::cout << "Expected token: " << expected << "\n";
        std::cout << "Predicted token chance: " << prediction_matrix.get(0, expected) << "\n";
        std::cout << "Prediction matrix: \n" << prediction_matrix.to_string() << "\n";
        std::cout << "Training took: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finished - time).count()
                  << "ms\n";
    }
}
