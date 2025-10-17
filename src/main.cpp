#include <iostream>

#include <input/input_data.h>
#include <network/tokenizer/tokenizer.h>
#include <network/neural_net.h>
#include <training/testing.h>
#include <training/training.h>

int main() {
    srand(123);
    
#ifdef MATRIX_CHECKS
    std::cout << "Matrix checks enabled.\n";
#endif
    tokenizer tokenizer;

    const auto data = get_file_data("data/talking_heads.txt").substr(0, 250);
    train_tokenizer(tokenizer, data, 512);

    const auto data_tokens = encode(tokenizer, data);
    std::cout << "String length: " << data.size() << "\n";
    std::cout << "Token count: " << data_tokens.size() << "\n";
    
    llm model { tokenizer.vocab_size(), 2, 128 };
    model.randomize();

    const auto prompt_span = std::span { data_tokens.begin(), 5 };
    auto prompt = std::vector<token_id_t> { prompt_span.begin(), prompt_span.end() };

    for (size_t i = 0; i < 1000; i++) {
        train(model, data_tokens);

        std::cout << "Iteration " << i + 1 << " complete.\n";
    }

    log_neuron_maxes(model);
    std::cout << "Prompt: " << decode(tokenizer, prompt) << '\n';
    std::cout << "Prediction: ";

    for (auto i = 0; i < 1000; i++) {
        const auto prediction = model.predict(prompt);
        const auto as_str = decode(tokenizer, std::span { &prediction, 1 });
        
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