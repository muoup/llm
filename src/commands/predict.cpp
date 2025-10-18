#include "predict.h"

#include <iostream>
#include <string>
#include <fstream>

#include <commands/arg_parser.h>
#include <tokenizer/tokenizer.h>
#include <network/neural_net.h>

void print_tokens(const std::span<const token_id_t> tokens) {
    for (auto token : tokens) {
        std::cout << (int) token << " ";
    }
    std::cout << std::endl;
}

// tokens_span should be the original input tokens used for training
void check_teacher_forcing(const llm &model, const std::span<const token_id_t> tokens) {
    // Build truncated input exactly like training (exclude last token)
    std::vector<token_id_t> truncated(tokens.begin(), tokens.end() - 1);
    auto predictions = model.prediction_matrix({truncated.data(), truncated.size()}); // returns logits -> we want softmax
    predictions.softmax();

    int correct = 0;
    for (size_t i = 0; i < truncated.size(); ++i) {
        // golden next token is tokens[i+1]
        size_t max_idx = 0;
        for (size_t j = 1; j < predictions.cols; ++j) {
            if (predictions.get(i, j) > predictions.get(i, max_idx)) max_idx = j;
        }
        bool is_correct = (max_idx == tokens[i + 1]);
        std::cout << "pos " << i << " pred=" << max_idx << " gold=" << tokens[i + 1] << " correct=" << is_correct << "\n";
        if (is_correct) ++correct;
    }
    std::cout << "Teacher-forcing accuracy: " << correct << " " << truncated.size() << "\n";
}

int handle_predict(int argc, char* argv[]) {
    std::string model_path = get_arg_value(argc, argv, "--model");
    std::string tokenizer_path = get_arg_value(argc, argv, "--tokenizer");
    std::string prompt = get_arg_value(argc, argv, "--prompt");
    std::string length_str = get_arg_value(argc, argv, "--length");

    if (model_path.empty() || tokenizer_path.empty() || prompt.empty()) {
        std::cerr << "Usage: ./llm predict --model <path> --tokenizer <path> --prompt \"text\" [--length <num>]" << std::endl;
        return 1;
    }

    size_t length = 50; // Default length
    if (!length_str.empty()) {
        length = std::stoul(length_str);
    }

    std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
    auto _tokenizer = *load_tokenizer(tokenizer_path).or_else([]() {
        std::cerr << "Error loading tokenizer." << std::endl;
        std::abort();

        return (std::optional<tokenizer>) std::nullopt;
    });

    std::cout << "Loading model from: " << model_path << std::endl;
    auto model = *load_llm(model_path).or_else([]() {
        std::cerr << "Error loading model." << std::endl;
        std::abort();

        return (std::optional<llm>) std::nullopt;
    });

    if (model.vocab_size() != _tokenizer.token_map.size()) {
        std::cerr << "Model vocabulary size does not match tokenizer size." << std::endl;
        return 1;
    }
    
    auto file = std::ifstream("datasets/wiki-trunc/th.txt");
    auto contents = std::string((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    
    auto test_tokens = encode(_tokenizer, contents);
    check_teacher_forcing(model, test_tokens);
    
    std::cout << "Test tokens (" << test_tokens.size() << "): ";
    print_tokens(test_tokens);
    
    std::cout << "\nPrompt: " << prompt << std::endl;
    std::vector<token_id_t> tokens = encode(_tokenizer, prompt);
    std::cout << "Initial tokens (" << tokens.size() << "): ";
    print_tokens(tokens);
    
    std::cout << "Generating: " << std::flush;

    for (size_t i = 0; i < length; ++i) {
        token_id_t next_token = model.predict(tokens);
        tokens.push_back(next_token);
        
        if (next_token < _tokenizer.token_map.size()) {
            std::cout << _tokenizer.token_map[next_token].text << std::flush;
        }
    }

    std::cout << "\n\nPrediction complete." << std::endl;
    return 0;
}
