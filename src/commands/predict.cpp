#include "predict.hpp"

#include <iostream>
#include <string>
#include <fstream>

#include <commands/arg_parser.hpp>
#include <tokenizer/tokenizer.hpp>
#include <inference/inference.hpp>

void print_tokens(const std::span<const token_id_t> tokens) {
    for (auto token : tokens) {
        std::cout << (int) token << " ";
    }
    std::cout << std::endl;
}

int handle_predict(int argc, char* argv[]) {
    std::string model_path = get_arg_value(argc, argv, "--model");
    std::string tokenizer_path = get_arg_value(argc, argv, "--tokenizer");
    std::string prompt = get_arg_value(argc, argv, "--prompt");
    std::string length_str = get_arg_value(argc, argv, "--length");
    std::string temperature_str = get_arg_value(argc, argv, "--temperature");

    if (model_path.empty() || tokenizer_path.empty() || prompt.empty()) {
        std::cerr << "Usage: ./llm predict --model <path> --tokenizer <path> --prompt \"text\" [--length <num>] [--temperature <float>]" << std::endl;
        return 1;
    }

    size_t length = 50; // Default length
    if (!length_str.empty()) {
        length = std::stoul(length_str);
    }

    float temperature = 0.75f; // Default temperature
    if (!temperature_str.empty()) {
        temperature = std::stof(temperature_str);
    }

    std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
    auto _tokenizer = *load_tokenizer(tokenizer_path).or_else([]() {
        std::cerr << "Error loading tokenizer." << std::endl;
        std::abort();

        return (std::optional<tokenizer>) std::nullopt;
    });

    std::cout << "Loading model from: " << model_path << std::endl;
    auto fstream = std::ifstream{ model_path };
    
    if (!fstream.is_open()) {
        std::cerr << "Error: Could not open model file." << std::endl;
        return 1;
    }
    
    auto model = InferenceModel::load(fstream);
    
    std::cout << "Model loaded. Parameter count: " << model.parameter_count() << std::endl;
    
    if (model.vocab_size() != _tokenizer.token_map.size()) {
        std::cerr << "Model vocabulary size does not match tokenizer size." << std::endl;
        return 1;
    }
    
    std::vector<token_id_t> tokens = encode(_tokenizer, prompt);
    std::cout << "Generating: " << prompt << std::flush;
    
    for (size_t i = 0; i < length; ++i) {
        token_id_t next_token = model.predict(tokens, temperature);
        tokens.push_back(next_token);
        
        if (next_token < _tokenizer.token_map.size()) {
            std::cout << _tokenizer.token_map[next_token].text << std::flush;
        } else {
            std::cout << "<UNK>" << std::flush;
        }
    }

    std::cout << "\n\nPrediction complete." << std::endl;
    return 0;
}
