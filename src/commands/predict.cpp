#include "predict.h"

#include <iostream>
#include <string>

#include <commands/arg_parser.h>
#include <tokenizer/tokenizer.h>
#include <network/neural_net.h>

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
    tokenizer tokenizer;
    load_tokenizer(tokenizer, tokenizer_path);

    std::cout << "Loading model from: " << model_path << std::endl;
    // TODO: Infer model dimensions from file instead of hardcoding
    llm model(tokenizer.vocab_size(), 4, 128);
    load_llm(model, model_path);

    std::cout << "\nPrompt: " << prompt << std::endl;
    std::cout << "Generating: " << std::flush;

    std::vector<token_id_t> tokens = encode(tokenizer, prompt);

    for (size_t i = 0; i < length; ++i) {
        token_id_t next_token = model.predict(tokens);
        tokens.push_back(next_token);
        if (next_token < tokenizer.token_map.size()) {
            std::cout << tokenizer.token_map[next_token].text << std::flush;
        }
    }

    std::cout << "\n\nPrediction complete." << std::endl;
    return 0;
}
