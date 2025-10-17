#include "train.h"
#include "training/training.h"

#include <iostream>
#include <string>

#include <commands/arg_parser.h>
#include <tokenizer/tokenizer.h>
#include <network/neural_net.h>

#include <dataset/dataset_factory.h>

int handle_train(int argc, char* argv[]) {
    std::string data_path = get_arg_value(argc, argv, "--data");
    std::string tokenizer_path = get_arg_value(argc, argv, "--tokenizer");
    std::string output_model_path = get_arg_value(argc, argv, "--output-model");
    std::string input_model_path = get_arg_value(argc, argv, "--input-model");

    if (data_path.empty() || tokenizer_path.empty() || output_model_path.empty()) {
        std::cerr << "Usage: ./llm train --data <path> --tokenizer <path> --output-model <path> [--input-model <path>]" << std::endl;
        return 1;
    }

    std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
    tokenizer tokenizer;
    load_tokenizer(tokenizer, tokenizer_path);
    if (tokenizer.vocab_size() == 0) {
        std::cerr << "Error: Failed to load tokenizer." << std::endl;
        return 1;
    }

    // TODO: Get these from CLI args
    size_t layers = 4;
    size_t dimensions = 128;

    llm model(tokenizer.vocab_size(), layers, dimensions);

    if (!input_model_path.empty()) {
        std::cout << "Loading existing model from: " << input_model_path << std::endl;
        load_llm(model, input_model_path);
    } else {
        std::cout << "Creating and randomizing new model." << std::endl;
        model.randomize();
    }

    std::cout << "Starting training process..." << std::endl;
    
    auto dataset = create_dataset(data_path, dataset_type::ROW_BASED);
    std::cout << "Dataset loaded. Iterating over rows..." << std::endl;

    dataset->for_each([&](std::string_view row) {
        auto tokens = encode(tokenizer, row);
        train(model, tokens);
    });

    std::cout << "Training complete. Saving model to: " << output_model_path << std::endl;
    save_llm(model, output_model_path);
    return 0;
}
