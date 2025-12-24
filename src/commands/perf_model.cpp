#include "perf_model.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include <commands/arg_parser.hpp>
#include <tokenizer/tokenizer.hpp>
#include <inference/inference.hpp>
#include <dataset/dataset_factory.hpp>

int handle_perf_model(int argc, char* argv[]) {
    std::string model_path = get_arg_value(argc, argv, "--model");
    std::string tokenizer_path = get_arg_value(argc, argv, "--tokenizer");
    std::string data_path = get_arg_value(argc, argv, "--data");
    std::string type_str = get_arg_value(argc, argv, "--dataset-type");

    if (model_path.empty() || tokenizer_path.empty() || data_path.empty()) {
        std::cerr << "Usage: ./llm perf-model --model <path> --tokenizer <path> --data <path> [--dataset-type raw|row-based]" << std::endl;
        return 1;
    }

    dataset_type type = detect_dataset_type(type_str);

    std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
    auto _tokenizer_opt = load_tokenizer(tokenizer_path);
    if (!_tokenizer_opt) {
        std::cerr << "Error loading tokenizer." << std::endl;
        return 1;
    }
    auto& _tokenizer = *_tokenizer_opt;

    std::cout << "Loading model from: " << model_path << std::endl;
    auto fstream = std::ifstream{ model_path, std::ios::binary };
    
    if (!fstream.is_open()) {
        std::cerr << "Error: Could not open model file." << std::endl;
        return 1;
    }
    
    auto model = InferenceModel::load(fstream);
    
    std::cout << "Model loaded. Parameter count: " << model.parameter_count() << std::endl;
    
    std::cout << "Loading dataset from: " << data_path << std::endl;
    auto dataset = create_dataset(data_path, type, 1); // Only need the first row
    
    std::string first_row;
    bool found = false;
    dataset->enumerate([&](size_t i, std::string_view row) {
        if (i == 0) {
            first_row = std::string(row);
            found = true;
        }
    });

    if (!found) {
        std::cerr << "Error: Dataset is empty." << std::endl;
        return 1;
    }

    std::vector<token_id_t> tokens = encode(_tokenizer, first_row);
    if (tokens.size() < 2) {
        std::cerr << "Error: First row of dataset encoded to less than 2 tokens. Need at least 2 for training step." << std::endl;
        return 1;
    }
    
    const auto input_tokens = std::span{ tokens.begin(), tokens.end() - 1 };
    const auto target_tokens = std::span{ tokens.begin() + 1, tokens.end() };

    std::cout << "\n--- Starting Performance Diagnostic (Training Step) ---" << std::endl;
    model.train_on(input_tokens, target_tokens, 0.0f, true);

    std::cout << "\nPerformance diagnostic complete." << std::endl;
    return 0;
}
