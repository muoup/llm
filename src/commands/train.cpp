#include "train.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>

#include <inference/inference.hpp>
#include <commands/arg_parser.hpp>
#include <tokenizer/tokenizer.hpp>

#include <dataset/dataset_factory.hpp>
#include "model_factories/standard_model.hpp"

int handle_train(int argc, char* argv[]) {
    std::string data_path = get_arg_value(argc, argv, "--data");
    std::string tokenizer_path = get_arg_value(argc, argv, "--tokenizer");
    std::string output_model_path = get_arg_value(argc, argv, "--output-model");
    std::string input_model_path = get_arg_value(argc, argv, "--input-model");
    std::string type_str = get_arg_value(argc, argv, "--dataset-type");
    std::string n_str = get_arg_value(argc, argv, "-n");

    if (data_path.empty() || tokenizer_path.empty() || output_model_path.empty()) {
        std::cerr << "Usage: ./llm train --data <path> --tokenizer <path> --output-model <path> [--input-model <path>] [--dataset-type raw|row-based] [-n <amount>]" << std::endl;
        return 1;
    }

    size_t n_rows = 0;
    if (!n_str.empty()) {
        try {
            n_rows = std::stoul(n_str);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid value for -n: " << n_str << std::endl;
            return 1;
        }
    }

    dataset_type type = detect_dataset_type(type_str);

    std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
    
    tokenizer _tokenizer = *load_tokenizer(tokenizer_path).or_else([]() {
        std::cerr << "Error loading tokenizer." << std::endl;
        std::abort();
        
        return (std::optional<tokenizer>) std::nullopt;
    });
    
    if (_tokenizer.vocab_size() == 0) {
        std::cerr << "Error: Failed to load tokenizer." << std::endl;
        return 1;
    }

    // TODO: Get these from CLI args
    size_t dimensions = 128;

    InferenceModel model = [&]() {
        if (!input_model_path.empty()) {
            std::cout << "Loading existing model from: " << input_model_path << std::endl;
            std::ifstream file(input_model_path);
            
            auto model = InferenceModel::load(file);
            std::cout << "Successfully loaded model. Parameter count: " << model.parameter_count() << '\n';
            return model;
        } else {
            std::cout << "Creating and randomizing new model." << std::endl;
            InferenceModel model = create_standard_model(dimensions, _tokenizer.vocab_size(), 8, 8);
            model.randomize();
            
            std::cout << "New model created. Parameter count: " << model.parameter_count() << '\n';
            return model;
        }
    }();
    
    if (model.vocab_size() != _tokenizer.token_map.size()) {
        std::cerr << "Model vocabulary size does not match tokenizer size." << std::endl;
        std::abort();
    }

    std::cout << "Starting training process..." << std::endl;

    try {
        auto dataset = create_dataset(data_path, type);
        std::cout << "Dataset loaded. Type: " << (type == dataset_type::RAW ? "raw" : "row-based") << ". Iterating over rows..." << std::endl;
 
        float learning_rate = 0;
        
        dataset->enumerate([&](size_t i, std::string_view row) {
            auto tokens = encode(_tokenizer, row);
            const auto truncated_input = std::span { tokens.begin(), tokens.end() - 1 };
            float loss = model.train_on(truncated_input, tokens, 0.0001f);
            
            std::cout << "Row " << i << "/" << dataset->size() << " processed. Loss: " << loss << std::endl;
            learning_rate = 0.00005f * loss;
        }, n_rows);
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range error during training: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    auto file = std::ifstream(data_path);
    if (!file.is_open()) {
        std::cerr << "Error opening data file for final encoding: " << data_path << std::endl;
        return 1;
    }
    auto contents = std::string((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    
    auto test_tokens = encode(_tokenizer, contents);
    
    std::cout << "Training complete. Saving model to: " << output_model_path << std::endl;
    
    std::ofstream file_out(output_model_path);
    model.save(file_out);
    return 0;
}
