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
    size_t layers = 4;
    size_t dimensions = 128;

    llm model = [&]() {
        if (!input_model_path.empty()) {
            std::cout << "Loading existing model from: " << input_model_path << std::endl;
            return *load_llm(input_model_path).or_else([&]() {
                std::cerr << "Error loading model." << std::endl;
                std::abort();
                
                return (std::optional<llm>) std::nullopt;
            });
        } else {
            std::cout << "Creating and randomizing new model." << std::endl;
            llm model(_tokenizer.vocab_size(), layers, dimensions);
            model.randomize();
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

        dataset->enumerate([&](size_t i, std::string_view row) {
            auto tokens = encode(_tokenizer, row);
            std::cout << "Training on row " << i + 1 << "/" << dataset->size() << " with " << tokens.size() << " tokens." << std::endl;
            train(model, tokens);
        }, n_rows);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Training complete. Saving model to: " << output_model_path << std::endl;
    save_llm(model, output_model_path);
    
    auto loaded_model = load_llm(output_model_path);
    
    // test if loaded model equals saved model
    for (size_t i = 0; i < model.m_ff_layer.size(); i++) {
        auto &layer = model.m_ff_layer[i];
        auto &loaded_layer = loaded_model->m_ff_layer[i];
        
        if (!layer.w1.equals(loaded_layer.w1) ||
            !layer.b1.equals(loaded_layer.b1) ||
            !layer.w2.equals(loaded_layer.w2) ||
            !layer.b2.equals(loaded_layer.b2)) {
            std::cerr << "Error: Saved and loaded model do not match!" << std::endl;
            return 1;
        }
    }
    
    for (size_t i = 0; i < model.m_attention_layers.size(); i++) {
        auto &layer = model.m_attention_layers[i];
        auto &loaded_layer = loaded_model->m_attention_layers[i];
        
        if (!layer.wq.equals(loaded_layer.wq) ||
            !layer.wk.equals(loaded_layer.wk) ||
            !layer.wv.equals(loaded_layer.wv) ||
            !layer.wo.equals(loaded_layer.wo)) {
            std::cerr << "Error: Saved and loaded model do not match!" << std::endl;
            return 1;
        }
    }
    
    if (!model.m_logit_layer.w.equals(loaded_model->m_logit_layer.w) ||
        !model.m_logit_layer.b.equals(loaded_model->m_logit_layer.b)) {
        std::cerr << "Error: Saved and loaded model do not match!" << std::endl;
        return 1;
    }
    
    return 0;
}
