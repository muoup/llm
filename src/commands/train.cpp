#include "train.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <commands/arg_parser.hpp>
#include <inference/inference.hpp>
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

    if (data_path.empty() || tokenizer_path.empty()
        || output_model_path.empty()) {
        std::cerr << "Usage: ./llm train --data <path> --tokenizer <path> "
                     "--output-model <path> [--input-model <path>] "
                     "[--dataset-type raw|row-based] [-n <amount>]"
                  << std::endl;
        return 1;
    }

    dataset_type type = detect_dataset_type(type_str);

    std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;

    tokenizer _tokenizer = *load_tokenizer(tokenizer_path).or_else([]() {
        std::cerr << "Error loading tokenizer." << std::endl;
        std::abort();

        return (std::optional<tokenizer>)std::nullopt;
    });

    if (_tokenizer.vocab_size() == 0) {
        std::cerr << "Error: Failed to load tokenizer." << std::endl;
        return 1;
    }

    // TODO: Get these from CLI args
    constexpr size_t dimensions = 128;
    constexpr size_t attention_heads = 4;
    constexpr size_t num_layers = 4;

    InferenceModel model = [&]() {
        if (!input_model_path.empty()) {
            std::cout << "Loading existing model from: " << input_model_path
                      << std::endl;
            std::ifstream file(input_model_path);

            auto model = InferenceModel::load(file);
            std::cout << "Successfully loaded model. Parameter count: "
                      << model.parameter_count() << '\n';
            return model;
        } else {
            std::cout << "Creating and randomizing new model." << std::endl;
            InferenceModel model
                = standard_attention_model(dimensions,
                                           _tokenizer.vocab_size(),
                                           num_layers, attention_heads);
            model.randomize();

            std::cout << "New model created. Parameter count: "
                      << model.parameter_count() << '\n';
            return model;
        }
    }();

    if (model.vocab_size() != _tokenizer.token_map.size()) {
        std::cerr << "Model vocabulary size does not match tokenizer size."
                  << std::endl;
        std::abort();
    }

    std::cout << "Starting training process..." << std::endl;

    try {
        size_t specified_size = std::numeric_limits<size_t>::max();

        if (!n_str.empty()) {
            try {
                specified_size = std::stoul(n_str);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for -n: " << n_str
                          << std::endl;
                std::exit(1);
            }
        }

        auto dataset = create_dataset(data_path, type, specified_size);
        std::cout << "Dataset loaded. Type: "
                  << (type == dataset_type::RAW ? "raw" : "row-based")
                  << ". Iterating over rows..." << std::endl;

        constexpr float learning_rate = 0.0001f;
        float rolling_average_loss = 0.0f;

        const size_t n_rows = dataset->size();

        dataset->enumerate([&](size_t i, std::string_view row) {
            auto tokens = encode(_tokenizer, row);
            if (tokens.size() < 2) {
                return;
            }
            const auto input_tokens
                = std::span{ tokens.begin(), tokens.end() - 1 };
            const auto target_tokens
                = std::span{ tokens.begin() + 1, tokens.end() };
            float loss
                = model.train_on(input_tokens, target_tokens, learning_rate);

            constexpr size_t ROLLING_AVG_WINDOW = 100;

            if (i == 0)
                rolling_average_loss = loss;
            
            rolling_average_loss
                = ((ROLLING_AVG_WINDOW - 1) * rolling_average_loss + loss) / ROLLING_AVG_WINDOW;
            

            if (i % 100 == 0) {
                float as_percentage = std::exp(-rolling_average_loss) * 100.0f;

                std::printf(
                    "Row %zu / %zu processed. Rolling Avg Loss: "
                    "%.2f | As Accuracy: %.3f%%\n",
                    i, n_rows, rolling_average_loss, as_percentage);
                std::fflush(stdout);
            }
        });
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range error during training: " << e.what()
                  << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Training complete. Saving model to: " << output_model_path
              << std::endl;

    std::ofstream file_out(output_model_path);
    model.save(file_out);
    return 0;
}
