#include "init_model.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include <commands/arg_parser.hpp>
#include <inference/feed_forward.hpp>
#include <inference/inference.hpp>
#include <tokenizer/tokenizer.hpp>
#include "model_factories/standard_model.hpp"

ActivationFunction parse_activation(const std::string& str) {
    if (str == "gelu")
        return ActivationFunction::GeLU;
    if (str == "leaky_relu")
        return ActivationFunction::LeakyReLU;
    if (str == "swiglu" || str.empty())
        return ActivationFunction::SwiGLU;
    std::cerr << "Unknown or unspecified activation function: " << str
              << ", defaulting to SwiGLU" << std::endl;
    return ActivationFunction::SwiGLU;
}

const char* activation_to_string(ActivationFunction activation) {
    switch (activation) {
        case ActivationFunction::LeakyReLU:
            return "LeakyReLU";
        case ActivationFunction::GeLU:
            return "GeLU";
        case ActivationFunction::SwiGLU:
            return "SwiGLU";
        default:
            return "Unknown";
    }
}

int handle_init_model(int argc, char* argv[]) {
    std::string output_path = get_arg_value(argc, argv, "--output");
    std::string tokenizer_path = get_arg_value(argc, argv, "--tokenizer");
    std::string dim_str = get_arg_value(argc, argv, "--dimensions");
    std::string heads_str = get_arg_value(argc, argv, "--heads");
    std::string layers_str = get_arg_value(argc, argv, "--layers");
    std::string activation_str = get_arg_value(argc, argv, "--activation");

    if (output_path.empty() || tokenizer_path.empty()) {
        std::cerr << "Usage: ./llm init-model --output <path> --tokenizer "
                     "<path> [--dimensions <n>] [--heads <n>] [--layers <n>] "
                     "[--activation <leaky_relu|gelu|swiglu>]"
                  << std::endl;
        return 1;
    }

    size_t dimensions = dim_str.empty() ? 128 : std::stoul(dim_str);
    size_t heads = heads_str.empty() ? 4 : std::stoul(heads_str);
    size_t layers = layers_str.empty() ? 4 : std::stoul(layers_str);
    ActivationFunction activation = parse_activation(activation_str);

    std::cout << "Loading tokenizer to determine vocab size: " << tokenizer_path
              << std::endl;
    auto _tokenizer = *load_tokenizer(tokenizer_path).or_else([]() {
        std::cerr << "Error loading tokenizer." << std::endl;
        std::abort();
        return (std::optional<tokenizer>)std::nullopt;
    });

    std::cout << "Initializing new model with shape:" << std::endl;
    std::cout << "  Dimensions: " << dimensions << std::endl;
    std::cout << "  Heads: " << heads << std::endl;
    std::cout << "  Layers: " << layers << std::endl;
    std::cout << "  Activation: " << activation_to_string(activation)
              << std::endl;
    std::cout << "  Vocab Size: " << _tokenizer.vocab_size() << std::endl;

    InferenceModel model = standard_attention_model(
        dimensions, _tokenizer.vocab_size(), layers, heads, activation);

    std::cout << "Total Parameters: " << model.parameter_count() << std::endl;
    std::cout << "Saving model to: " << output_path << std::endl;
    std::ofstream file_out(output_path);
    model.save(file_out);

    std::cout << "Model initialization complete." << std::endl;
    return 0;
}
