#include "train_tokenizer.h"

#include <iostream>
#include <string>

#include <commands/arg_parser.h>
#include <tokenizer/tokenizer.h>

#include "../dataset/dataset_factory.h"

int handle_train_tokenizer(int argc, char* argv[]) {
    std::string corpus_path = get_arg_value(argc, argv, "--corpus");
    std::string output_path = get_arg_value(argc, argv, "--output");
    std::string vocab_size_str = get_arg_value(argc, argv, "--vocab-size");
    std::string type_str = get_arg_value(argc, argv, "--dataset-type");

    if (corpus_path.empty() || output_path.empty() || vocab_size_str.empty()) {
        std::cerr << "Usage: ./llm train-tokenizer --corpus <path> --output <path> --vocab-size <size> [--dataset-type raw|row-based]" << std::endl;
        return 1;
    }

    dataset_type type = detect_dataset_type(type_str);

    std::cout << "Training tokenizer..." << std::endl;
    std::cout << "  Corpus: " << corpus_path << std::endl;
    std::cout << "  Dataset type: " << (type == dataset_type::RAW ? "raw" : "row-based") << std::endl;

    std::string corpus_data;
    try {
        auto dataset = create_dataset(corpus_path, type);
        dataset->for_each([&corpus_data](std::string_view row) {
            corpus_data.append(row);
            corpus_data.append("\n");
        });
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    size_t vocab_size = std::stoul(vocab_size_str);

    std::cout << "Training tokenizer..." << std::endl;
    std::cout << "  Corpus: " << corpus_path << std::endl;
    std::cout << "  Dataset type: " << (type == dataset_type::RAW ? "raw" : "row-based") << std::endl;

    tokenizer tokenizer;
    auto dataset = create_dataset(corpus_path, type);

    std::printf("Training tokenizer from row-based dataset...\b");
    std::printf("  Corpus: %s\n", corpus_path.data());
    
    dataset->for_each([&](std::string_view row) {
        train_tokenizer(tokenizer, row, vocab_size);
    });

    save_tokenizer(tokenizer, output_path);

    std::printf("Tokenizer training complete. Saved to: %s\n", output_path.data());
    return 0;
}
