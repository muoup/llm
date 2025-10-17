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

    if (corpus_path.empty() || output_path.empty() || vocab_size_str.empty()) {
        std::cerr << "Usage: ./llm train-tokenizer --corpus <path> --output <path> --vocab-size <size>" << std::endl;
        return 1;
    }

    size_t vocab_size = std::stoul(vocab_size_str);
    tokenizer tokenizer;
   
    auto dataset = create_dataset(corpus_path, dataset_type::ROW_BASED);

    std::printf("Training tokenizer from row-based dataset...\b");
    std::printf("  Corpus: %s\n", corpus_path.data());
    
    dataset->for_each([&](std::string_view row) {
        train_tokenizer(tokenizer, row, vocab_size);
    });

    save_tokenizer(tokenizer, output_path);

    std::printf("Tokenizer training complete. Saved to: %s\n", output_path.data());
    return 0;
}
