#include "train_tokenizer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <commands/arg_parser.h>
#include <tokenizer/tokenizer.h>

int handle_train_tokenizer(int argc, char* argv[]) {
    std::string corpus_path = get_arg_value(argc, argv, "--corpus");
    std::string output_path = get_arg_value(argc, argv, "--output");
    std::string vocab_size_str = get_arg_value(argc, argv, "--vocab-size");

    if (corpus_path.empty() || output_path.empty() || vocab_size_str.empty()) {
        std::cerr << "Usage: ./llm train-tokenizer --corpus <path> --output <path> --vocab-size <size>" << std::endl;
        return 1;
    }

    size_t vocab_size = std::stoul(vocab_size_str);

    std::cout << "Training tokenizer..." << std::endl;
    std::cout << "  Corpus: " << corpus_path << std::endl;
    std::cout << "  Vocab size: " << vocab_size << std::endl;

    std::ifstream file(corpus_path);
    if (!file) {
        std::cerr << "Error: Could not open corpus file: " << corpus_path << std::endl;
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string corpus = buffer.str();

    tokenizer tokenizer;
    train_tokenizer(tokenizer, corpus, vocab_size);
    save_tokenizer(tokenizer, output_path);

    std::cout << "Tokenizer training complete. Saved to: " << output_path << std::endl;
    return 0;
}
