#include <functional>
#include <iostream>
#include <map>
#include <string>

#include <commands/predict.hpp>
#include <commands/train.hpp>
#include <commands/train_tokenizer.hpp>

#include <kernels/feed_forward.hpp>
#include <kernels/matrix_kernels.hpp>
#include <util/matrix.hpp>

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc < 2) {
        std::cerr << "Usage: ./llm <command> [options]" << std::endl;
        std::cerr << "Commands: train-tokenizer, train, predict" <<
        std::endl; return 1;
    }

    using CommandHandler = std::function<int(int, char*[])>;
    std::map<std::string, CommandHandler> commands;

    commands["train-tokenizer"] = handle_train_tokenizer;
    commands["train"] = handle_train;
    commands["predict"] = handle_predict;

    std::string command = argv[1];

    auto it = commands.find(command);
    if (it != commands.end()) {
        // Call the handler
        return it->second(argc, argv);
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        return 1;
    }
}
