#include <iostream>
#include <string>
#include <map>
#include <functional>

#include "commands/train_tokenizer.h"
#include "commands/train.h"
#include "commands/predict.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./llm <command> [options]" << std::endl;
        std::cerr << "Commands: train-tokenizer, train, predict" << std::endl;
        return 1;
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
