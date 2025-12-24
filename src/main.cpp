#include <functional>
#include <iostream>
#include <map>
#include <string>

#include <commands/predict.hpp>
#include <commands/train.hpp>
#include <commands/train_tokenizer.hpp>
#include <commands/init_model.hpp>
#include <commands/perf_model.hpp>

#include <kernels/feed_forward.hpp>
#include <kernels/matrix_kernels.hpp>
#include <util/matrix.hpp>
#include <util/logger.hpp>

int main(int argc, char* argv[]) {
    srand(time(NULL));

#ifdef MATRIX_CHECKS
    logger::set_log_level(LogLevel::DEBUG);
#else
    logger::set_log_level(LogLevel::INFO);
#endif

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
    commands["init-model"] = handle_init_model;
    commands["perf-model"] = handle_perf_model;

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
