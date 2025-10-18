#include "arg_parser.h"

std::string get_arg_value(int argc, char* argv[], const std::string& arg_name) {
    for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == arg_name) {
            return std::string(argv[i + 1]);
        }
    }
    return "";
}

bool arg_exists(int argc, char* argv[], const std::string& arg_name) {
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == arg_name) {
            return true;
        }
    }
    return false;
}