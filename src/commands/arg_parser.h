#pragma once

#include <string>
#include <dataset/dataset_factory.h>

// A simple helper to find the value of an argument like "--key value"
std::string get_arg_value(int argc, char* argv[], const std::string& arg_name);

// Checks if a flag like "--verbose" exists
bool arg_exists(int argc, char* argv[], const std::string& arg_name);