//
// Created by user on 7/17/25.
//

#include "assert.h"
#include <cstdio>
#include <stdexcept>

void llm_assert(const bool condition, const char* message) {
    if (condition) {
        return;
    }

    std::printf("Assertion failed: %s\n", message);
    throw std::runtime_error(message);
}
