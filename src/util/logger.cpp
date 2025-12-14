#include "logger.hpp"

#include <cstdarg>
#include <cstdio>

static LogLevel current_log_level = LogLevel::INFO;

void logger::log(LogLevel level, const char* format, ...) {
    if (level < current_log_level) {
        return;
    }

    switch (level) {
        case LogLevel::DEBUG:
            std::printf("[DEBUG] ");
            break;
        case LogLevel::INFO:
            std::printf("[INFO] ");
            break;
        case LogLevel::WARNING:
            std::printf("[WARNING] ");
            break;
        case LogLevel::ERROR:
            std::printf("[ERROR] ");
            break;
    }

    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);

    std::printf("\n");
}

void logger::set_log_level(LogLevel level) {
    current_log_level = level;
}

LogLevel logger::get_log_level() {
    return current_log_level;
}