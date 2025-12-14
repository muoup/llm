#pragma once

// FIXME: Replace with a macro system to avoid forcing argument calculations when logging is disabled.

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

#ifdef MATRIX_CHECKS
#define LOG_DEBUG(...) LOG_DEBUG(__VA_ARGS__)
#else
#define LOG_DEBUG(...) do {} while(0)
#endif

namespace logger {

void log(LogLevel level, const char* format, ...);

void set_log_level(LogLevel level);
LogLevel get_log_level();

}