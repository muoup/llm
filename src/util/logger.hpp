#pragma once

// FIXME: Replace with a macro system to avoid forcing argument calculations when logging is disabled.

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

namespace logger {

void log(LogLevel level, const char* format, ...);

void set_log_level(LogLevel level);
LogLevel get_log_level();

}