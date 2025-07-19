//
// Created by user on 7/17/25.
//

#include "matrix.h"

#include <cmath>
#include <iomanip>

void matrix::randomize(const float min, const float max) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            set(i, j, min + static_cast<float>(std::rand()) / (RAND_MAX / (max - min)));
        }
    }
}


float matrix::max() const {
    return this->reduce<float>([](const float a, const float b) { return std::max(a, b); },
                    std::numeric_limits<float>::lowest());
}

float matrix::min() const {
    return this->reduce<float>([](const float a, const float b) { return std::min(a, b); },
                    std::numeric_limits<float>::max());
}

float matrix::variance() const {
    const float sum = this->reduce<float>([](const float acc, const float value) { return acc + value; }, 0.0f);
    const float sum_sq = this->reduce<float>([](const float acc, const float value) { return acc + value * value; }, 0.0f);

    return (sum_sq / size()) - (sum / size()) * (sum / size());
}

float matrix::stddev() const {
    return std::sqrt(variance());
}

std::string matrix::header() const {
    std::stringstream ss;
    ss << "Matrix(" << rows << "x" << cols << "):\n";

    return ss.str();
}

std::string matrix::to_string(std::uint8_t precision) const {
    std::stringstream ss;

    ss << header();
    ss << std::fixed << std::setprecision(precision);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ss << get(i, j) << " ";
        }
        ss << "\n";
    }

    return ss.str();
}