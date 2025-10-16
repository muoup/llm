#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>

#ifdef MATRIX_CHECKS
#include "../util/assert.h"
#include <sstream>
#endif

#ifdef MATRIX_CHECKS
#define MATRIX_ASSERT(condition, message, ...) \
    if (!(condition)) { \
        std::println("Matrix assertion failed: " message, __VA_ARGS__); \
        std::abort(); \
    }
#else
#define MATRIX_ASSERT(condition, message, ...) \
    (void)(condition); \
    (void)(message);
#endif

struct matrix {
    size_t rows, cols;
    std::unique_ptr<float[]> data;

    matrix(const size_t rows, const size_t cols)
        : rows(rows), cols(cols), data(std::make_unique<float[]>(cols * rows)) {
        std::memset(data.get(), 0, cols * rows * sizeof(float));
    }
    matrix(matrix &&) = default;
    matrix(const matrix &other) : matrix(other.rows, other.cols) {
        std::memcpy(data.get(), other.data.get(), cols * rows * sizeof(float));
    }

    matrix &operator=(matrix &&) = default;
    matrix &operator=(const matrix &other) {
        *this = matrix(other);
        return *this;
    }

    void randomize(float min = -1, float max = 1);

    [[nodiscard]] float get(const size_t row, const size_t col) const {
        verify_bounds(row, col);
        return data[col + row * cols];
    }

    void set(const size_t row, const size_t col, const float value) {
        verify_bounds(row, col);

        data[col + row * cols] = value;
    }

    void offset(const size_t row, const size_t col, const float offset) {
        verify_bounds(row, col);
        data[col + row * cols] += offset;
    }

    void set_row_vector(const size_t row, const matrix &row_vector) {
        MATRIX_ASSERT(
            this->cols == row_vector.cols,
            "Row vector must have the same number of columns as the matrix");

        for (size_t j = 0; j < row_vector.cols; ++j) {
            set(row, j, row_vector.get(0, j));
        }
    }

    void add_row_vector(const size_t row, const matrix &other) {
        MATRIX_ASSERT(
            this->cols == other.cols,
            "Row vector must have the same number of columns as the matrix");

        for (size_t i = 0; i < cols; ++i) {
            set(row, i, get(row, i) + other.get(0, i));
        }
    }

    matrix &softmax() {
        for (size_t i = 0; i < rows; ++i) {
            // Find the maximum value in the row
            float max_val = get(i, 0);
            for (size_t j = 1; j < cols; ++j) {
                if (get(i, j) > max_val) {
                    max_val = get(i, j);
                }
            }

            // Subtract the max value from each element in the row to prevent
            // overflow
            for (size_t j = 0; j < cols; ++j) {
                set(i, j, get(i, j) - max_val);
            }
        }

        this->map([](const float f) { return std::exp(f); });

        for (size_t i = 0; i < rows; ++i) {
            const auto row_sum
                = this->row_sum(i)
                  + 1e-8f; // Adding a small value to avoid division by zero

            for (size_t j = 0; j < cols; ++j) {
                set(i, j, get(i, j) / row_sum);
            }
        }

        return *this;
    }

    matrix &normalize() {
        this->scale(1.0f / this->absmax());

        return *this;
    }

    matrix cross_multiply(const matrix &other) const {
        MATRIX_ASSERT(
            this->cols == other.rows,
            "Matrix dimensions do not match for cross multiplication");

        matrix result{ this->rows, other.cols };
        const matrix other_t = other.transposed();

#ifndef MATRIX_CHECKS
#pragma omp parallel for
#endif
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < this->cols; ++k) {
                    sum += this->get(i, k) * other_t.get(j, k);
                }
                result.set(i, j, sum);
                verify_bounds(i, j);
            }
        }

        return result;
    }

    matrix &scale(const float factor) {
        this->map([factor](const float value) { return value * factor; });

        return *this;
    }

    matrix scaled(const float factor) const {
        matrix copy{ *this };
        copy.scale(factor);
        return copy;
    }

    matrix &offset(const matrix &offset) {
#ifdef MATRIX_CHECKS
        llm_assert(this->cols == offset.cols && this->rows == offset.rows,
                   "Matrix dimensions do not match for offset operation");
#endif

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                set(i, j, get(i, j) + offset.get(i, j));
            }
        }

        return *this;
    }

    matrix &map(const auto mapping) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const auto value = get(i, j);
                set(i, j, mapping(value));
            }
        }

        return *this;
    }

    matrix mapped(const auto mapping) const {
        matrix copy{ *this };
        copy.map(mapping);
        return copy;
    }

    template <typename ret> ret reduce(const auto reducer, ret acc = 0) const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                acc = reducer(acc, get(i, j));
            }
        }

        return acc;
    }

    matrix get_row_vector(const size_t row) const {
        matrix row_vector{ 1, cols };
        for (size_t j = 0; j < cols; ++j) {
            row_vector.set(0, j, get(row, j));
        }
        return row_vector;
    }

    float row_sum(const size_t row) const {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            sum += get(row, j);
        }
        return sum;
    }

    float col_sum(const size_t col) const {
        float sum = 0.0f;
        for (size_t i = 0; i < rows; ++i) {
            sum += get(i, col);
        }
        return sum;
    }

    float min() const;
    float max() const;
    float absmax() const;
    float variance() const;
    float stddev() const;

    matrix transposed() const {
        matrix transposed{ cols, rows };
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transposed.set(j, i, get(i, j));
            }
        }
        return transposed;
    }

    std::string header() const;
    std::string to_string(std::uint8_t precision = 4) const;

    std::span<float> to_span() const {
        return std::span(data.get(), cols * rows);
    }

    [[nodiscard]] size_t size() const { return cols * rows; }

  private:
    void verify_bounds(const size_t row, const size_t col) const;
};
