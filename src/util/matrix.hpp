#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

#ifdef MATRIX_CHECKS
#include <sstream>
#include "../util/assert.hpp"
#endif

#ifdef MATRIX_CHECKS
#define MATRIX_ASSERT(condition, message, ...)                            \
    if (!(condition)) {                                                   \
        std::printf("Matrix assertion failed: " message, ##__VA_ARGS__);  \
        std::fflush(stdout);                                              \
        std::abort();                                                     \
    }
#else
#define MATRIX_ASSERT(condition, message, ...) \
    (void)(condition);                         \
    (void)(message);
#endif

struct aligned_deleter {
    void operator()(float* ptr) const { std::free(ptr); }
};

struct matrix {
    size_t rows, cols;
    size_t row_width;

    std::unique_ptr<float[], aligned_deleter> data;

    constexpr static auto MATRIX_ELEMENT_ALIGNMENT = 8;

    matrix() : rows(0), cols(0), row_width(0), data(nullptr) {}
    matrix(const size_t rows, const size_t cols);

    matrix(matrix&&) = default;
    matrix(const matrix& other) = delete;

    matrix& operator=(matrix&&) = default;
    matrix& operator=(const matrix& other) = delete;

    size_t buffer_size() const;
    void randomize(float min = -1, float max = 1);

    float* data_ptr() { return data.get(); }
    const float* data_ptr() const { return data.get(); }

    [[nodiscard]] float get(const size_t row, const size_t col) const {
        verify_bounds(row, col);
        return data[col + row * row_width];
    }

    void set(const size_t row, const size_t col, const float value) {
        verify_bounds(row, col);

        data[col + row * row_width] = value;
    }

    void offset(const size_t row, const size_t col, const float offset) {
        verify_bounds(row, col);
        data[col + row * row_width] += offset;
    }

    void set_row_vector(const size_t row, const matrix& row_vector) {
        MATRIX_ASSERT(
            this->cols == row_vector.cols,
            "Row vector must have the same number of columns as the matrix");

        for (size_t j = 0; j < row_vector.cols; ++j) {
            set(row, j, row_vector.get(0, j));
        }
    }

    void add_row_vector(const size_t row, const matrix& other) {
        MATRIX_ASSERT(
            this->cols == other.cols,
            "Row vector must have the same number of columns as the matrix");

        for (size_t i = 0; i < cols; ++i) {
            set(row, i, get(row, i) + other.get(0, i));
        }
    }

    void set_horizontal_slice(const size_t col_start, const matrix& slice) {
        MATRIX_ASSERT(this->rows == slice.rows,
                      "Slice must have the same number of rows as the matrix");
        MATRIX_ASSERT(col_start + slice.cols <= this->cols,
                      "Slice exceeds matrix row bounds");

        for (size_t i = 0; i < slice.rows; ++i) {
            for (size_t j = 0; j < slice.cols; ++j) {
                set(i, j + col_start, slice.get(i, j));
            }
        }
    }

    matrix get_horizontal_slice(const size_t col_start,
                                const size_t slice_cols) const {
        MATRIX_ASSERT(col_start + slice_cols <= this->cols,
                      "Slice exceeds matrix row bounds");

        matrix slice{ this->rows, slice_cols };
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < slice_cols; ++j) {
                slice.set(i, j, this->get(i, j + col_start));
            }
        }
        return slice;
    }

    matrix& softmax();
    matrix& mask_upper_triangular(float mask_value
                                  = -std::numeric_limits<float>::infinity());

    matrix& normalize() {
        this->scale(1.0f / this->absmax());

        return *this;
    }

    float dot_product(const matrix& other) const;

    void cross_multiply_into(const matrix& other, matrix& out) const;
    matrix cross_multiplied(const matrix& other) const;
    matrix t_cross_multiplied(const matrix& other) const;
    matrix cross_t_multiplied(const matrix& other) const;

    matrix backprop_softmax(const matrix& gradient) const;

    matrix clone() const {
        matrix copy{ this->rows, this->cols };
        std::memcpy(copy.data_ptr(), this->data_ptr(), this->buffer_size());
        return copy;
    }

    matrix& scale(const float factor) {
        this->map([factor](const float value) { return value * factor; });

        return *this;
    }

    matrix scaled(const float factor) const {
        auto copy = this->clone();
        copy.scale(factor);
        return copy;
    }

    matrix& add(const matrix& offset) {
        MATRIX_ASSERT(this->cols == offset.cols && this->rows == offset.rows,
                   "Matrix dimensions do not match for offset operation");

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                set(i, j, get(i, j) + offset.get(i, j));
            }
        }

        return *this;
    }

    matrix& add_scaled(const matrix& other, const float factor) {
        MATRIX_ASSERT(
            this->cols == other.cols && this->rows == other.rows,
            "Matrix dimensions do not match for scaled addition operation");

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                set(i, j, get(i, j) + other.get(i, j) * factor);
            }
        }

        return *this;
    }

    matrix& map(const auto mapping) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const auto value = get(i, j);
                set(i, j, mapping(value));
            }
        }

        return *this;
    }

    matrix mapped(const auto mapping) const {
        auto copy = this->clone();
        copy.map(mapping);
        return copy;
    }

    matrix& element_wise_multiply(const matrix& other) {
        MATRIX_ASSERT(
            this->cols == other.cols && this->rows == other.rows,
            "Matrix dimensions do not match for element-wise multiplication");
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                set(i, j, get(i, j) * other.get(i, j));
            }
        }

        return *this;
    }

    template <typename ret>
    ret reduce(const auto reducer, ret acc = 0) const {
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

    template <typename... Args>
    static std::vector<matrix> construct_vec(Args&... args) {
        std::vector<matrix> vec;

        (vec.emplace_back(std::move(args)), ...);

        return vec;
    }

    std::span<float> to_span() {
        return std::span(this->data_ptr(), buffer_size() / sizeof(float));
    }

    std::span<const float> to_span() const {
        return std::span(this->data_ptr(), buffer_size() / sizeof(float));
    }

    void save(std::ostream& out) const;
    static matrix load(std::istream& in);

    bool equals(const matrix& other, const float epsilon = 1e-6f) const;

    void print_bounds() const {
        std::cout << "Matrix bounds: rows=" << rows << ", cols=" << cols
                  << "\n";
    }

    [[nodiscard]] size_t size() const { return cols * rows; }

   private:
    void verify_bounds(const size_t row, const size_t col) const;
};
