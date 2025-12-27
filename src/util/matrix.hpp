#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#ifdef MATRIX_CHECKS
#include <sstream>
#endif

#ifdef MATRIX_CHECKS
#define MATRIX_ASSERT(condition, message, ...)                           \
    if (!(condition)) {                                                  \
        std::printf("Matrix assertion failed: " message, ##__VA_ARGS__); \
        std::printf("\nAt: %s:%d\n", __FILE__, __LINE__);                \
        std::fflush(stdout);                                             \
        std::abort();                                                    \
    }
#else
#define MATRIX_ASSERT(condition, message, ...) \
    (void)(condition);                         \
    (void)(message);
#endif

struct const_matrix_view;
struct matrix_view;

constexpr size_t calculate_stride(const size_t rows, const size_t cols) {
    // The Stride is Equal to the Least Multiple of (256 / sizeof(float)) Equal
    // to or Greater Than i
    constexpr size_t alignment = 256 / sizeof(float);

    return ((cols + alignment - 1) / alignment) * alignment;
}

struct matrix {
    std::uint64_t rows, cols, stride;

    float* data;

    constexpr static auto MATRIX_ELEMENT_ALIGNMENT = 256;

    matrix() : rows(0), cols(0), stride(0), data(nullptr) {}
    matrix(const size_t rows, const size_t cols);
    matrix(matrix&&);
    matrix(const matrix& other) = delete;

    ~matrix();

    matrix& operator=(matrix&&);
    matrix& operator=(const matrix& other) = delete;
    matrix& operator=(matrix&) = delete;

    size_t buffer_size() const;
    void randomize(float min = -1, float max = 1);
    void leaky_kaiming_randomize();
    void xavier_randomize();
    void zero();

    float* data_ptr() { return data; }
    const float* data_ptr() const { return data; }

    matrix& scale(const float factor);
    matrix& add(const matrix& offset);
    matrix& add(float f);
    matrix& add_scaled(const matrix& other, const float factor);
    matrix& set_all(const float value);

    [[nodiscard]] float get(const size_t row, const size_t col) const;
    void set(const size_t row, const size_t col, const float value);
    void offset(const size_t row, const size_t col, const float offset) {
        verify_bounds(row, col);
        const auto current_value = get(row, col);
        set(row, col, current_value + offset);
    }

    const_matrix_view get_row_vector(const size_t row) const;
    void set_row_vector(const size_t row, const matrix& row_vector);
    void add_row_vector(const size_t row, const matrix& other);
    void set_horizontal_slice(const size_t col_start, const matrix& slice);
    const_matrix_view get_horizontal_slice(const size_t col_start,
                                const size_t slice_cols) const;

    matrix& softmax();

    matrix& mask_upper_triangular(float mask_value
                                  = -std::numeric_limits<float>::infinity());
    matrix& normalize() {
        this->scale(1.0f / this->absmax());

        return *this;
    }

    float dot_product(const matrix& other) const;

    matrix cross_multiplied(const matrix& other) const;
    matrix t_cross_multiplied(const matrix& other) const;
    matrix cross_t_multiplied(const matrix& other) const;

    matrix& element_wise_multiply(const matrix& other);

    matrix clone() const;

    matrix scaled(const float factor) const {
        auto copy = this->clone();
        copy.scale(factor);
        return copy;
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

    float sum() const;
    float min() const;
    float max() const;
    float absmax() const;
    float abssum() const;
    float variance() const;
    float stddev() const;
    float norm() const;

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

    void save(std::ostream& out) const;
    static matrix load(std::istream& in);

    bool equals(const matrix& other, const float epsilon = 1e-6f) const;

    void print_bounds() const {
        std::cout << "Matrix bounds: rows=" << rows << ", cols=" << cols
                  << ", stride=" << this->stride << "\n";
    }

    void print_contents() const;

    [[nodiscard]] size_t size() const { return cols * rows; }

   private:
    void verify_bounds(const size_t row, const size_t col) const;
};

struct matrix_view {
    size_t rows, cols, stride;
    float* data;

    matrix_view() : rows(0), cols(0), stride(0), data(nullptr) {}
    matrix_view(matrix& other)
        : rows(other.rows),
          cols(other.cols),
          stride(other.stride),
          data(other.data) {}
    matrix_view(const size_t rows,
                const size_t cols,
                const size_t stride,
                float* data)
        : rows(rows), cols(cols), stride(stride), data(data) {}

    matrix to_matrix() const;
};

struct const_matrix_view {
    size_t rows, cols, stride;
    const float* data;

    const_matrix_view() : rows(0), cols(0), stride(0), data(nullptr) {}
    const_matrix_view(const matrix_view& other)
        : rows(other.rows),
          cols(other.cols),
          stride(other.stride),
          data(other.data) {}
    const_matrix_view(matrix& other)
        : rows(other.rows),
          cols(other.cols),
          stride(other.stride),
          data(other.data) {}
    const_matrix_view(const matrix& other)
        : rows(other.rows),
          cols(other.cols),
          stride(other.stride),
          data(other.data) {}
    const_matrix_view(const size_t rows,
                      const size_t cols,
                      const size_t stride,
                      const float* data)
        : rows(rows), cols(cols), stride(stride), data(data) {}

    matrix to_matrix() const;
};
