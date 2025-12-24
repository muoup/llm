#pragma once

#include <util/matrix.hpp>

#ifdef MATRIX_CHECKS
#define CHECK_ERRORS(step) check_errors(step)
#else
#define CHECK_ERRORS(step)
#endif

namespace kernel::matrix {

typedef void* matmul_stream_t;
    
void test_print();
void check_errors(const char* step);
    
float get(const ::matrix& mat, const size_t row, const size_t col);
void set(::matrix& mat, const size_t row, const size_t col, const float value);

void load_into(::matrix& matrix, const float* buffer);
void store_from(const ::matrix& matrix, float* buffer);

void randomize(::matrix& matrix, const float min, const float max);

float* allocate_buffer(const size_t size);
void free_buffer(float* buffer);

float sum(const ::matrix& mat);
float sum_of_squares(const ::matrix& mat);
float abssum(const ::matrix& mat);
float max(const ::matrix& mat);
float min(const ::matrix& mat);
float absmax(const ::matrix& mat);
float variance(const ::matrix& mat);

::matrix clone(const ::const_matrix_view mat);

void set_all(::matrix& mat, float value);
void offset_all(::matrix& mat, float offset);

void transfer_row(::matrix& dest, const size_t dest_row,
                  const ::matrix& src, const size_t src_row);
void set_row_vector(::matrix& mat, const size_t mat_row, const ::matrix& vec, const size_t vec_row = 0);
::matrix get_row_vector(const ::matrix& mat, const size_t row);
void add_row_vector(::matrix& mat, const size_t row, const ::matrix& vec, const size_t vec_row = 0);
void set_horizontal_slice(::matrix& mat,
                          const size_t start_col,
                          const ::matrix& slice);

void add(::matrix& mat, float value);
void add(::matrix& mat, const ::matrix& offset);
void add_scaled(::matrix& mat, const ::matrix& other, const float factor);
void scale(::matrix& mat, float factor);

void softmax(::matrix& mat);
::matrix backprop_softmax(const ::matrix& output, const ::matrix& gradient);

void mask_upper_triangle(::matrix& mat, const float mask_value);

::matrix dot_product(const ::matrix& a, const ::matrix& b);
::matrix cross_multiplied(const ::const_matrix_view a, const ::const_matrix_view b, matmul_stream_t stream = nullptr);
::matrix t_cross_multiplied(const ::const_matrix_view a, const ::const_matrix_view b, matmul_stream_t stream = nullptr);
::matrix cross_t_multiplied(const ::const_matrix_view a, const ::const_matrix_view b, matmul_stream_t stream = nullptr);

void element_wise_multiply(::matrix& a, const ::matrix& b);

bool is_equal(const ::matrix& a, const ::matrix& b, const float epsilon);

matmul_stream_t create_matmul_stream();
void destroy_matmul_stream(matmul_stream_t stream);

}  // namespace kernel::matrix
