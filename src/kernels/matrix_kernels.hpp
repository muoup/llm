#pragma once

#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>

#ifdef MATRIX_CHECKS
#define CHECK_ERRORS(step) CHECK_ERRORS(step)
#else
#define CHECK_ERRORS(step)
#endif

namespace kernel::matrix {

void test_print(kernel_stream_t stream = nullptr);
void check_errors(const char *step);

float get(const ::matrix &mat, const size_t row, const size_t col,
          kernel_stream_t stream = nullptr);
void set(::matrix &mat, const size_t row, const size_t col, const float value,
         kernel_stream_t stream = nullptr);

void load_into(::matrix &matrix, const float *buffer,
               kernel_stream_t stream = nullptr);
void store_from(const ::matrix &matrix, float *buffer,
                kernel_stream_t stream = nullptr);

void randomize(::matrix &matrix, const float min, const float max,
               kernel_stream_t stream = nullptr);

::matrix async_allocate(const size_t rows, const size_t cols,
                         kernel_stream_t stream = nullptr);

float *allocate_buffer(const size_t size, kernel_stream_t stream = nullptr);
void free_buffer(float *buffer);

float_device_ptr_t sum(const ::matrix &mat, kernel_stream_t stream = nullptr);
float_device_ptr_t sum_of_squares(const ::matrix &mat, kernel_stream_t stream = nullptr);
float_device_ptr_t abssum(const ::matrix &mat, kernel_stream_t stream = nullptr);
float_device_ptr_t max(const ::matrix &mat, kernel_stream_t stream = nullptr);
float_device_ptr_t min(const ::matrix &mat, kernel_stream_t stream = nullptr);
float_device_ptr_t absmax(const ::matrix &mat, kernel_stream_t stream = nullptr);
float_device_ptr_t variance(const ::matrix &mat, kernel_stream_t stream = nullptr);

::matrix clone(const ::const_matrix_view mat, kernel_stream_t stream = nullptr);

void set_all(::matrix &mat, float value, kernel_stream_t stream = nullptr);
void offset_all(::matrix &mat, float offset, kernel_stream_t stream = nullptr);

void transfer_row(::matrix &dest, const size_t dest_row, const ::matrix &src,
                  const size_t src_row, kernel_stream_t stream = nullptr);
void set_row_vector(::matrix &mat, const size_t mat_row, const ::matrix &vec,
                    const size_t vec_row = 0, kernel_stream_t stream = nullptr);
::matrix get_row_vector(const ::matrix &mat, const size_t row,
                        kernel_stream_t stream = nullptr);
void add_row_vector(::matrix &mat, const size_t row, const ::matrix &vec,
                    const size_t vec_row = 0, kernel_stream_t stream = nullptr);
void set_horizontal_slice(::matrix &mat, const size_t start_col,
                          const ::matrix &slice,
                          kernel_stream_t stream = nullptr);

void add(::matrix &mat, float value, kernel_stream_t stream = nullptr);
void add(::matrix &mat, const ::matrix &offset,
         kernel_stream_t stream = nullptr);
void add_scaled(::matrix &mat, const ::matrix &other, const float factor,
                kernel_stream_t stream = nullptr);
void scale(::matrix &mat, float factor, kernel_stream_t stream = nullptr);

void softmax(::matrix &mat, kernel_stream_t stream = nullptr);
::matrix backprop_softmax(const ::matrix &output, const ::matrix &gradient,
                          kernel_stream_t stream = nullptr);

void mask_upper_triangle(::matrix &mat, const float mask_value,
                         kernel_stream_t stream = nullptr);

::matrix dot_product(const ::matrix &a, const ::matrix &b,
                     kernel_stream_t stream = nullptr);
::matrix cross_multiplied(const ::const_matrix_view a,
                          const ::const_matrix_view b,
                          kernel_stream_t stream = nullptr);
::matrix t_cross_multiplied(const ::const_matrix_view a,
                            const ::const_matrix_view b,
                            kernel_stream_t stream = nullptr);
::matrix cross_t_multiplied(const ::const_matrix_view a,
                            const ::const_matrix_view b,
                            kernel_stream_t stream = nullptr);

void element_wise_multiply(::matrix &a, const ::matrix &b,
                           kernel_stream_t stream = nullptr);

bool is_equal(const ::matrix &a, const ::matrix &b, const float epsilon,
              kernel_stream_t stream = nullptr);

} // namespace kernel::matrix
