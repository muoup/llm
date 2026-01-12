#include "device.cuh"

#include <cuda_fp16.hpp>
#include <cuda_bf16.hpp>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <kernels/matrix/host.hpp>
#include <kernels/scheduling.cuh>
#include <util/matrix.hpp>

namespace kernel::matrix::device {

// ===== Device Helpers =====

__device__ float* get_addr_float(void* data, size_t stride, size_t row, size_t col) {
    return &((float*)data)[row * stride + col];
}

__device__ half* get_addr_half(void* data, size_t stride, size_t row, size_t col) {
    return &((half*)data)[row * stride + col];
}

__device__ __nv_bfloat16* get_addr_bf16(void* data, size_t stride, size_t row, size_t col) {
    return &((__nv_bfloat16*)data)[row * stride + col];
}

__device__ const float* get_addr_float(const void* data, size_t stride, size_t row, size_t col) {
    return &((float*)data)[row * stride + col];
}

__device__ const half* get_addr_half(const void* data, size_t stride, size_t row, size_t col) {
    return &((half*)data)[row * stride + col];
}

__device__ const __nv_bfloat16* get_addr_bf16(const void* data, size_t stride, size_t row, size_t col) {
    return &((__nv_bfloat16*)data)[row * stride + col];
}

__device__ void* get_addr(matrix_view& view, size_t row, size_t col) {
    switch (view.type) {
        case DataType::Float:
            return get_addr_float(view.data, view.stride, row, col);
        case DataType::Half:
            return get_addr_half(view.data, view.stride, row, col);
        case DataType::BFloat16:
            return get_addr_bf16(view.data, view.stride, row, col);
    }
    return nullptr;
}

__device__ const void* get_addr(const const_matrix_view& view, size_t row, size_t col) {
    switch (view.type) {
        case DataType::Float:
            return get_addr_float(view.data, view.stride, row, col);
        case DataType::Half:
            return get_addr_half(view.data, view.stride, row, col);
        case DataType::BFloat16:
            return get_addr_bf16(view.data, view.stride, row, col);
    }
    return nullptr;
}

__device__ float get_float(const void* data, size_t stride, size_t row, size_t col) {
    return *get_addr_float(const_cast<void*>(data), stride, row, col);
}

__device__ half get_half(const void* data, size_t stride, size_t row, size_t col) {
    return ((const half*)data)[row * stride + col];
}

__device__ __nv_bfloat16 get_bf16(const void* data, size_t stride, size_t row, size_t col) {
    return ((const __nv_bfloat16*)data)[row * stride + col];
}

__device__ void set_float(void* data, size_t stride, size_t row, size_t col, float value) {
    *get_addr_float(data, stride, row, col) = value;
}

__device__ void set_half(void* data, size_t stride, size_t row, size_t col, half value) {
    *get_addr_half(data, stride, row, col) = value;
}

__device__ void set_bf16(void* data, size_t stride, size_t row, size_t col, __nv_bfloat16 value) {
    *get_addr_bf16(data, stride, row, col) = value;
}

__device__ void offset_elem_float(void* data, size_t stride, size_t row, size_t col, float value) {
    *get_addr_float(data, stride, row, col) += value;
}

__device__ void offset_elem_half(void* data, size_t stride, size_t row, size_t col, half value) {
    *get_addr_half(data, stride, row, col) += value;
}

__device__ void offset_elem_bf16(void* data, size_t stride, size_t row, size_t col, __nv_bfloat16 value) {
    *get_addr_bf16(data, stride, row, col) += value;
}

__device__ void offset_elem(const matrix_view& view, size_t row, size_t col, float value) {
    switch (view.type) {
        case DataType::Float:
            offset_elem_float(view.data, view.stride, row, col, value);
            break;
        case DataType::Half:
            offset_elem_half(view.data, view.stride, row, col, __float2half(value));
            break;
        case DataType::BFloat16:
            offset_elem_bf16(view.data, view.stride, row, col, __float2bfloat16(value));
            break;
    }
}

__device__ void offset_elem_atomic_float(void* data, size_t stride, size_t row, size_t col, float value) {
    float* addr = get_addr_float(data, stride, row, col);
    atomicAdd(addr, value);
}

__device__ void offset_elem_atomic_half(void* data, size_t stride, size_t row, size_t col, half value) {
    half* addr = get_addr_half(data, stride, row, col);
    atomicAdd(addr, value);
}

__device__ void offset_elem_atomic_bf16(void* data, size_t stride, size_t row, size_t col, __nv_bfloat16 value) {
    __nv_bfloat16* addr = get_addr_bf16(data, stride, row, col);
    atomicAdd(addr, value);
}

__device__ void offset_elem_atomic(const matrix_view& view, size_t row, size_t col, float value) {
    switch (view.type) {
        case DataType::Float:
            offset_elem_atomic_float(view.data, view.stride, row, col, value);
            break;
        case DataType::Half:
            offset_elem_atomic_half(view.data, view.stride, row, col, __float2half(value));
            break;
        case DataType::BFloat16:
            offset_elem_atomic_bf16(view.data, view.stride, row, col, __float2bfloat16(value));
            break;
    }
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset, 32);
    return val;
}

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset, 32));
    return val;
}

__device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32.0f) ? shared[lane] : 0;
    if (wid == 0)
        val = warp_reduce_sum(val);
    __syncthreads();
    return val;
}

__device__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32.0f) ? shared[lane] : -CUDART_INF_F;
    if (wid == 0)
        val = warp_reduce_max(val);
    __syncthreads();
    return val;
}

// ===== Device Kernels =====

// Copy kernels
__global__ void copy_matrix_kernel_float(const matrix_view dest, const const_matrix_view src) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < src.rows && col < src.cols) {
        float val = get_float(src.data, src.stride, row, col);
        set_float(dest.data, dest.stride, row, col, val);
    }
}

__global__ void copy_matrix_kernel_half(matrix_view dest, const const_matrix_view src) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < src.rows && col < src.cols) {
        half val = get_half(src.data, src.stride, row, col);
        set_half(dest.data, dest.stride, row, col, val);
    }
}

__global__ void copy_matrix_kernel_bf16(matrix_view dest, const const_matrix_view src) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < src.rows && col < src.cols) {
        __nv_bfloat16 val = get_bf16(src.data, src.stride, row, col);
        set_bf16(dest.data, dest.stride, row, col, val);
    }
}

// Set all kernels
__global__ void set_all_kernel_float(matrix_view data, float value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        set_float(data.data, data.stride, row, col, value);
    }
}

__global__ void set_all_kernel_half(matrix_view data, float value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        set_half(data.data, data.stride, row, col, __float2half(value));
    }
}

__global__ void set_all_kernel_bf16(matrix_view data, float value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        set_bf16(data.data, data.stride, row, col, __float2bfloat16(value));
    }
}

// Scale kernels
__global__ void scale_kernel_float(matrix_view data, float factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        float val = get_float(data.data, data.stride, row, col);
        set_float(data.data, data.stride, row, col, val * factor);
    }
}

__global__ void scale_kernel_half(matrix_view data, float factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        half val = get_half(data.data, data.stride, row, col);
        set_half(data.data, data.stride, row, col, __float2half(__half2float(val) * factor));
    }
}

__global__ void scale_kernel_bf16(matrix_view data, float factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        __nv_bfloat16 val = get_bf16(data.data, data.stride, row, col);
        set_bf16(data.data, data.stride, row, col, __float2bfloat16(__bfloat162float(val) * factor));
    }
}

// Add matrix kernels
__global__ void add_matrix_kernel_float(matrix_view data, const const_matrix_view other) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        float val = get_float(other.data, other.stride, row, col);
        offset_elem_float(data.data, data.stride, row, col, val);
    }
}

__global__ void add_matrix_kernel_half(matrix_view data, const const_matrix_view other) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        half val = get_half(other.data, other.stride, row, col);
        offset_elem_half(data.data, data.stride, row, col, val);
    }
}

__global__ void add_matrix_kernel_bf16(matrix_view data, const const_matrix_view other) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        __nv_bfloat16 val = get_bf16(other.data, other.stride, row, col);
        offset_elem_bf16(data.data, data.stride, row, col, val);
    }
}

// Atomic add matrix kernels
__global__ void add_matrix_atomic_kernel_float(matrix_view data, const const_matrix_view other) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        float val = get_float(other.data, other.stride, row, col);
        offset_elem_atomic_float(data.data, data.stride, row, col, val);
    }
}

__global__ void add_matrix_atomic_kernel_half(matrix_view data, const const_matrix_view other) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        half val = get_half(other.data, other.stride, row, col);
        offset_elem_atomic_half(data.data, data.stride, row, col, val);
    }
}

__global__ void add_matrix_atomic_kernel_bf16(matrix_view data, const const_matrix_view other) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        __nv_bfloat16 val = get_bf16(other.data, other.stride, row, col);
        offset_elem_atomic_bf16(data.data, data.stride, row, col, val);
    }
}

// Add scaled kernels
__global__ void add_scaled_kernel_float(matrix_view data, const const_matrix_view other, float factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= data.rows || col >= data.cols)
        return;
    float value = get_float(other.data, other.stride, row, col);
    offset_elem_float(data.data, data.stride, row, col, value * factor);
}

__global__ void add_scaled_kernel_half(matrix_view data, const const_matrix_view other, float factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= data.rows || col >= data.cols)
        return;
    half value = get_half(other.data, other.stride, row, col);
    offset_elem_half(data.data, data.stride, row, col, __float2half(__half2float(value) * factor));
}

__global__ void add_scaled_kernel_bf16(matrix_view data, const const_matrix_view other, float factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= data.rows || col >= data.cols)
        return;
    __nv_bfloat16 value = get_bf16(other.data, other.stride, row, col);
    offset_elem_bf16(data.data, data.stride, row, col, __float2bfloat16(__bfloat162float(value) * factor));
}

// Add value kernels
__global__ void add_value_kernel_float(matrix_view data, float value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        offset_elem_float(data.data, data.stride, row, col, value);
    }
}

__global__ void add_value_kernel_half(matrix_view data, float value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        offset_elem_half(data.data, data.stride, row, col, __float2half(value));
    }
}

__global__ void add_value_kernel_bf16(matrix_view data, float value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        offset_elem_bf16(data.data, data.stride, row, col, __float2bfloat16(value));
    }
}

// Transfer row kernels
__global__ void transfer_row_kernel_float(matrix_view dest, size_t dest_row,
                                   const const_matrix_view src, size_t src_row) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < src.cols && col < dest.cols) {
        float val = get_float(src.data, src.stride, src_row, col);
        set_float(dest.data, dest.stride, dest_row, col, val);
    }
}

__global__ void transfer_row_kernel_half(matrix_view dest, size_t dest_row,
                                    const const_matrix_view src, size_t src_row) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < src.cols && col < dest.cols) {
        half val = get_half(src.data, src.stride, src_row, col);
        set_half(dest.data, dest.stride, dest_row, col, val);
    }
}

__global__ void transfer_row_kernel_bf16(matrix_view dest, size_t dest_row,
                                      const const_matrix_view src, size_t src_row) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < src.cols && col < dest.cols) {
        __nv_bfloat16 val = get_bf16(src.data, src.stride, src_row, col);
        set_bf16(dest.data, dest.stride, dest_row, col, val);
    }
}

// Add row vector kernels
__global__ void add_row_vector_kernel_float(matrix_view data, size_t data_row,
                                        const const_matrix_view row_vec, size_t offset_row, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < data.cols) {
        float val = get_float(row_vec.data, row_vec.stride, offset_row, col);
        offset_elem_float(data.data, data.stride, data_row, col, val * scale);
    }
}

__global__ void add_row_vector_kernel_half(matrix_view data, size_t data_row,
                                       const const_matrix_view row_vec, size_t offset_row, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < data.cols) {
        half val = get_half(row_vec.data, row_vec.stride, offset_row, col);
        offset_elem_half(data.data, data.stride, data_row, col, __float2half(__half2float(val) * scale));
    }
}

__global__ void add_row_vector_kernel_bf16(matrix_view data, size_t data_row,
                                        const const_matrix_view row_vec, size_t offset_row, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < data.cols) {
        __nv_bfloat16 val = get_bf16(row_vec.data, row_vec.stride, offset_row, col);
        offset_elem_bf16(data.data, data.stride, data_row, col, __float2bfloat16(__bfloat162float(val) * scale));
    }
}

// Atomic add row vector kernels
__global__ void atomic_add_row_vector_kernel_float(matrix_view data, size_t data_row,
                                               const const_matrix_view row_vec, size_t offset_row, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < data.cols) {
        float val = get_float(row_vec.data, row_vec.stride, offset_row, col);
        offset_elem_atomic_float(data.data, data.stride, data_row, col, val * scale);
    }
}

__global__ void atomic_add_row_vector_kernel_half(matrix_view data, size_t data_row,
                                                const const_matrix_view row_vec, size_t offset_row, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < data.cols) {
        half val = get_half(row_vec.data, row_vec.stride, offset_row, col);
        offset_elem_atomic_half(data.data, data.stride, data_row, col, __float2half(__half2float(val) * scale));
    }
}

__global__ void atomic_add_row_vector_kernel_bf16(matrix_view data, size_t data_row,
                                                 const const_matrix_view row_vec, size_t offset_row, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < data.cols) {
        __nv_bfloat16 val = get_bf16(row_vec.data, row_vec.stride, offset_row, col);
        offset_elem_atomic_bf16(data.data, data.stride, data_row, col, __float2bfloat16(__bfloat162float(val) * scale));
    }
}

// Set horizontal slice kernels
__global__ void set_horizontal_slice_kernel_float(matrix_view data, size_t start_col,
                                             const const_matrix_view slice) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < slice.rows && col < slice.cols) {
        float val = get_float(slice.data, slice.stride, row, col);
        set_float(data.data, data.stride, row, start_col + col, val);
    }
}

__global__ void set_horizontal_slice_kernel_half(matrix_view data, size_t start_col,
                                              const const_matrix_view slice) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < slice.rows && col < slice.cols) {
        half val = get_half(slice.data, slice.stride, row, col);
        set_half(data.data, data.stride, row, start_col + col, val);
    }
}

__global__ void set_horizontal_slice_kernel_bf16(matrix_view data, size_t start_col,
                                               const const_matrix_view slice) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < slice.rows && col < slice.cols) {
        __nv_bfloat16 val = get_bf16(slice.data, slice.stride, row, col);
        set_bf16(data.data, data.stride, row, start_col + col, val);
    }
}

// Mask upper triangle kernels
__global__ void mask_upper_triangle_kernel_float(matrix_view data, float mask_value) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < data.rows && col < data.cols && col > row) {
        set_float(data.data, data.stride, row, col, mask_value);
    }
}

__global__ void mask_upper_triangle_kernel_half(matrix_view data, float mask_value) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < data.rows && col < data.cols && col > row) {
        set_half(data.data, data.stride, row, col, __float2half(mask_value));
    }
}

__global__ void mask_upper_triangle_kernel_bf16(matrix_view data, float mask_value) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < data.rows && col < data.cols && col > row) {
        set_bf16(data.data, data.stride, row, col, __float2bfloat16(mask_value));
    }
}

// Element-wise multiply kernels
__global__ void element_wise_multiply_kernel_float(matrix_view a_data, const const_matrix_view b_data) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < a_data.rows && col < a_data.cols) {
        float val_a = get_float(a_data.data, a_data.stride, row, col);
        float val_b = get_float(b_data.data, b_data.stride, row, col);
        set_float(a_data.data, a_data.stride, row, col, val_a * val_b);
    }
}

__global__ void element_wise_multiply_kernel_half(matrix_view a_data, const const_matrix_view b_data) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < a_data.rows && col < a_data.cols) {
        half val_a = get_half(a_data.data, a_data.stride, row, col);
        half val_b = get_half(b_data.data, b_data.stride, row, col);
        set_half(a_data.data, a_data.stride, row, col, __float2half(__half2float(val_a) * __half2float(val_b)));
    }
}

__global__ void element_wise_multiply_kernel_bf16(matrix_view a_data, const const_matrix_view b_data) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < a_data.rows && col < a_data.cols) {
        __nv_bfloat16 val_a = get_bf16(a_data.data, a_data.stride, row, col);
        __nv_bfloat16 val_b = get_bf16(b_data.data, b_data.stride, row, col);
        set_bf16(a_data.data, a_data.stride, row, col, __float2bfloat16(__bfloat162float(val_a) * __bfloat162float(val_b)));
    }
}

// Type conversion kernels
__global__ void convert_float_to_half(float* float_data, uint16_t* half_data, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        half_data[idx] = __half_as_ushort(__float2half(float_data[idx]));
    }
}

__global__ void convert_float_to_bf16(float* float_data, uint16_t* bf16_data, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        bf16_data[idx] = __bfloat16_as_ushort(__float2bfloat16(float_data[idx]));
    }
}

// Softmax kernels
__global__ void softmax_kernel_float(matrix_view data) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float local_max = -CUDART_INF_F;
    for (int col = tid; col < data.cols; col += blockDim.x) {
        float val = get_float(data.data, data.stride, row, col);
        local_max = fmaxf(local_max, val);
    }
    float row_max = block_reduce_max(local_max);
    __shared__ float shared_max;
    if (tid == 0) {
        shared_max = row_max;
    }
    __syncthreads();
    float local_sum = 0.0f;
    if (shared_max > -CUDART_INF_F) {
        for (int col = tid; col < data.cols; col += blockDim.x) {
            float val = get_float(data.data, data.stride, row, col);
            float exp_val = expf(val - shared_max);
            set_float(data.data, data.stride, row, col, exp_val);
            local_sum += exp_val;
        }
    } else {
        for (int col = tid; col < data.cols; col += blockDim.x) {
            set_float(data.data, data.stride, row, col, 0.0f);
        }
    }
    __syncthreads();
    float row_sum = block_reduce_sum(local_sum);
    __shared__ float shared_denom;
    if (tid == 0) {
        shared_denom = row_sum;
    }
    __syncthreads();
    if (shared_denom > 0.0f) {
        for (int col = tid; col < data.cols; col += blockDim.x) {
            float val = get_float(data.data, data.stride, row, col);
            set_float(data.data, data.stride, row, col, val / shared_denom);
        }
    }
}

__global__ void softmax_kernel_half(matrix_view data) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float local_max = -CUDART_INF_F;
    for (int col = tid; col < data.cols; col += blockDim.x) {
        half val = get_half(data.data, data.stride, row, col);
        local_max = fmaxf(local_max, __half2float(val));
    }
    float row_max = block_reduce_max(local_max);
    __shared__ float shared_max;
    if (tid == 0) {
        shared_max = row_max;
    }
    __syncthreads();
    float local_sum = 0.0f;
    if (shared_max > -CUDART_INF_F) {
        for (int col = tid; col < data.cols; col += blockDim.x) {
            half val = get_half(data.data, data.stride, row, col);
            float exp_val = expf(__half2float(val) - shared_max);
            set_half(data.data, data.stride, row, col, __float2half(exp_val));
            local_sum += exp_val;
        }
    } else {
        for (int col = tid; col < data.cols; col += blockDim.x) {
            set_half(data.data, data.stride, row, col, __float2half(0.0f));
        }
    }
    __syncthreads();
    float row_sum = block_reduce_sum(local_sum);
    __shared__ float shared_denom;
    if (tid == 0) {
        shared_denom = row_sum;
    }
    __syncthreads();
    if (shared_denom > 0.0f) {
        for (int col = tid; col < data.cols; col += blockDim.x) {
            half val = get_half(data.data, data.stride, row, col);
            set_half(data.data, data.stride, row, col, __float2half(__half2float(val) / shared_denom));
        }
    }
}

__global__ void softmax_kernel_bf16(matrix_view data) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float local_max = -CUDART_INF_F;
    for (int col = tid; col < data.cols; col += blockDim.x) {
        __nv_bfloat16 val = get_bf16(data.data, data.stride, row, col);
        local_max = fmaxf(local_max, __bfloat162float(val));
    }
    float row_max = block_reduce_max(local_max);
    __shared__ float shared_max;
    if (tid == 0) {
        shared_max = row_max;
    }
    __syncthreads();
    float local_sum = 0.0f;
    if (shared_max > -CUDART_INF_F) {
        for (int col = tid; col < data.cols; col += blockDim.x) {
            __nv_bfloat16 val = get_bf16(data.data, data.stride, row, col);
            float exp_val = expf(__bfloat162float(val) - shared_max);
            set_bf16(data.data, data.stride, row, col, __float2bfloat16(exp_val));
            local_sum += exp_val;
        }
    } else {
        for (int col = tid; col < data.cols; col += blockDim.x) {
            set_bf16(data.data, data.stride, row, col, __float2bfloat16(0.0f));
        }
    }
    __syncthreads();
    float row_sum = block_reduce_sum(local_sum);
    __shared__ float shared_denom;
    if (tid == 0) {
        shared_denom = row_sum;
    }
    __syncthreads();
    if (shared_denom > 0.0f) {
        for (int col = tid; col < data.cols; col += blockDim.x) {
            __nv_bfloat16 val = get_bf16(data.data, data.stride, row, col);
            set_bf16(data.data, data.stride, row, col, __float2bfloat16(__bfloat162float(val) / shared_denom));
        }
    }
}

// Backprop softmax kernels
__global__ void backprop_softmax_kernel_float(const const_matrix_view softmax_output,
                                          const const_matrix_view output_gradient,
                                          matrix_view softmax_gradient) {
    int row = blockIdx.x;
    float local_dot = 0.0f;
    for (int col = threadIdx.x; col < softmax_output.cols; col += blockDim.x) {
        float s_j = get_float(softmax_output.data, softmax_output.stride, row, col);
        float g_j = get_float(output_gradient.data, output_gradient.stride, row, col);
        local_dot += s_j * g_j;
    }
    float row_dot = block_reduce_sum(local_dot);
    __shared__ float shared_dot;
    if (threadIdx.x == 0) {
        shared_dot = row_dot;
    }
    __syncthreads();
    row_dot = shared_dot;
    for (int col = threadIdx.x; col < softmax_gradient.cols; col += blockDim.x) {
        float s_j = get_float(softmax_output.data, softmax_output.stride, row, col);
        float g_j = get_float(output_gradient.data, output_gradient.stride, row, col);
        set_float(softmax_gradient.data, softmax_gradient.stride, row, col, s_j * (g_j - row_dot));
    }
}

__global__ void backprop_softmax_kernel_half(const const_matrix_view softmax_output,
                                          const const_matrix_view output_gradient,
                                          matrix_view softmax_gradient) {
    int row = blockIdx.x;
    float local_dot = 0.0f;
    for (int col = threadIdx.x; col < softmax_output.cols; col += blockDim.x) {
        half s_j = get_half(softmax_output.data, softmax_output.stride, row, col);
        half g_j = get_half(output_gradient.data, output_gradient.stride, row, col);
        local_dot += __half2float(s_j) * __half2float(g_j);
    }
    float row_dot = block_reduce_sum(local_dot);
    __shared__ float shared_dot;
    if (threadIdx.x == 0) {
        shared_dot = row_dot;
    }
    __syncthreads();
    row_dot = shared_dot;
    for (int col = threadIdx.x; col < softmax_gradient.cols; col += blockDim.x) {
        half s_j = get_half(softmax_output.data, softmax_output.stride, row, col);
        half g_j = get_half(output_gradient.data, output_gradient.stride, row, col);
        set_half(softmax_gradient.data, softmax_gradient.stride, row, col,
                 __float2half(__half2float(s_j) * (__half2float(g_j) - row_dot)));
    }
}

__global__ void backprop_softmax_kernel_bf16(const const_matrix_view softmax_output,
                                           const const_matrix_view output_gradient,
                                           matrix_view softmax_gradient) {
    int row = blockIdx.x;
    float local_dot = 0.0f;
    for (int col = threadIdx.x; col < softmax_output.cols; col += blockDim.x) {
        __nv_bfloat16 s_j = get_bf16(softmax_output.data, softmax_output.stride, row, col);
        __nv_bfloat16 g_j = get_bf16(output_gradient.data, output_gradient.stride, row, col);
        local_dot += __bfloat162float(s_j) * __bfloat162float(g_j);
    }
    float row_dot = block_reduce_sum(local_dot);
    __shared__ float shared_dot;
    if (threadIdx.x == 0) {
        shared_dot = row_dot;
    }
    __syncthreads();
    row_dot = shared_dot;
    for (int col = threadIdx.x; col < softmax_gradient.cols; col += blockDim.x) {
        __nv_bfloat16 s_j = get_bf16(softmax_output.data, softmax_output.stride, row, col);
        __nv_bfloat16 g_j = get_bf16(output_gradient.data, output_gradient.stride, row, col);
        set_bf16(softmax_gradient.data, softmax_gradient.stride, row, col,
                  __float2bfloat16(__bfloat162float(s_j) * (__bfloat162float(g_j) - row_dot)));
    }
}

// Comparison kernels
__global__ void compare_kernel_float(const const_matrix_view a, const const_matrix_view b,
                                   float epsilon, bool* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = a.rows * a.cols;
    if (idx < total_size) {
        int row = idx / a.cols;
        int col = idx % a.cols;
        float val_a = get_float(a.data, a.stride, row, col);
        float val_b = get_float(b.data, b.stride, row, col);
        if (fabsf(val_a - val_b) > epsilon) {
            *result = false;
        }
    }
}

__global__ void compare_kernel_half(const const_matrix_view a, const const_matrix_view b,
                                  float epsilon, bool* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = a.rows * a.cols;
    if (idx < total_size) {
        int row = idx / a.cols;
        int col = idx % a.cols;
        half val_a = get_half(a.data, a.stride, row, col);
        half val_b = get_half(b.data, b.stride, row, col);
        if (fabsf(__half2float(val_a) - __half2float(val_b)) > epsilon) {
            *result = false;
        }
    }
}

__global__ void compare_kernel_bf16(const const_matrix_view a, const const_matrix_view b,
                                   float epsilon, bool* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = a.rows * a.cols;
    if (idx < total_size) {
        int row = idx / a.cols;
        int col = idx % a.cols;
        __nv_bfloat16 val_a = get_bf16(a.data, a.stride, row, col);
        __nv_bfloat16 val_b = get_bf16(b.data, b.stride, row, col);
        if (fabsf(__bfloat162float(val_a) - __bfloat162float(val_b)) > epsilon) {
            *result = false;
        }
    }
}

// Reduction kernels
__global__ void sum_reduction_float(const const_matrix_view data, float* result, float identity) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    int size = data.rows * data.cols;
    float local_acc = identity;
    for (int i = blockIdx.x * total_threads + tid; i < size; i += gridDim.x * total_threads) {
        int r = i / data.cols;
        int c = i % data.cols;
        local_acc += get_float(data.data, data.stride, r, c);
    }
    float local_sum = block_reduce_sum(local_acc);
    if (tid == 0) {
        atomicAdd(result, local_sum);
    }
}

__global__ void sum_reduction_half(const const_matrix_view data, float* result, float identity) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    int size = data.rows * data.cols;
    float local_acc = identity;
    for (int i = blockIdx.x * total_threads + tid; i < size; i += gridDim.x * total_threads) {
        int r = i / data.cols;
        int c = i % data.cols;
        local_acc += __half2float(get_half(data.data, data.stride, r, c));
    }
    float local_sum = block_reduce_sum(local_acc);
    if (tid == 0) {
        atomicAdd(result, local_sum);
    }
}

__global__ void sum_reduction_bf16(const const_matrix_view data, float* result, float identity) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    int size = data.rows * data.cols;
    float local_acc = identity;
    for (int i = blockIdx.x * total_threads + tid; i < size; i += gridDim.x * total_threads) {
        int r = i / data.cols;
        int c = i % data.cols;
        local_acc += __bfloat162float(get_bf16(data.data, data.stride, r, c));
    }
    float local_sum = block_reduce_sum(local_acc);
    if (tid == 0) {
        atomicAdd(result, local_sum);
    }
}

// Variance kernels
__global__ void variance_kernel_float(const float* sum_ptr, const float* sum_sq_ptr,
                                   float* result, int total_elements) {
    float sum = *sum_ptr;
    float sum_of_squares = *sum_sq_ptr;
    float n = (float)total_elements;
    *result = (sum_of_squares / n) - (sum * sum) / (n * n);
}

__global__ void variance_kernel_half(const float* sum_ptr, const float* sum_sq_ptr,
                                  float* result, int total_elements) {
    float sum = *sum_ptr;
    float sum_of_squares = *sum_sq_ptr;
    float n = (float)total_elements;
    *result = (sum_of_squares / n) - (sum * sum) / (n * n);
}

__global__ void variance_kernel_bf16(const float* sum_ptr, const float* sum_sq_ptr,
                                    float* result, int total_elements) {
    float sum = *sum_ptr;
    float sum_of_squares = *sum_sq_ptr;
    float n = (float)total_elements;
    *result = (sum_of_squares / n) - (sum * sum) / (n * n);
}

} // namespace kernel::matrix::device
