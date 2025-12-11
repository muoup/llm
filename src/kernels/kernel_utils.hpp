#pragma once

#include <cublas_v2.h>
#include <cstdlib>

struct cublas_handle {
    cublasHandle_t handle;

    cublas_handle() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            std::abort();
        }
    }

    ~cublas_handle() { cublasDestroy(handle); }

    operator cublasHandle_t() const { return handle; }
};

inline cublas_handle handle;