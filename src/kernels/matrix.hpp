#pragma once

// Main header file for matrix kernel operations
// This includes all necessary headers for type-generic matrix operations

#include <kernels/matrix/cublas.hpp>
#include <kernels/matrix/host.hpp>
#include <kernels/scheduling.hpp>

#define CHECK_ERRORS(name) 