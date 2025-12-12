if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

cmake_minimum_required(VERSION 3.25)
project(llm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_compile_options(-fopenmp)
add_link_options(-fopenmp)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMATRIX_CHECKS")
    add_compile_options(-DMATRIX_CHECKS -DATTENTION_DEBUG)
    add_compile_options(-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG)
    add_compile_options(-O0 -g -fno-omit-frame-pointer -fPIC)
    add_link_options(-fno-omit-frame-pointer -fPIC)
else()
    message(STATUS "Release build")
    add_compile_options(-O3 -march=native)
endif()

file(GLOB_RECURSE SOURCE "src/*.cpp" "src/*.cu")

add_executable(llm ${SOURCE})

target_compile_definitions(llm PRIVATE
    $<$<CONFIG:Debug>:MATRIX_CHECKS>
)

target_include_directories(llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(llm PRIVATE CUDA::cudart CUDA::cublas CUDA::curand)
