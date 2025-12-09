cmake_minimum_required(VERSION 3.25)
project(llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BLA_VENDOR "OpenBLAS")
find_package(BLAS REQUIRED)

add_compile_options(-fopenmp)
add_link_options(-fopenmp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build")
    add_compile_options(-DMATRIX_CHECKS -DATTENTION_DEBUG)
    add_compile_options(-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG)
    add_compile_options(-Og -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
else()
    message(STATUS "Release build")
    add_compile_options(-O3 -march=native)
endif()

include(FetchContent)
FetchContent_Declare(
    blaze
    GIT_REPOSITORY https://bitbucket.org/blaze-lib/blaze.git
    GIT_TAG        v3.8
)
FetchContent_MakeAvailable(blaze)

file(GLOB_RECURSE SOURCE "src/*.cpp")

add_executable(llm ${SOURCE})

target_compile_definitions(llm PRIVATE
    $<$<CONFIG:Debug>:MATRIX_CHECKS>
)

target_include_directories(llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(llm PRIVATE blaze)
