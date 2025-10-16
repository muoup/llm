cmake_minimum_required(VERSION 3.25)
project(llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-fopenmp -O3)
add_compile_options(-fno-exceptions -Wall -Wextra -Wpedantic -Werror)
add_link_options(-fopenmp)

file(GLOB_RECURSE SOURCE "src/*.cpp")

add_executable(llm ${SOURCE})

target_compile_definitions(llm PRIVATE
    $<$<CONFIG:Debug>:MATRIX_CHECKS>
)

target_include_directories(llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
