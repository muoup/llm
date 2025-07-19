cmake_minimum_required(VERSION 3.31)
project(llm LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -fsanitize=undefined -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

find_package(OpenMP)

file(GLOB_RECURSE SOURCE "src/*.cpp")

add_executable(llm ${SOURCE})

if(OpenMP_CXX_FOUND)
    target_link_libraries(llm PUBLIC OpenMP::OpenMP_CXX)
endif()