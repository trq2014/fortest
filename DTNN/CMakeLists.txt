cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(DTNN CUDA)

set(CMAKE_CUDA_STANDARD 14)
include_directories(/usr/local/cuda/samples/common/inc/)
add_executable(DTNN main.cu test.cuh)

set_target_properties(
        DTNN
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(DTNN -lcublas -lcurand)