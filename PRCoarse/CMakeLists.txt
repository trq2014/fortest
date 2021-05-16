cmake_minimum_required(VERSION 3.19)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(BFSCoarse CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(BFSCoarse main.cu common.cu common.cuh ArgumentParser.cu ArgumentParser.cuh gpu_kernels.cu gpu_kernels.cuh range.hpp globals.cuh pagerank.cu pagerank.cuh)
target_link_libraries(BFSCoarse -lcublas -lcurand)
set_target_properties(
        BFSCoarse
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)