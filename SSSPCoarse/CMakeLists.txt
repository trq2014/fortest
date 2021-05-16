cmake_minimum_required(VERSION 3.19)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(SSSPCoarse CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(SSSPCoarse main.cu common.cu common.cuh ArgumentParser.cu ArgumentParser.cuh gpu_kernels.cu gpu_kernels.cuh range.hpp globals.cuh pagerank.cu pagerank.cuh sssp.cu sssp.cuh)
target_link_libraries(SSSPCoarse -lcublas -lcurand)
set_target_properties(
        SSSPCoarse
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)