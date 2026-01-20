cmake_minimum_required(VERSION 3.18)
project(CudaReLU LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_relu
    src/main.cu
    src/relu_naive.cu
    src/relu_optimized.cu)

set_target_properties(cuda_relu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
