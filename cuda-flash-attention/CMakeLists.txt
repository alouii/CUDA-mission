cmake_minimum_required(VERSION 3.18)
project(FlashAttentionBenchmark LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_executable(flash_attention
    src/main.cu
    src/flash_attention_warp.cu
    src/naive_attention.cu)

set_target_properties(flash_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
