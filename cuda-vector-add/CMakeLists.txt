cmake_minimum_required(VERSION 3.18)
project(cuda_vector_add LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(vector_add
    src/vector_add.cu
)

set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
add_executable(vector_add_graph
    src/vector_add_graph.cu
)
add_executable(vector_add_graph_events
    src/vector_add_graph_events.cu
)