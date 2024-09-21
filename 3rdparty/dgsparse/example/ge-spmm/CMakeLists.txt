cmake_minimum_required(VERSION 3.16)

project(gespmm C CXX CUDA)
find_package( CUDAToolkit )
set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++14")

file(GLOB GESPMMSRC
  *.cc
)

add_executable(dgsparse-gespmm ${GESPMMSRC})
target_include_directories(dgsparse-gespmm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cnpy/)
target_include_directories(dgsparse-gespmm PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(dgsparse-gespmm PRIVATE CUDA::cusparse CUDA::cublas CUDA::curand CUDA::cudart cnpy dgsparse)

install(TARGETS dgsparse-gespmm)
