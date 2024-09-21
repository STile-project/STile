cmake_minimum_required(VERSION 3.24)

project(sddmm C CXX CUDA)
find_package( CUDAToolkit )
set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++14")

file(GLOB SDDMMSRC
  *.cc
)

add_executable(dgsparse-sddmm ${SDDMMSRC})
target_include_directories(dgsparse-sddmm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cnpy/)
target_include_directories(dgsparse-sddmm PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(dgsparse-sddmm PRIVATE CUDA::cusparse CUDA::cublas CUDA::curand CUDA::cudart cnpy dgsparse)

install(TARGETS dgsparse-sddmm)