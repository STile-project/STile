cmake_minimum_required(VERSION 3.16)

project(eval_prepared_gpu C CXX CUDA)
find_package( CUDAToolkit )
set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++14")

file(GLOB GTEST_SOURCE ../test/gtest/gtest-all.cc)
file(GLOB GTEST_HEADER ../test/gtest/gtest.h)
include_directories(${TACO_TEST_DIR} ${TACO_SRC_DIR})
add_library(gtest ${GTEST_HEADER} ${GTEST_SOURCE})

add_executable(taco-sddmm sddmm_csr_gpu_taco.cu)
add_executable(taco-spmm spmm_csr_gpu_taco.cu)

target_include_directories(taco-sddmm PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(taco-spmm PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(taco-sddmm PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
set_target_properties(taco-spmm PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
target_link_libraries(taco-sddmm PRIVATE CUDA::cusparse CUDA::cublas CUDA::curand CUDA::cudart taco gtest cnpy)
target_link_libraries(taco-spmm PRIVATE CUDA::cusparse CUDA::cublas CUDA::curand CUDA::cudart taco gtest cnpy)
install(TARGETS taco-spmm taco-sddmm)
