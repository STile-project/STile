cmake_minimum_required(VERSION 3.24)

add_subdirectory(3rdparty/cnpy)

project(dgsparse C CXX CUDA)
find_package( CUDAToolkit )
set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++14")

file(GLOB DGSPARSE_SRC
	src/ge-spmm/*.cu
  src/ge-spmm/*.cc
  src/sddmm/*.cu
)

# Helper to create CUDA gencode flags.
function(create_cuda_gencode_flags out archs_args)
  set(archs ${archs_args} ${ARGN})
  set(tmp "")
  foreach(arch IN LISTS archs)
    set(tmp "${tmp} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach(arch)
  set(${out} ${tmp} PARENT_SCOPE)
endfunction()

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS "70;75;80;86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")

add_library(dgsparse SHARED ${DGSPARSE_SRC})
target_include_directories(dgsparse PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(dgsparse PRIVATE CUDA::cusparse CUDA::cublas CUDA::curand CUDA::cudart cnpy)
install(TARGETS dgsparse)

add_subdirectory(example/ge-spmm)
add_subdirectory(example/sddmm)

set_property(TARGET dgsparse dgsparse-sddmm dgsparse-gespmm PROPERTY CUDA_ARCHITECTURES native)
