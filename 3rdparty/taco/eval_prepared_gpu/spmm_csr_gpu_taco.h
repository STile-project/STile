#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <thrust/complex.h>
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
__device__ __host__ int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}
__device__ __host__ int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
__global__ void taco_binarySearchBeforeBlock(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int num_blocks) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx >= num_blocks+1) {
    return;
  }

  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, idx * values_per_block);
}

__host__ int * taco_binarySearchBeforeBlockLaunch(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int block_size, int num_blocks){
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  CUDA_KERNEL_CALL(taco_binarySearchBeforeBlock, num_search_blocks, block_size, 0, nullptr, array, results, arrayStart, arrayEnd, values_per_block, num_blocks);
  return results;
}
__global__ void taco_binarySearchIndirectBeforeBlock(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int * __restrict__ targets, int num_blocks) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx >= num_blocks+1) {
    return;
  }

  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, targets[idx]);
}

__host__ int * taco_binarySearchIndirectBeforeBlockLaunch(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int * __restrict__ targets, int block_size, int num_blocks){
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  CUDA_KERNEL_CALL(taco_binarySearchIndirectBeforeBlock, num_search_blocks, block_size, 0, nullptr, array, results, arrayStart, arrayEnd, targets, num_blocks);
  return results;
}
template<typename T>
__device__ inline void atomicAddWarp(T *array, int index, T val)
{
  int leader_index = __shfl_sync(-1, index, 0);
  int mask = __ballot_sync(-1, leader_index == index);
  if(mask == -1) {
    val += __shfl_down_sync(-1, val, 16);
    val += __shfl_down_sync(-1, val, 8);
    val += __shfl_down_sync(-1, val, 4);
    val += __shfl_down_sync(-1, val, 2);
    val += __shfl_down_sync(-1, val, 1);
    if(threadIdx.x % 32 == 0) {
      atomicAdd(&array[index], val);
    }
  } else {
    atomicAdd(&array[index], val);
  }
}

typedef void(*kernelFunction_t)(taco_tensor_t*, taco_tensor_t*, taco_tensor_t*, int32_t*);

__global__
void spmm_csr_gpu_tacoDeviceKernel_16_1(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16;
    int32_t fposA = block * 16 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 16; nnz++) {
      int32_t fpos1 = warp * 16 + nnz;
      int32_t fposA = block * 16 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_16_2(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16;
    int32_t fposA = block * 32 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 16; nnz++) {
      int32_t fpos1 = warp * 16 + nnz;
      int32_t fposA = block * 32 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_16_4(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16;
    int32_t fposA = block * 64 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 16; nnz++) {
      int32_t fpos1 = warp * 16 + nnz;
      int32_t fposA = block * 64 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_16_8(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    } 
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16;
    int32_t fposA = block * 128 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 16; nnz++) {
      int32_t fpos1 = warp * 16 + nnz;
      int32_t fposA = block * 128 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_16_16(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16;
    int32_t fposA = block * 256 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 16; nnz++) {
      int32_t fpos1 = warp * 16 + nnz;
      int32_t fposA = block * 256 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_16_32(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16;
    int32_t fposA = block * 512 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 16; nnz++) {
      int32_t fpos1 = warp * 16 + nnz;
      int32_t fposA = block * 512 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_32_1(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 32;
    int32_t fposA = block * 32 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 32; nnz++) {
      int32_t fpos1 = warp * 32 + nnz;
      int32_t fposA = block * 32 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_32_2(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 32;
    int32_t fposA = block * 64 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 32; nnz++) {
      int32_t fpos1 = warp * 32 + nnz;
      int32_t fposA = block * 64 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_32_4(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 32;
    int32_t fposA = block * 128 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 32; nnz++) {
      int32_t fpos1 = warp * 32 + nnz;
      int32_t fposA = block * 128 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_32_8(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 32;
    int32_t fposA = block * 256 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 32; nnz++) {
      int32_t fpos1 = warp * 32 + nnz;
      int32_t fposA = block * 256 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_32_16(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 32;
    int32_t fposA = block * 512 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 32; nnz++) {
      int32_t fpos1 = warp * 32 + nnz;
      int32_t fposA = block * 512 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_32_32(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 32;
    int32_t fposA = block * 1024 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 32; nnz++) {
      int32_t fpos1 = warp * 32 + nnz;
      int32_t fposA = block * 1024 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_64_1(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 64;
    int32_t fposA = block * 64 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 64; nnz++) {
      int32_t fpos1 = warp * 64 + nnz;
      int32_t fposA = block * 64 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_64_2(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 64;
    int32_t fposA = block * 128 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 64; nnz++) {
      int32_t fpos1 = warp * 64 + nnz;
      int32_t fposA = block * 128 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_64_4(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 64;
    int32_t fposA = block * 256 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 64; nnz++) {
      int32_t fpos1 = warp * 64 + nnz;
      int32_t fposA = block * 256 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_64_8(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 64;
    int32_t fposA = block * 512 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 64; nnz++) {
      int32_t fpos1 = warp * 64 + nnz;
      int32_t fposA = block * 512 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_64_16(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 64;
    int32_t fposA = block * 1024 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 64; nnz++) {
      int32_t fpos1 = warp * 64 + nnz;
      int32_t fposA = block * 1024 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_64_32(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 64;
    int32_t fposA = block * 2048 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 64; nnz++) {
      int32_t fpos1 = warp * 64 + nnz;
      int32_t fposA = block * 2048 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_128_1(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 128;
    int32_t fposA = block * 128 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 128; nnz++) {
      int32_t fpos1 = warp * 128 + nnz;
      int32_t fposA = block * 128 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_128_2(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 128;
    int32_t fposA = block * 256 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 128; nnz++) {
      int32_t fpos1 = warp * 128 + nnz;
      int32_t fposA = block * 256 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_128_4(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 128;
    int32_t fposA = block * 512 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 128; nnz++) {
      int32_t fpos1 = warp * 128 + nnz;
      int32_t fposA = block * 512 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_128_8(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 128;
    int32_t fposA = block * 1024 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 128; nnz++) {
      int32_t fpos1 = warp * 128 + nnz;
      int32_t fposA = block * 1024 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_128_16(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 128;
    int32_t fposA = block * 2048 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 128; nnz++) {
      int32_t fpos1 = warp * 128 + nnz;
      int32_t fposA = block * 2048 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_128_32(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 128;
    int32_t fposA = block * 4096 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 128; nnz++) {
      int32_t fpos1 = warp * 128 + nnz;
      int32_t fposA = block * 4096 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_256_1(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 256;
    int32_t fposA = block * 256 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 256; nnz++) {
      int32_t fpos1 = warp * 256 + nnz;
      int32_t fposA = block * 256 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_256_2(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 256;
    int32_t fposA = block * 512 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 256; nnz++) {
      int32_t fpos1 = warp * 256 + nnz;
      int32_t fposA = block * 512 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_256_4(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 256;
    int32_t fposA = block * 1024 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 256; nnz++) {
      int32_t fpos1 = warp * 256 + nnz;
      int32_t fposA = block * 1024 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_256_8(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 256;
    int32_t fposA = block * 2048 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 256; nnz++) {
      int32_t fpos1 = warp * 256 + nnz;
      int32_t fposA = block * 2048 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_256_16(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 256;
    int32_t fposA = block * 4096 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 256; nnz++) {
      int32_t fpos1 = warp * 256 + nnz;
      int32_t fposA = block * 4096 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_256_32(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 256;
    int32_t fposA = block * 8192 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 256; nnz++) {
      int32_t fpos1 = warp * 256 + nnz;
      int32_t fposA = block * 8192 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_512_1(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 512;
    int32_t fposA = block * 512 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 512; nnz++) {
      int32_t fpos1 = warp * 512 + nnz;
      int32_t fposA = block * 512 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_512_2(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 512;
    int32_t fposA = block * 1024 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 512; nnz++) {
      int32_t fpos1 = warp * 512 + nnz;
      int32_t fposA = block * 1024 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_512_4(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 512;
    int32_t fposA = block * 2048 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 512; nnz++) {
      int32_t fpos1 = warp * 512 + nnz;
      int32_t fposA = block * 2048 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_512_8(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 512;
    int32_t fposA = block * 4096 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 512; nnz++) {
      int32_t fpos1 = warp * 512 + nnz;
      int32_t fposA = block * 4096 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_512_16(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 512;
    int32_t fposA = block * 8192 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 512; nnz++) {
      int32_t fpos1 = warp * 512 + nnz;
      int32_t fposA = block * 8192 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel_512_32(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);

  for (int32_t dense_val = 0; ; ++dense_val) {
    int32_t k = dense_val * 32 + thread;
    if (k >= B2_dimension) {
      break;
    }
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 512;
    int32_t fposA = block * 16384 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 512; nnz++) {
      int32_t fpos1 = warp * 512 + nnz;
      int32_t fposA = block * 16384 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = i * B2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    if (i < C1_dimension) {
      int32_t iC = i * B2_dimension + k;
      atomicAdd(&C_vals[iC], tnnz_val);
    }
  }
}

kernelFunction_t GetKernelFunc(int nnz_per_warp, int warp_per_tb) {
  if (nnz_per_warp == 16) {
    switch (warp_per_tb) {
      case 1: return &spmm_csr_gpu_tacoDeviceKernel_16_1;
      case 2: return &spmm_csr_gpu_tacoDeviceKernel_16_2;
      case 4: return &spmm_csr_gpu_tacoDeviceKernel_16_4;
      case 8: return &spmm_csr_gpu_tacoDeviceKernel_16_8;
      case 16: return &spmm_csr_gpu_tacoDeviceKernel_16_16;
      case 32: return &spmm_csr_gpu_tacoDeviceKernel_16_32;
    }
  } else if (nnz_per_warp == 32) {
    switch (warp_per_tb) {
      case 1: return &spmm_csr_gpu_tacoDeviceKernel_32_1;
      case 2: return &spmm_csr_gpu_tacoDeviceKernel_32_2;
      case 4: return &spmm_csr_gpu_tacoDeviceKernel_32_4;
      case 8: return &spmm_csr_gpu_tacoDeviceKernel_32_8;
      case 16: return &spmm_csr_gpu_tacoDeviceKernel_32_16;
      case 32: return &spmm_csr_gpu_tacoDeviceKernel_32_32;
    }
  } else if (nnz_per_warp == 64) {
    switch (warp_per_tb) {
      case 1: return &spmm_csr_gpu_tacoDeviceKernel_64_1;
      case 2: return &spmm_csr_gpu_tacoDeviceKernel_64_2;
      case 4: return &spmm_csr_gpu_tacoDeviceKernel_64_4;
      case 8: return &spmm_csr_gpu_tacoDeviceKernel_64_8;
      case 16: return &spmm_csr_gpu_tacoDeviceKernel_64_16;
      case 32: return &spmm_csr_gpu_tacoDeviceKernel_64_32;
    }
  } else if (nnz_per_warp == 128) {
    switch(warp_per_tb) {
      case 1: return &spmm_csr_gpu_tacoDeviceKernel_128_1;
      case 2: return &spmm_csr_gpu_tacoDeviceKernel_128_2;
      case 4: return &spmm_csr_gpu_tacoDeviceKernel_128_4;
      case 8: return &spmm_csr_gpu_tacoDeviceKernel_128_8;
      case 16: return &spmm_csr_gpu_tacoDeviceKernel_128_16;
      case 32: return &spmm_csr_gpu_tacoDeviceKernel_128_32;
    }
  } else if (nnz_per_warp == 256) {
    switch (warp_per_tb) {
      case 1: return &spmm_csr_gpu_tacoDeviceKernel_256_1;
      case 2: return &spmm_csr_gpu_tacoDeviceKernel_256_2;
      case 4: return &spmm_csr_gpu_tacoDeviceKernel_256_4;
      case 8: return &spmm_csr_gpu_tacoDeviceKernel_256_8;
      case 16: return &spmm_csr_gpu_tacoDeviceKernel_256_16;
      case 32: return &spmm_csr_gpu_tacoDeviceKernel_256_32;
    }
  } else if (nnz_per_warp == 512) {
    switch (warp_per_tb) {
      case 1: return &spmm_csr_gpu_tacoDeviceKernel_512_1;
      case 2: return &spmm_csr_gpu_tacoDeviceKernel_512_2;
      case 4: return &spmm_csr_gpu_tacoDeviceKernel_512_4;
      case 8: return &spmm_csr_gpu_tacoDeviceKernel_512_8;
      case 16: return &spmm_csr_gpu_tacoDeviceKernel_512_16;
      case 32: return &spmm_csr_gpu_tacoDeviceKernel_512_32;
    }
  }
  throw;
}
