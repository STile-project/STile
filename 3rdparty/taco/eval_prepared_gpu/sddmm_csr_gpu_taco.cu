#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cnpy.h>
#include <thrust/complex.h>
#include "atomic.cuh"
#include <random>
#include <string>
#include "taco.h"
#include <cuda_runtime_api.h>
#include "../test/gtest/gtest.h"
#include "utils.cuh"
#include "sddmm_csr_gpu_taco.h"
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

using namespace taco;


void ASSERT_TENSOR_EQ(TensorBase expected, TensorBase actual) {
  SCOPED_TRACE(std::string("expected: ") + util::toString(expected));
  SCOPED_TRACE(std::string("  actual: ") + util::toString(actual));
  ASSERT_TRUE(equals(expected, actual));
}

int sddmm_csr_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D, bool profile, int nnz_per_warp, int warp_per_tb) {
  int B1_dimension = (int)(B->dimensions[0]);
  int* __restrict__ B2_pos = (int*)(B->indices[1][0]);

  int nnz_per_tb = nnz_per_warp * warp_per_tb;
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B2_pos[B1_dimension] + nnz_per_tb - 1) / nnz_per_tb + 1)));
  i_blockStarts = taco_binarySearchBeforeBlockLaunch(B2_pos, i_blockStarts, (int32_t) 0, B1_dimension, (int32_t) nnz_per_tb, (int32_t) nnz_per_warp, ((B2_pos[B1_dimension] + nnz_per_tb - 1) / nnz_per_tb));

  kernelFunction_t kernel_func = GetKernelFunc(nnz_per_warp, warp_per_tb);

  if (profile) {
    char *env_flush_l2 = std::getenv("FLUSH_L2");
    bool flush_l2 = env_flush_l2 ? std::strcmp(env_flush_l2, "ON") == 0 : false;

    GpuTimer gpu_timer;
    int warmup_iter = 10;
    int repeat_iter = 100;
    float kernel_dur_msecs = 0;
    if (flush_l2) {
      for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
        if (iter >= warmup_iter) {
          gpu_timer.start(true);
        }
        CUDA_KERNEL_CALL(kernel_func, (B2_pos[B1_dimension] + nnz_per_tb - 1) / nnz_per_tb, 32 * warp_per_tb, 0, nullptr, A, B, C, D, i_blockStarts);
        if (iter >= warmup_iter) {
          gpu_timer.stop();
          kernel_dur_msecs += gpu_timer.elapsed_msecs();
        }
      }
      kernel_dur_msecs /= repeat_iter;
    } else {
      for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
        if (iter == warmup_iter) {
          gpu_timer.start(false);
        }
        CUDA_KERNEL_CALL(kernel_func, (B2_pos[B1_dimension] + nnz_per_tb - 1) / nnz_per_tb, 32 * warp_per_tb, 0, nullptr, A, B, C, D, i_blockStarts);
      }
      gpu_timer.stop();
      kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
    }
    printf("nnz_per_warp %d warp_per_tb %d Time %f (ms)\n", nnz_per_warp, warp_per_tb, kernel_dur_msecs);
  } else {
    kernel_func<<<(B2_pos[B1_dimension] + nnz_per_tb - 1) / nnz_per_tb, 32 * warp_per_tb>>>(A, B, C, D, i_blockStarts);
  }
  cudaDeviceSynchronize();

  cudaFree(i_blockStarts);

  return 0;
}

void read_npz_file(const std::string& filename, int &M, int &K, int &NNZ, std::vector<int>& indptr, std::vector<int>& indices) {
  cnpy::npz_t data = cnpy::npz_load(filename);
  cnpy::NpyArray shape = data["shape"];
  int* shape_data = shape.data<int>();
  M = shape_data[0];
  K = shape_data[1];
  NNZ = shape_data[2];
  indptr = std::move(data["indptr"].as_vec<int>());
  indices = std::move(data["indices"].as_vec<int>());
}

// float C_val[100000000], D_val[100000000];

int main(int argc, char *argv[]) {
  // std::default_random_engine gen(0);
  // std::uniform_real_distribution<float> unif(0.0, 1.0);
  int M;
  int N;
  int nnz;
  std::vector<int> csr_indptr_buffer;
  std::vector<int> row_buffer;
  std::vector<int> csr_indices_buffer;

  read_npz_file(argv[1], M, N, nnz, csr_indptr_buffer, csr_indices_buffer);
  int row = 0;
  for (int i = 0; i < csr_indptr_buffer.size() - 1; ++i) {
    for (int j = 0; j < csr_indptr_buffer[i + 1] - csr_indptr_buffer[i]; ++j) {
      row_buffer.push_back(row);
    } 
    row++;
  }
  printf("M %d nnz %d\n", M, nnz);

  int K = std::stoi(argv[2]);

  Tensor<float> A("A", {M, N}, CSR);
  Tensor<float> B("B", {M, N}, CSR);
  Tensor<float> C("C", {M, K}, {Dense, Dense});
  Tensor<float> D("D", {K, N}, Format({{Dense, Dense}, {1, 0}}));
  // Tensor<float> expected("expected", {M, N}, CSR);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      float rand_float = (float)rand() / (float)(RAND_MAX);
      C.insert({i, j}, rand_float);
    }
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      float rand_float = (float)rand() / (float)(RAND_MAX);
      D.insert({i, j}, rand_float);
    }
  }

  C.pack();
  D.pack();

  for (int t = 0; t < nnz; ++t) {
    int i = row_buffer[t];
    int j = csr_indices_buffer[t];
    B.insert({i, j}, 1.0f);
    A.insert({i, j}, 0.0f);

    // float dot = 0.0f;
    // for (int k = 0; k < K; ++k) {
    //   dot += C_val[i * K + k] * D_val[k * N + j];
    // }
    // expected.insert({i, j}, dot);
  }
  B.pack();
  A.pack();
  // expected.pack();

  const int arr_nnz_per_warp[] = {16, 32, 64, 128, 256, 512};
  const int arr_warp_per_tb[] = {1, 2, 4, 8, 16, 32};

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      sddmm_csr_gpu_taco(A.getTacoTensorT(), B.getTacoTensorT(), C.getTacoTensorT(), D.getTacoTensorT(), true, arr_nnz_per_warp[i], arr_warp_per_tb[j]);
    }
  }
  // sddmm_csr_gpu_taco(A.getTacoTensorT(), B.getTacoTensorT(), C.getTacoTensorT(), D.getTacoTensorT(), false, 512, 16);

  // ASSERT_TENSOR_EQ(expected, A);

  return 0;
}
