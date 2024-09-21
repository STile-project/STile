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
#include "spmm_csr_gpu_taco.h"
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

std::string config_str(int nnz_per_warp, int warp_per_tb) {
  return "config(nnz_per_warp=" + std::to_string(nnz_per_warp) + ", warp_per_tb=" + std::to_string(warp_per_tb) + ")";
}

int spmm_csr_gpu_taco(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B, bool profile, int nnz_per_warp, int warp_per_tb) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  float* __restrict__ C_vals = (float*)(C->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);

  int nnz_per_tb = nnz_per_warp * warp_per_tb;
  int32_t* i_blockStarts = 0;
  cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + nnz_per_tb - 1) / nnz_per_tb + 1));
  bool bsearch_failed = false;
  try {
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(A2_pos, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) nnz_per_tb, (int32_t) warp_per_tb * 32, ((A2_pos[A1_dimension] + nnz_per_tb - 1) / nnz_per_tb));
  } catch (std::exception const& e) {
    std::cerr << "Binary search failed: " << config_str(nnz_per_warp, warp_per_tb) << ", error string:\n" << e.what() << "\n";
    bsearch_failed = true;
  }
  cudaDeviceSynchronize();

  kernelFunction_t kernel_func = GetKernelFunc(nnz_per_warp, warp_per_tb);

  if (!bsearch_failed) {
    try {
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
            CUDA_KERNEL_CALL(kernel_func, (A2_pos[A1_dimension] + nnz_per_tb - 1) / nnz_per_tb, 32 * warp_per_tb, 0, nullptr, A, B, C, i_blockStarts);
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
            CUDA_KERNEL_CALL(kernel_func, (A2_pos[A1_dimension] + nnz_per_tb - 1) / nnz_per_tb, 32 * warp_per_tb, 0, nullptr, A, B, C, i_blockStarts);
          }
          gpu_timer.stop();
          kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
        }
        printf("nnz_per_warp %d warp_per_tb %d Time %f (ms)\n", nnz_per_warp, warp_per_tb, kernel_dur_msecs);
      } else {
        kernel_func<<<(A2_pos[A1_dimension] + nnz_per_tb - 1) / nnz_per_tb, 32 * warp_per_tb>>>(A, B, C, i_blockStarts);
      }
    } catch (std::exception const& e) {
      std::cerr << "Profile failed: " << config_str(nnz_per_warp, warp_per_tb) << ", error string:\n" << e.what() << "\n";
    }
  }
  cudaDeviceSynchronize();

  cudaFree(i_blockStarts);

  C->vals = (uint8_t*)C_vals;
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

int main(int argc, char *argv[]) {
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
  Tensor<float> B("B", {N, K}, {Dense, Dense});
  Tensor<float> C("C", {M, K}, Format({{Dense, Dense}, {1, 0}}));
  // Tensor<float> expected("expected", {M, K}, Format({{Dense, Dense}, {1, 0}}));

  for (int t = 0; t < nnz; ++t) {
    int i = row_buffer[t];
    int j = csr_indices_buffer[t];
    float rand_float = (float)rand() / (float)(RAND_MAX);
    A.insert({i, j}, rand_float);
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      float rand_float = (float)rand() / (float)(RAND_MAX);
      B.insert({i, j}, rand_float);
    }
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      C.insert({i, j}, 0.0f);
    }
  }

  // for (int t = 0; t < nnz; ++t) {
  //   int i = row_buffer[t];
  //   int j = csr_indices_buffer[t];
  //   for (int k = 0; k < K; ++k) {
  //     expected_val[i * K + k] += A_val[t] * B_val[j * K + k];
  //   }
  // }
  // for (int i = 0; i < M; ++i) {
  //   for (int k = 0; k < K; ++k) {
  //     expected.insert({i, k}, expected_val[i * K + k]);
  //   }
  // }

  A.pack();
  B.pack();
  C.pack();
  // expected.pack();

  const int arr_nnz_per_warp[] = {16, 32, 64, 128, 256, 512};
  const int arr_warp_per_tb[] = {1, 2, 4, 8, 16, 32};

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      spmm_csr_gpu_taco(C.getTacoTensorT(), A.getTacoTensorT(), B.getTacoTensorT(), true, arr_nnz_per_warp[i], arr_warp_per_tb[j]);
    }
  }
  // spmm_csr_gpu_taco(C.getTacoTensorT(), A.getTacoTensorT(), B.getTacoTensorT(), false, 512, 32);

  // ASSERT_TENSOR_EQ(expected, C);

  return 0;
}

