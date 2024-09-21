// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../benchmark_utils.h"
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "sputnik/spmm/spmm_config.h"
#include "sputnik/test_utils.h"


// Fill a host array with all 0
template <typename DType>
void fill_zero(DType array[], int size) {
  memset(array, 0x0, sizeof(array[0]) * size);
}

// Compute spmm correct numbers. All arrays are host memory locations.
template <typename Index, typename DType>
void spmm_reference_host(int M,  // number of A-rows
                         int N,  // number of B_columns
                         int K,  // number of A columns
                         const Index *csr_indptr, const int *csr_indices,
                         const DType *csr_values,  // three arrays of A's CSR format
                         const DType *B,           // assume row-major
                         DType *C_ref)             // assume row-major
{
  fill_zero(C_ref, M * N);
  for (int64_t i = 0; i < M; i++) {
    Index begin = csr_indptr[i];
    Index end = csr_indptr[i + 1];
    for (Index p = begin; p < end; p++) {
      int k = csr_indices[p];
      DType val = csr_values[p];
      for (int64_t j = 0; j < N; j++) {
        C_ref[i * N + j] += val * B[k * N + j];
      }
    }
  }
}

// Fill a host array with random numbers.
void fill_random(float array[], int size) {
  for (int i = 0; i < size; i++) {
    array[i] = (float)(std::rand() % 3) / 10;
  }
}

// Compare two MxN matrices
template <typename DType>
bool check_result(int M, int N, DType *C, DType *C_ref) {
  bool passed = true;
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      DType c = C[i * N + j];
      DType c_ref = C_ref[i * N + j];
      if (fabs(c - c_ref) > 1e-2 * fabs(c_ref)) {
        printf("Wrong result: i = %ld, j = %ld, result = %lf, reference = %lf.\n", i, j, c, c_ref);
        passed = false;
      }
    }
  }
  return passed;
}

int main(int argc, const char **argv) {
  /// check command-line argument

  if (argc < 2) {
    printf(
        "Require command-line argument: name of the sparse matrix file in "
        ".mtx format.\n");
    return EXIT_FAILURE;
  }

  //
  // Load sparse matrix
  //

  int M;                               // number of A-rows
  int K;                               // number of A-columns
  int nnz;                             // number of non-zeros in A
  std::vector<int> csr_indptr_buffer;  // buffer for indptr array in CSR format
  std::vector<int> row_indices_buffer;
  std::vector<int> csr_indices_buffer;  // buffer for indices (column-ids) array in CSR format
  // load sparse matrix from mtx file
  // read_mtx_file(argv[1], M, K, nnz, csr_indptr_buffer, csr_indices_buffer);
  read_npz_file(argv[1], M, K, nnz, csr_indptr_buffer, csr_indices_buffer);
  for (int i = 0; i < M; ++i) {
    row_indices_buffer.push_back(i);
  }

  printf(
      "Finish reading matrix %d rows, %d columns, %d nnz. \nIgnore original "
      "values and use randomly generated values.\n",
      M, K, nnz);

  // Create GPU arrays
  int N = 128;  // number of B-columns
  if (argc > 2) {
    N = atoi(argv[2]);
  }
  assert(N > 0 && "second command-line argument is number of B columns, should be >0.\n");

  float *B_h = NULL, *C_h = NULL, *csr_values_h = NULL, *C_ref = NULL;
  float *B_d = NULL, *C_d = NULL, *csr_values_d = NULL;
  int *csr_indptr_d = NULL, *csr_indices_d = NULL, *row_indices_d = NULL;

  B_h = (float *)malloc(sizeof(float) * K * N);
  C_h = (float *)malloc(sizeof(float) * M * N);
  C_ref = (float *)malloc(sizeof(float) * M * N);
  csr_values_h = (float *)malloc(sizeof(float) * nnz);
  if (!B_h || !C_h || !C_ref || !csr_values_h) {
    printf("Host allocation failed.\n");
    return EXIT_FAILURE;
  }

  fill_random(csr_values_h, nnz);
  fill_random(B_h, K * N);

  CUDA_CHECK(cudaMalloc((void **)&B_d, sizeof(float) * K * N));
  CUDA_CHECK(cudaMalloc((void **)&C_d, sizeof(float) * M * N));
  CUDA_CHECK(cudaMalloc((void **)&csr_values_d, sizeof(float) * nnz));
  CUDA_CHECK(cudaMalloc((void **)&csr_indptr_d, sizeof(int) * (M + 1)));
  CUDA_CHECK(cudaMalloc((void **)&row_indices_d, sizeof(int) * M));
  CUDA_CHECK(cudaMalloc((void **)&csr_indices_d, sizeof(int) * nnz));

  CUDA_CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * K * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(C_d, 0x0, sizeof(float) * M * N));
  CUDA_CHECK(cudaMemcpy(row_indices_d, row_indices_buffer.data(), sizeof(int) * M,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(csr_values_d, csr_values_h, sizeof(float) * nnz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(csr_indptr_d, csr_indptr_buffer.data(), sizeof(int) * (M + 1),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(csr_indices_d, csr_indices_buffer.data(), sizeof(int) * nnz,
                        cudaMemcpyHostToDevice));
  // check result
  CUDA_CHECK(sputnik::CudaSpmm(M, K, N, nnz, row_indices_d, csr_values_d, csr_indptr_d,
                               csr_indices_d, B_d, C_d, 0));
  CUDA_CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
  spmm_reference_host<int, float>(M, N, K, csr_indptr_buffer.data(), csr_indices_buffer.data(),
                                  csr_values_h, B_h, C_ref);

  bool correct = check_result<float>(M, N, C_h, C_ref);

  // benchmark
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
      CUDA_CHECK(sputnik::CudaSpmm(M, K, N, nnz, row_indices_d, csr_values_d, csr_indptr_d,
                                  csr_indices_d, B_d, C_d, 0));

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
      CUDA_CHECK(sputnik::CudaSpmm(M, K, N, nnz, row_indices_d, csr_values_d, csr_indptr_d,
                                  csr_indices_d, B_d, C_d, 0));
    }
    gpu_timer.stop();
    kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
  }
  float MFlop_count = (float)nnz / 1e6 * N * 2;
  float gflops = MFlop_count / kernel_dur_msecs;
  printf(
      "[Sputnik] Report: spmm (A(%d x %d) * B(%d x %d))"
      "sparsity "
      "%f (nnz=%d) \n Time %f (ms), Throughput %f (gflops).\n",
      M, K, K, N, (float)nnz / (float)M / (float)K, nnz, kernel_dur_msecs, gflops);

  /// free memory

  if (B_h) free(B_h);
  if (C_h) free(C_h);
  if (C_ref) free(C_ref);
  if (csr_values_h) free(csr_values_h);
  if (B_d) CUDA_CHECK(cudaFree(B_d));
  if (C_d) CUDA_CHECK(cudaFree(C_d));
  if (csr_values_d) CUDA_CHECK(cudaFree(csr_values_d));
  if (row_indices_d) CUDA_CHECK(cudaFree(row_indices_d));
  if (csr_indptr_d) CUDA_CHECK(cudaFree(csr_indptr_d));
  if (csr_indices_d) CUDA_CHECK(cudaFree(csr_indices_d));
  return 0;
}