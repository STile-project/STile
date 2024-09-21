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

#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/sddmm/cuda_sddmm.h"


// Fill a host array with random numbers.
void fill_random(float array[], int size) {
  for (int i = 0; i < size; i++) {
    array[i] = (float)(std::rand() % 3) / 10;
  }
}

// Fill a host array with all 0
template <typename DType>
void fill_zero(DType array[], int size) {
  memset(array, 0x0, sizeof(array[0]) * size);
}

// Compute sddmm correct numbers. All arrays are host memory locations.
template <typename Index, typename DType>
void sddmm_reference_host(
    int M,    // number of S-rows, S is the sparse matrix
    int N,    // number of S_cols
    int K,    // number of A columns
    int nnz,  // number of nonzeros in S

    const Index *csr_indptr, const Index *csr_indices,
    const DType *A,           // assume row-major
    const DType *B,           // assume row-major, assume transposed
    DType *C_ref)             // assume row-major
{
  for (int i = 0; i < M; i++) {
    Index lb = csr_indptr[i];
    Index hb = csr_indptr[i + 1];
    Index offset1, offset2;
    DType acc = 0;
    for (int ptr = lb; ptr < hb; ptr++) {
      offset1 = i * K;
      offset2 = csr_indices[ptr] * K;
      for (int k = 0; k < K; k++) {
        acc += A[k + offset1] * B[k + offset2];
      }
      C_ref[ptr] = acc;
      acc = 0;
    }
  }
}

// Compare two nnz matrices
template <typename DType>
bool check_result(int nnz, DType *C, DType *C_ref) {
  bool passed = true;
  for (int64_t i = 0; i < nnz; i++) {
    DType c = C[i];
    DType c_ref = C_ref[i];
    if (fabs(c - c_ref) > 1e-2 * fabs(c_ref)) {
      printf("Wrong result: eid = %ld, result = %lf, reference = %lf.\n", i, c, c_ref);
      passed = false;
    }
  }
  return passed;
}

int main(int argc, char *argv[]) {
  // check command-line argument

  if (argc < 2) {
    printf(
        "Require command-line argument: name of the sparse matrix file in "
        ".mtx format.\n");
    return EXIT_FAILURE;
  }

  //
  // Load sparse matrix
  //

  int M;                               // number of S-rows
  int N;                               // number of S-columns
  int nnz;                             // number of non-zeros in S
  std::vector<int> csr_indptr_buffer;  // buffer for indptr array in CSR format
  std::vector<int> row_indices_buffer;
  std::vector<int> csr_indices_buffer;  // buffer for indices (column-ids) array in CSR format
  // load sparse matrix from mtx file
  // read_mtx_file(argv[1], M, N, nnz, csr_indptr_buffer, csr_indices_buffer);
  read_npz_file(argv[1], M, N, nnz, csr_indptr_buffer, csr_indices_buffer);
  for (int i = 0; i < M; ++i) {
    row_indices_buffer.push_back(i);
  }

  printf(
      "Finish reading matrix %d rows, %d columns, %d nnz. \nIgnore original "
      "values and use randomly generated values.\n",
      M, N, nnz);

  // Create GPU arrays
  int K = 128;  // number of A-columns
  if (argc > 2) {
    K = atoi(argv[2]);
  }
  assert(K > 0 && "second command-line argument is number of B columns, should be >0.\n");

  float *A_h = NULL, *B_h = NULL, *C_h = NULL, *C_ref = NULL;
  float *A_d = NULL, *B_d = NULL, *C_d = NULL;
  int *csr_indptr_d = NULL, *csr_indices_d = NULL, *row_indices_d = NULL;
  A_h = (float *)malloc(sizeof(float) * M * K);
  B_h = (float *)malloc(sizeof(float) * N * K);
  C_h = (float *)malloc(sizeof(float) * nnz);
  C_ref = (float *)malloc(sizeof(float) * nnz);
  if (!A_h || !B_h || !C_h || !C_ref) {
    printf("Host allocation failed.\n");
    return EXIT_FAILURE;
  }
  fill_random(A_h, M * K);
  fill_random(B_h, N * K);

  cudaDeviceReset();
  cudaSetDevice(0);
  // allocate device memory
  CUDA_CHECK(cudaMalloc((void **)&A_d, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc((void **)&B_d, sizeof(float) * N * K));
  CUDA_CHECK(cudaMalloc((void **)&C_d, sizeof(float) * nnz));
  CUDA_CHECK(cudaMalloc((void **)&csr_indptr_d, sizeof(int) * (M + 1)));
  CUDA_CHECK(cudaMalloc((void **)&row_indices_d, sizeof(int) * M));
  CUDA_CHECK(cudaMalloc((void **)&csr_indices_d, sizeof(int) * nnz));

  CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * N * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(C_d, 0x0, sizeof(float) * nnz));
  CUDA_CHECK(cudaMemcpy(csr_indptr_d, csr_indptr_buffer.data(), sizeof(int) * (M + 1),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(row_indices_d, row_indices_buffer.data(), sizeof(int) * M,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(csr_indices_d, csr_indices_buffer.data(), sizeof(int) * nnz,
                        cudaMemcpyHostToDevice));

  // check result
  CUDA_CALL(sputnik::CudaSddmm(M, K, N, nnz, row_indices_d, csr_indptr_d, csr_indices_d, A_d, B_d,
                               C_d, 0));
  CUDA_CHECK(cudaMemcpy(C_h, C_d, nnz * sizeof(float), cudaMemcpyDeviceToHost));
  sddmm_reference_host<int, float>(M, N, K, nnz, csr_indptr_buffer.data(),
                                   csr_indices_buffer.data(), A_h, B_h, C_ref);
  bool correct = check_result<float>(nnz, C_h, C_ref);

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
      CUDA_CALL(sputnik::CudaSddmm(M, K, N, nnz, row_indices_d, csr_indptr_d, csr_indices_d, A_d, B_d,
                                  C_d, 0));
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
      CUDA_CALL(sputnik::CudaSddmm(M, K, N, nnz, row_indices_d, csr_indptr_d, csr_indices_d, A_d, B_d,
                                  C_d, 0));
    }
    gpu_timer.stop();
    kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
  }
  float MFlop_count = (float)nnz / 1e6 * K * 2;
  float gflops = MFlop_count / kernel_dur_msecs;
  printf(
      "[Sputnik] Report: sddmm (A(%d x %d) * B^T(%d x %d)) odot S(%d x %d) "
      "sparsity "
      "%f (nnz=%d) \n Time %f (ms), Throughput %f (gflops).\n",
      M, K, N, K, M, N, (float)nnz / (float)M / (float)N, nnz, kernel_dur_msecs, gflops);

  /// free memory

  if (A_h) free(A_h);
  if (B_h) free(B_h);
  if (C_h) free(C_h);
  if (C_ref) free(C_ref);
  if (A_d) CUDA_CHECK(cudaFree(A_d));
  if (B_d) CUDA_CHECK(cudaFree(B_d));
  if (C_d) CUDA_CHECK(cudaFree(C_d));
  if (row_indices_d) CUDA_CHECK(cudaFree(row_indices_d));
  if (csr_indptr_d) CUDA_CHECK(cudaFree(csr_indptr_d));
  if (csr_indices_d) CUDA_CHECK(cudaFree(csr_indices_d));

  return 0;
}