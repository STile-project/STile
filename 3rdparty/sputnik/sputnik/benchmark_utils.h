/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cuda_runtime_api.h>
#include <cnpy.h>
#include <iostream>

namespace nvbench::detail
{

#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    if (e == cudaSuccess || e == cudaErrorCudartUnloading)   \
        std::cerr << "CUDA: " << cudaGetErrorString(e);      \
  }

struct l2flush
{
  __forceinline__ l2flush()
  {
    int dev_id{};
    cudaGetDevice(&dev_id);
    cudaDeviceGetAttribute(&m_l2_size, cudaDevAttrL2CacheSize, dev_id);
    if (m_l2_size > 0)
    {
      void *buffer = m_l2_buffer;
      cudaMalloc(&buffer, m_l2_size);
      m_l2_buffer = reinterpret_cast<int *>(buffer);
    }
  }

  __forceinline__ ~l2flush()
  {
    if (m_l2_buffer)
    {
      cudaFree(m_l2_buffer);
    }
  }

  __forceinline__ void flush()
  {
    if (m_l2_size > 0)
    {
      cudaMemset(m_l2_buffer, 0, m_l2_size);
    }
  }

private:
  int m_l2_size{};
  int *m_l2_buffer{};
};

} // namespace nvbench::detail

struct GpuTimer {
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  nvbench::detail::l2flush l2flush;

  GpuTimer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
  }

  ~GpuTimer() {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
  }

  void start(bool flush_l2) {
    if (flush_l2) {
      l2flush.flush();
    }
    cudaEventRecord(startEvent, 0);
  }

  void stop() {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
  }

  float elapsed_msecs() {
    float elapsed;
    cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
    return elapsed;
  }
};

#define CUDA_CHECK(func)                                                   \
  {                                                                        \
    cudaError_t status = (func);                                           \
    if (status != cudaSuccess) {                                           \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
             cudaGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  }

#define CUSPARSE_CHECK(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

void read_npz_file(const std::string &filename, int &M, int &K, int &NNZ, std::vector<int> &indptr,
                   std::vector<int> &indices) {
  cnpy::npz_t data = cnpy::npz_load(filename);
  cnpy::NpyArray shape = data["shape"];
  int *shape_data = shape.data<int>();
  M = shape_data[0];
  K = shape_data[1];
  NNZ = shape_data[2];
  indptr = std::move(data["indptr"].as_vec<int>());
  indices = std::move(data["indices"].as_vec<int>());
}

