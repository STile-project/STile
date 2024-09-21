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
#include <iostream>

#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    if (e == cudaSuccess || e == cudaErrorCudartUnloading)   \
        std::cerr << "CUDA: " << cudaGetErrorString(e);      \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...) \
  {                                                                \
    (kernel) <<< (nblks), (nthrs), (shmem), (stream) >>>           \
      (__VA_ARGS__);                                               \
    cudaError_t e = cudaGetLastError();                            \
    if (e != cudaSuccess) {                                        \
        throw std::runtime_error("CUDA kernel launch error: " + std::string(cudaGetErrorString(e)));           \
    }                                                              \
  }

namespace nvbench::detail
{

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
