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
