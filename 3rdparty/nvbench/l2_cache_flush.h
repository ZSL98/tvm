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
 * \file l2_cache_flush.h
 * \brief Functions to flush L2 cache using CUDA's API, adopted from nvbench.
 */
#ifndef L2_CACHE_FLUSH_H_
#define L2_CACHE_FLUSH_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <dmlc/logging.h>
#include <iostream>

namespace tvm {
namespace runtime {

const int BLOCK = 1024;
const int BENCH_SIZE = (1lu << 26); 
const int THREAD_STRIDE = (1lu << 16);
const int BLOCK_STRIDE = (1lu << 8);
const int BENCH_ITER = 16;

#define CUDA_CALL(func)                                       \
  {                                                           \
    cudaError_t e = (func);                                   \
    ICHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                 \
  }

class L2Flush {
 public:
  L2Flush() : initialized_(false), l2_size_(0), l2_buffer_(nullptr) {}

  ~L2Flush() {
    if (l2_size_ > 0) {
      CUDA_CALL(cudaFree(l2_buffer_));
    }
  }

  void Flush(CUcontext context, cudaStream_t stream) {
    cuCtxSetCurrent(context);
    if (!initialized_) {
      // initialize l2_buffer_ and l2_size_
      initialized_ = true;
      int device_id;
      CUDA_CALL(cudaGetDevice(&device_id));
      CUDA_CALL(cudaDeviceGetAttribute(&l2_size_, cudaDevAttrL2CacheSize, device_id));
      if (l2_size_ > 0) {
        CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&l2_buffer_), l2_size_));
      }
    }
    if (l2_size_ > 0) {
      std::cout << "l2 cache flush" << std::endl;
      CUDA_CALL(cudaMemsetAsync(l2_buffer_, 0, l2_size_, stream));
    }
  }

  void resident_kernel(CUcontext context, cudaStream_t stream) {
    cuCtxSetCurrent(context);

    size_t size_in_byte = (1lu << 30) * 4; //(4GB)
    char *ws;
    cudaMalloc(&ws, size_in_byte);
    cudaMemset(ws, 0, size_in_byte);

    char *rs;
    cudaMalloc(&rs, size_in_byte);
    cudaMemset(rs, 0, size_in_byte);

    // int* worker_num;
    // cudaMallocManaged(&worker_num, sizeof(int));

    const int L2_FLUSH_SIZE = (1 << 20) * 128;
    int *l2_flush;
    cudaMalloc(&l2_flush, L2_FLUSH_SIZE);
    cudaMemset(l2_flush, 0, L2_FLUSH_SIZE);

    int* workernum_host = new int;
    int* workernum_dev;
    *workernum_host = 0;
    cudaMalloc(&workernum_dev, sizeof(int));
    cudaMemcpy(workernum_dev, workernum_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t strm;
    CUDA_CALL(cudaStreamCreate(&strm));
    CUmodule module;
    CUfunction func;
    // char *module_file = (char*) "/root/compsche/spatial_codegen/resident_kernel/read_write_kernel.cubin";
    // char *kernel_name = (char*) "_Z21read_write_kernel_PTBPKvPi";
    char *module_file = (char*) "/root/compsche/evaluator/interference_evaluate/kernel_test/resident_kernels/resident_kernel.cubin";
    char *kernel_name = (char*) "resident_kernel_1_T_PTB";

    cuModuleLoad(&module, module_file);
    cuModuleGetFunction(&func, module, kernel_name);

    void *param[] = { (void*)&ws, (void*)&rs, (void*)&workernum_dev };
    // CUcontext ctx;
    // cuCtxGetCurrent(&ctx);
    // std::cout << "current ctx: " << ctx << std::endl;
    cuLaunchKernel(func, 2*108, 1, 1, BLOCK, 1, 1, 0, strm, param, NULL);
    // cudaStreamSynchronize(strm);
    // std::cout << "Resident kernel worker num: " << *worker_num << std::endl;
    cudaFreeAsync(ws, strm);
    cudaFreeAsync(rs, strm);
    cudaFreeAsync(workernum_dev, strm);
    cudaFreeAsync(l2_flush, strm);
    // cuModuleUnload(module);
  }

  static L2Flush* ThreadLocal();

 private:
  bool initialized_ = false;
  int l2_size_;
  int* l2_buffer_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // L2_CACHE_FLUSH_H_
