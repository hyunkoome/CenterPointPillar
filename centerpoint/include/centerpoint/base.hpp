#ifndef _BASE_HPP_
#define _BASE_HPP_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <yaml-cpp/yaml.h>

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)
static inline bool check_runtime(cudaError_t e, const char *call, int line, const char *file) {
  if (e != cudaSuccess) {
    fprintf(stderr,
            "CUDA Runtime error %s # %s, code = %s [ %d ] in file "
            "%s:%d\n",
            call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
    abort();
    return false;
  }
  return true;
}

struct Box{
  float x;
  float y;
  float z;
  float w;
  float l;
  float h;
  float rt;
  int id;
  float score;
};

#endif // _BASE_HPP_
