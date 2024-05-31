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

struct Box {
  float val[9];
  float x() const { return val[0]; }
  float y() const { return val[1]; }
  float z() const { return val[2]; }
  float l() const { return val[3]; }
  float w() const { return val[4]; }
  float h() const { return val[5]; }
  float yaw() const { return val[6]; }
  float score() const { return val[7]; }
  int cls() const { return static_cast<int>(val[8]); }
};

#endif // _BASE_HPP_
