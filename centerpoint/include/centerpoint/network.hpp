#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include <memory>
#include <fstream>
#include <vector>
#include <regex>
#include <experimental/filesystem>

#include "centerpoint/base.hpp"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"
#include "NvOnnxConfig.h"

template <typename T>
static void destroy_pointer(T *ptr) {
  if (ptr) delete ptr;
}

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kWARNING) {
      std::cerr << "[NVINFER LOG]: " << msg << std::endl;
    }
  }
};

class Network {
public:
  Network(const std::string &model_path);
  ~Network();
  void forward(const float* voxels, const unsigned int* voxel_idxs, const unsigned int* params, void* stream);
  const float* center() const { return dev_center_output_; }
  const float* center_z() const { return dev_center_z_output_; }
  const float* dim() const { return dev_dim_output_; }
  const float* rot() const { return dev_rot_output_; }
  const float* score() const { return dev_score_output_; }
  const int32_t* label() const { return dev_label_output_; }
  const float* iou() const { return dev_iou_output_; }


private:
  void memoryInit();
  void engineInfer(const std::vector<const void *> &bindings, void *stream, void *input_consum_event = nullptr);
  void buildEngine(const std::string& model_path);
  void onnxInfo(nvinfer1::INetworkDefinition *network);
  void loadEngine(const std::string &file);
  void destroy();

private:
  // TensorRT
  Logger nvlogger_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_        = nullptr;
  std::shared_ptr<nvinfer1::IRuntime> runtime_          = nullptr;

  // Output
  float* dev_center_output_ = nullptr;
  float* dev_center_z_output_ = nullptr;
  float* dev_dim_output_ = nullptr;
  float* dev_rot_output_ = nullptr;
  float* dev_score_output_ = nullptr;
  int32_t* dev_label_output_ = nullptr;
  float* dev_iou_output_ = nullptr;
};

#endif // _NETWORK_HPP_