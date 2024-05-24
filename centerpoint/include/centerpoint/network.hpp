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
  Network(const std::string &model_path, const YAML::Node& config) {
    std::cout << "==== Network Initialization ====" << std::endl;

    if (!std::experimental::filesystem::exists(model_path)) {
      buildEngine(model_path);
    }
    loadEngine(model_path);

    init(config);
    std::cout << "================================" << std::endl;
  }
  ~Network() {
    destroy();

    checkRuntime(cudaFree(center_output_));
    checkRuntime(cudaFree(center_z_output_));
    checkRuntime(cudaFree(dim_output_));
    checkRuntime(cudaFree(rot_output_));
    checkRuntime(cudaFree(score_output_));
    checkRuntime(cudaFree(label_output_));
    checkRuntime(cudaFree(iou_output_));
  }

  void forward(const float* voxels, const unsigned int* voxel_idxs, const unsigned int* params, void* stream) {
    cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
    engine_infer({voxels, voxel_idxs, params,
                  center_output_, center_z_output_, rot_output_, dim_output_, score_output_, label_output_, iou_output_},
                  _stream);
  }


private:
  void init(const YAML::Node& config) {
    unsigned int feature_x_size = config["feature_size"]["x"].as<unsigned int>();
    unsigned int feature_y_size = config["feature_size"]["y"].as<unsigned int>();
    unsigned int feature_size = feature_x_size * feature_y_size;

    unsigned int reg_channel = config["channel"]["center"].as<unsigned int>();
    unsigned int height_channel = config["channel"]["center_z"].as<unsigned int>();
    unsigned int rot_channel = config["channel"]["rot"].as<unsigned int>();
    unsigned int dim_channel = config["channel"]["dim"].as<unsigned int>();

    center_size_ = feature_size * reg_channel * sizeof(float);
    center_z_size_ = feature_size * height_channel * sizeof(float);
    dim_size_ = feature_size * dim_channel * sizeof(float);
    rot_size_ = feature_size * rot_channel * sizeof(float);
    score_size_ = feature_size * sizeof(float);
    label_size_ = feature_size * sizeof(int32_t);
    iou_size_ = feature_size * sizeof(float);

    checkRuntime(cudaMalloc((void**)&center_output_, center_size_));
    checkRuntime(cudaMalloc((void**)&center_z_output_, center_z_size_));
    checkRuntime(cudaMalloc((void**)&dim_output_, dim_size_));
    checkRuntime(cudaMalloc((void**)&rot_output_, rot_size_));
    checkRuntime(cudaMalloc((void**)&score_output_, score_size_));
    checkRuntime(cudaMalloc((void**)&label_output_, label_size_));
    checkRuntime(cudaMalloc((void**)&iou_output_, iou_size_));
  }

  void engine_infer(const std::vector<const void *> &bindings, void *stream, void *input_consum_event = nullptr) {
    context_->enqueueV2((void **)bindings.data(), (cudaStream_t)stream, (cudaEvent_t *)input_consum_event);
  }

  void buildEngine(const std::string& model_path) {
    std::cout << "Model file not found: " << model_path << std::endl;
    std::cout << "Start building engine..." << std::endl;

    std::string onnx_path = std::regex_replace(model_path, std::regex(".trt"), ".onnx");
    std::cout << "onnx_path: " << onnx_path << std::endl;

    auto builder = nvinfer1::createInferBuilder(nvlogger_);

    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    auto parser = nvonnxparser::createParser(*network, nvlogger_);
    parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE));
    onnxInfo(network);

    auto config = builder->createBuilderConfig();
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1UL << 30);
    builder->setMaxBatchSize(1);

    // save the engine
    auto serialized_engine = builder->buildSerializedNetwork(*network, *config);
    std::fstream file(model_path, std::ifstream::out);
    if (!file.is_open()) {
      std::cerr << "Failed to save engine file" << std::endl;
      exit(1);
    }

    file.write((char*)serialized_engine->data(), serialized_engine->size());
    file.close();

    destroy_pointer(serialized_engine);
    destroy_pointer(config);
    destroy_pointer(parser);
    destroy_pointer(network);
    destroy_pointer(builder);

    std::cout << "=== Engine Build Complete ===" << std::endl;
  }

  void onnxInfo(nvinfer1::INetworkDefinition *network) {
    std::cout << "===== ONNX Network Inputs ======\n";
    int numInputs = network->getNbInputs();
    for (int i = 0; i < numInputs; i++) {
      nvinfer1::ITensor* input = network->getInput(i);
      nvinfer1::Dims dims = input->getDimensions();
      std::cout << "[" << i << "] " << input->getName() << ": ";
      for (int j = 0; j < dims.nbDims; j++) {
        std::cout << dims.d[j];
        if (j != dims.nbDims - 1) std::cout << " x ";
      }
      std::cout << std::endl;
    }

    std::cout << "===== ONNX Network Outputs =====\n";
    int numOutputs = network->getNbOutputs();
    for (int i = 0; i < numOutputs; i++) {
      nvinfer1::ITensor* output = network->getOutput(i);
      nvinfer1::Dims dims = output->getDimensions();
      std::cout << "[" << i << "] " << output->getName() << ": ";
      for (int j = 0; j < dims.nbDims; j++) {
        std::cout << dims.d[j];
        if (j != dims.nbDims - 1) std::cout << " x ";
      }
      std::cout << std::endl;
    }
    std::cout << "================================" << std::endl;
  }

  void loadEngine(const std::string &file) {
    std::cout << "model_path: " << file << std::endl;

    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "Failed to open engine file" << std::endl;
      exit(1);
    }

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
      in.seekg(0, std::ios::beg);
      data.resize(length);

      in.read((char *)&data[0], length);
    }
    in.close();

    if (data.empty() || data.data() == nullptr || data.size() == 0) {
      printf("An empty file has been loaded. Please confirm your file path: %s\n", file.c_str());
      exit(1);
    }

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(nvlogger_), destroy_pointer<nvinfer1::IRuntime>);
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(data.data(), data.size()),
                                                     destroy_pointer<nvinfer1::ICudaEngine>);
    if (engine_ == nullptr) {
      printf("Failed to deserialize the engine from file: %s\n", file.c_str());
      exit(1);
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(),
                                                            destroy_pointer<nvinfer1::IExecutionContext>);

    std::cout << "== Engine Load Complete ==" << std::endl;
  }

  void destroy() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }


private:
  // TensorRT
  Logger nvlogger_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_        = nullptr;
  std::shared_ptr<nvinfer1::IRuntime> runtime_          = nullptr;

  // Output
  unsigned int center_size_;
  unsigned int center_z_size_;
  unsigned int dim_size_;
  unsigned int rot_size_;
  unsigned int score_size_;
  unsigned int label_size_;
  unsigned int iou_size_;

  float* center_output_ = nullptr;
  float* center_z_output_ = nullptr;
  float* dim_output_ = nullptr;
  float* rot_output_ = nullptr;
  float* score_output_ = nullptr;
  int32_t* label_output_ = nullptr;
  float* iou_output_ = nullptr;
};


#endif // _NETWORK_HPP_