#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include "centerpoint/base.hpp"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"
#include "NvOnnxConfig.h"

#include <fstream>
#include <regex>
#include <experimental/filesystem>
#include <unordered_map>

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      std::cerr << "[NVINFER LOG]: " << msg << std::endl;
    }
  }
};

class Network {
public:
  Network(const std::string &model_path, const YAML::Node config,
          nvtype::half *pillar_features, unsigned int *voxel_idxs, unsigned int *num_of_pillars)
    : config_(config), pillar_features_(pillar_features), voxel_idxs_(voxel_idxs), num_of_pillars_(num_of_pillars) {
    std::cout << "== Network Initialization ==" << std::endl;
    runtime_ = nvinfer1::createInferRuntime(logger_);

    if (!std::experimental::filesystem::exists(model_path)) {
      buildEngine(model_path);
    }
    else {
      loadEngine(model_path);
    }

    context_ = engine_->createExecutionContext();

    int num_bindings = engine_->getNbBindings();
    for (int i = 0; i < num_bindings; i++) {
      const char *name = engine_->getBindingName(i);
      buffer_names_.push_back(name);
    }
    init();
  }

  ~Network() {
    if (context_ != nullptr) { context_->destroy(); }
    if (engine_ != nullptr) { engine_->destroy(); }
    if (runtime_ != nullptr) { runtime_->destroy(); }

    checkRuntime(cudaFree(center_output_));
    checkRuntime(cudaFree(center_z_output_));
    checkRuntime(cudaFree(dim_output_));
    checkRuntime(cudaFree(rot_output_));
    checkRuntime(cudaFree(score_output_));
    checkRuntime(cudaFree(label_output_));
    checkRuntime(cudaFree(iou_output_));

  }

private:
  void buildEngine(const std::string& model_path) {
    std::cout << "Model file not found: " << model_path << std::endl;
    std::cout << "Start building engine..." << std::endl;

    std::string onnx_path = std::regex_replace(model_path, std::regex(".trt"), ".onnx");
    std::cout << "onnx_path: " << onnx_path << std::endl;

    auto builder = nvinfer1::createInferBuilder(logger_);

    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    auto parser = nvonnxparser::createParser(*network, logger_);
    parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    onnxInfo(network);

    auto config = builder->createBuilderConfig();
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMaxWorkspaceSize(size_t(1) << 30);
    builder->setMaxBatchSize(1);

    engine_ = builder->buildEngineWithConfig(*network, *config);
    if (engine_ == nullptr) {
      std::cerr << "Failed to build engine" << std::endl;
      exit(1);
    }

    // save the engine
    auto serialized_engine = engine_->serialize();
    std::fstream file(model_path, std::ifstream::out);
    if (!file.is_open()) {
      std::cerr << "Failed to save engine file" << std::endl;
      exit(1);
    }

    file.write((char*)serialized_engine->data(), serialized_engine->size());
    file.close();

    serialized_engine->destroy();
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

    std::cout << "== Engine Build Complete ==" << std::endl;
  }

  void loadEngine(const std::string& model_path) {
    std::cout << "model_path: " << model_path << std::endl;

    std::fstream file(model_path, std::ifstream::in);
    if (!file.is_open()) {
      std::cerr << "Failed to open engine file" << std::endl;
      exit(1);
    }

    char* data;
    unsigned int length;

    file.seekg(0, file.end);
    length = file.tellg();
    file.seekg(0, file.beg);
    data = new char[length];
    file.read(data, length);

    engine_ = runtime_->deserializeCudaEngine(data, length, nullptr);
    if (engine_ == nullptr) {
      std::cerr << "Failed to deserialize the engine from file: " << model_path << std::endl;
      exit(1);
    }

    file.close();
    delete[] data;

    std::cout << "== Engine Load Complete ==" << std::endl;
  }

  void onnxInfo(nvinfer1::INetworkDefinition *network) {
    std::cout << "== ONNX Network Inputs ==\n";
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

    std::cout << "== ONNX Network Outputs ==\n";
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

  void init() {
    unsigned int feature_size_x = config_["feature_size"]["x"].as<unsigned int>();
    unsigned int feature_size_y = config_["feature_size"]["y"].as<unsigned int>();
    unsigned int feature_size = feature_size_x * feature_size_y;

    center_size_ = feature_size * 2 * sizeof(float);
    center_z_size_ = feature_size * sizeof(float);
    dim_size_ = feature_size * 3 * sizeof(float);
    rot_size_ = feature_size * 2 * sizeof(float);
    score_size_ = feature_size * sizeof(float);
    label_size_ = feature_size * sizeof(int32_t);
    iou_size_ = feature_size * sizeof(float);

    checkRuntime(cudaMalloc(&center_output_, center_size_));
    checkRuntime(cudaMalloc(&center_z_output_, center_z_size_));
    checkRuntime(cudaMalloc(&dim_output_, dim_size_));
    checkRuntime(cudaMalloc(&rot_output_, rot_size_));
    checkRuntime(cudaMalloc(&score_output_, score_size_));
    checkRuntime(cudaMalloc(&label_output_, label_size_));
    checkRuntime(cudaMalloc(&iou_output_, iou_size_));

    std::unordered_map<std::string, void*> map {
      {"voxels", pillar_features_},
      {"voxel_idxs", voxel_idxs_},
      {"voxel_num", num_of_pillars_},
      {"center", center_output_},
      {"center_z", center_z_output_},
      {"dim", dim_output_},
      {"rot", rot_output_},
      {"score", score_output_},
      {"label", label_output_},
      {"iou", iou_output_}
    };
    buffers_ = std::unique_ptr<std::unordered_map<std::string, void*>>(
        new std::unordered_map<std::string, void*>(map));
    // buffers_ = std::make_unique<std::unordered_map<std::string, void*>>(map);
  }
public:
  bool forward (cudaStream_t stream) {
    std::vector<std::string> order(buffer_names_);
    std::vector<void*> buffers;
    buffers.resize(order.size());
    std::transform(order.begin(), order.end(), buffers.begin(),
                   [this](const std::string& name) { return buffers_->at(name); });
    bool status = context_->enqueueV2(buffers.data(), stream, nullptr);

    return status;
  }
private:
  // TensorRT
  Logger logger_;
  nvinfer1::IExecutionContext *context_ = nullptr;
  nvinfer1::IRuntime *runtime_          = nullptr;
  nvinfer1::ICudaEngine *engine_        = nullptr;
  std::vector<std::string> buffer_names_;
  std::unique_ptr<std::unordered_map<std::string, void*>> buffers_;

  // Input
  nvtype::half* pillar_features_ = nullptr;
  unsigned int* voxel_idxs_ = nullptr;
  unsigned int* num_of_pillars_ = nullptr;

  // Output
  float* center_output_ = nullptr;
  float* center_z_output_ = nullptr;
  float* dim_output_ = nullptr;
  float* rot_output_ = nullptr;
  float* score_output_ = nullptr;
  int32_t* label_output_ = nullptr;
  float* iou_output_ = nullptr;

  unsigned int center_size_;
  unsigned int center_z_size_;
  unsigned int dim_size_;
  unsigned int rot_size_;
  unsigned int score_size_;
  unsigned int label_size_;
  unsigned int iou_size_;

  // config
  YAML::Node config_;
};


#endif // _NETWORK_HPP_