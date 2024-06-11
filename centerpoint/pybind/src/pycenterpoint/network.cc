#include "pycenterpoint/network.hpp"

Network::Network(const std::string &model_path)
{
  std::cout << "==== Network Initialization ====" << std::endl;

  if (!std::experimental::filesystem::exists(model_path)) {
    buildEngine(model_path);
  }
  loadEngine(model_path);

  memoryInit();
  std::cout << "================================" << std::endl;
}

Network::~Network()
{
  destroy();

  checkRuntime(cudaFree(dev_center_output_));
  checkRuntime(cudaFree(dev_center_z_output_));
  checkRuntime(cudaFree(dev_dim_output_));
  checkRuntime(cudaFree(dev_rot_output_));
  checkRuntime(cudaFree(dev_score_output_));
  checkRuntime(cudaFree(dev_label_output_));
  checkRuntime(cudaFree(dev_iou_output_));
}

void Network::forward(const float* voxels, const unsigned int* voxel_idxs, const unsigned int* params, void* stream)
{
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  engineInfer({voxels, voxel_idxs, params,
               dev_center_output_, dev_center_z_output_, dev_rot_output_, dev_dim_output_, dev_score_output_, dev_label_output_, dev_iou_output_},
               _stream);
}

void Network::memoryInit()
{
  unsigned int feature_y_size = engine_->getTensorShape("center").d[2];
  unsigned int feature_x_size = engine_->getTensorShape("center").d[3];
  unsigned int feature_size = feature_x_size * feature_y_size;

  unsigned int center_size = feature_size * 2 * sizeof(float);
  unsigned int center_z_size = feature_size * 1 * sizeof(float);
  unsigned int dim_size = feature_size * 3 * sizeof(float);
  unsigned int rot_size = feature_size * 2 * sizeof(float);
  unsigned int score_size = feature_size * sizeof(float);
  unsigned int label_size = feature_size * sizeof(int32_t);
  unsigned int iou_size = feature_size * sizeof(float);

  checkRuntime(cudaMalloc((void**)&dev_center_output_, center_size));
  checkRuntime(cudaMalloc((void**)&dev_center_z_output_, center_z_size));
  checkRuntime(cudaMalloc((void**)&dev_dim_output_, dim_size));
  checkRuntime(cudaMalloc((void**)&dev_rot_output_, rot_size));
  checkRuntime(cudaMalloc((void**)&dev_score_output_, score_size));
  checkRuntime(cudaMalloc((void**)&dev_label_output_, label_size));
  checkRuntime(cudaMalloc((void**)&dev_iou_output_, iou_size));
}

void Network::engineInfer(const std::vector<const void *> &bindings, void *stream, void *input_consum_event)
{
  context_->enqueueV2((void **)bindings.data(), (cudaStream_t)stream, (cudaEvent_t *)input_consum_event);
}

void Network::buildEngine(const std::string& model_path)
{
  std::cout << "Model file not found: " << model_path << std::endl;
  std::cout << "Start building engine..." << std::endl;

  std::string onnx_path = std::regex_replace(model_path, std::regex(".trt"), ".onnx");
  std::cout << "onnx_path: " << onnx_path << std::endl;

  auto builder = nvinfer1::createInferBuilder(nvlogger_);

  auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = builder->createNetworkV2(explicit_batch);

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

void Network::onnxInfo(nvinfer1::INetworkDefinition *network)
{
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

void Network::loadEngine(const std::string &file)
{
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

void Network::destroy()
{
  context_.reset();
  engine_.reset();
  runtime_.reset();
}