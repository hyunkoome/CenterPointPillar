#include "pycenterpoint/centerpoint.hpp"

CenterPoint::CenterPoint(const std::string& config_path, const std::string& model_path)
{
  config_ = YAML::LoadFile(config_path);
  score_threshold_ = config_["centerpoint"]["score_threshold"].as<float>();
  std::string sub_topic_name = config_["centerpoint"]["sub"].as<std::string>();
  std::string pub_topic_name = config_["centerpoint"]["pub"].as<std::string>();

  checkRuntime(cudaStreamCreate(&stream_));
  memoryInit(model_path);
}

CenterPoint::~CenterPoint()
{
  checkRuntime(cudaStreamDestroy(stream_));
  checkRuntime(cudaFree(dev_input_points_));
}

void CenterPoint::memoryInit(const std::string& model_path)
{
  size_t max_points = config_["centerpoint"]["max_points"].as<size_t>();
  size_t max_points_feature = max_points * config_["voxelization"]["num_feature"].as<size_t>();
  points_.reserve(max_points_feature);
  size_t bytes_points_capacity = max_points_feature * sizeof(float);
  checkRuntime(cudaMalloc(&dev_input_points_, bytes_points_capacity));
  checkRuntime(cudaDeviceSynchronize());

  voxelization_ = std::make_shared<Voxelization>(config_["voxelization"]);
  checkRuntime(cudaDeviceSynchronize());

  network_ = std::make_shared<Network>(model_path);
  checkRuntime(cudaDeviceSynchronize());

  postprocess_ = std::make_shared<PostProcess>(config_["postprocess"]);
  checkRuntime(cudaDeviceSynchronize());
}

std::vector<Box> CenterPoint::forward(PyArray<float> np_array)
{
  size_t num_points = np_array.request().shape[0];
  size_t bytes_points = num_points * voxelization_->param().num_feature * sizeof(float);

  input_points_ = static_cast<float*>(np_array.request().ptr);
  checkRuntime(cudaMemcpyAsync(dev_input_points_, input_points_, bytes_points, cudaMemcpyHostToDevice, stream_));

  voxelization_->forward(dev_input_points_, num_points, stream_);

  network_->forward(voxelization_->features(), voxelization_->coords(), voxelization_->nums(), stream_);

  int box_num = postprocess_->forward(network_->center(), network_->center_z(), network_->dim(), network_->rot(),
                                      network_->score(), network_->label(), network_->iou(), stream_);

  checkRuntime(cudaStreamSynchronize(stream_));
  return postprocess_->getBoxes();
}