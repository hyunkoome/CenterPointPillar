#include "pycenterpoint/voxelization.hpp"

int3 VoxelizationParameter::compute_grid_size(const float3 &max_range, const float3 &min_range,
                                              const float3 &voxel_size)
{
  int3 size;
  size.x = static_cast<int>(std::round((max_range.x - min_range.x) / voxel_size.x));
  size.y = static_cast<int>(std::round((max_range.y - min_range.y) / voxel_size.y));
  size.z = static_cast<int>(std::round((max_range.z - min_range.z) / voxel_size.z));

  return size;
}

Voxelization::Voxelization(const YAML::Node& config) { memoryInit(config); }

Voxelization::~Voxelization() {
  if (dev_voxel_features_) checkRuntime(cudaFree(dev_voxel_features_));
  if (dev_voxel_num_) checkRuntime(cudaFree(dev_voxel_num_));
  if (dev_voxel_idxs_) checkRuntime(cudaFree(dev_voxel_idxs_));

  if (dev_features_input_) checkRuntime(cudaFree(dev_features_input_));
  if (dev_params_input_) checkRuntime(cudaFree(dev_params_input_));

  if (dev_mask_) checkRuntime(cudaFree(dev_mask_));
  if (dev_voxels_) checkRuntime(cudaFree(dev_voxels_));
}

void Voxelization::memoryInit(const YAML::Node& config)
{
  setParams(config);
  unsigned int mask_size = param_.grid_size.z * param_.grid_size.y
              * param_.grid_size.x * sizeof(unsigned int);
  unsigned int voxels_size = param_.grid_size.z * param_.grid_size.y * param_.grid_size.x
              * param_.max_points_per_voxel * 4 * sizeof(float);
  unsigned int voxel_features_size = param_.max_voxels * param_.max_points_per_voxel * 4 * sizeof(float);

  unsigned int voxel_num_size = param_.max_voxels * sizeof(unsigned int);
  unsigned int voxel_idxs_size = param_.max_voxels * 4 * sizeof(unsigned int);
  unsigned int features_input_size = param_.max_voxels * param_.max_points_per_voxel * 10 * sizeof(float);

  checkRuntime(cudaMalloc((void **)&dev_voxel_features_, voxel_features_size));
  checkRuntime(cudaMalloc((void **)&dev_voxel_num_, voxel_num_size));

  checkRuntime(cudaMalloc((void **)&dev_features_input_, features_input_size));
  checkRuntime(cudaMalloc((void **)&dev_voxel_idxs_, voxel_idxs_size));
  checkRuntime(cudaMalloc((void **)&dev_params_input_, sizeof(unsigned int)));

  checkRuntime(cudaMalloc((void **)&dev_mask_, mask_size));
  checkRuntime(cudaMalloc((void **)&dev_voxels_, voxels_size));

  checkRuntime(cudaMemset(dev_voxel_features_, 0, voxel_features_size));
  checkRuntime(cudaMemset(dev_voxel_num_, 0, voxel_num_size));

  checkRuntime(cudaMemset(dev_mask_, 0, mask_size));
  checkRuntime(cudaMemset(dev_voxels_, 0, voxels_size));

  checkRuntime(cudaMemset(dev_features_input_, 0, features_input_size));
  checkRuntime(cudaMemset(dev_voxel_idxs_, 0, voxel_idxs_size));
}

void Voxelization::forward(const float *_points, int num_points, void *stream)
{
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  checkRuntime(cudaMemsetAsync(dev_params_input_, 0, sizeof(unsigned int), _stream));

  checkRuntime(generateVoxels_random_launch(_points, num_points,
              param_.min_range.x, param_.max_range.x,
              param_.min_range.y, param_.max_range.y,
              param_.min_range.z, param_.max_range.z,
              param_.voxel_size.x, param_.voxel_size.y, param_.voxel_size.z,
              param_.grid_size.y, param_.grid_size.x,
              dev_mask_, dev_voxels_, _stream));

  checkRuntime(generateBaseFeatures_launch(dev_mask_, dev_voxels_,
              param_.grid_size.y, param_.grid_size.x, param_.max_voxels,
              dev_params_input_,
              dev_voxel_features_,
              dev_voxel_num_,
              dev_voxel_idxs_, _stream));

  checkRuntime(generateFeatures_launch(dev_voxel_features_,
              dev_voxel_num_,
              dev_voxel_idxs_,
              dev_params_input_, param_.max_voxels,
              param_.voxel_size.x, param_.voxel_size.y, param_.voxel_size.z,
              param_.min_range.x, param_.min_range.y, param_.min_range.z,
              dev_features_input_, _stream));
}

void Voxelization::setParams(const YAML::Node& config)
{
  std::cout << "=== Voxelization Parameters ===" << std::endl << config << std::endl;
  param_.min_range = make_float3(config["min_range"]["x"].as<float>(),
                                 config["min_range"]["y"].as<float>(),
                                 config["min_range"]["z"].as<float>());
  param_.max_range = make_float3(config["max_range"]["x"].as<float>(),
                                 config["max_range"]["y"].as<float>(),
                                 config["max_range"]["z"].as<float>());
  param_.voxel_size = make_float3(config["voxel_size"]["x"].as<float>(),
                                  config["voxel_size"]["y"].as<float>(),
                                  config["voxel_size"]["z"].as<float>());
  param_.grid_size = param_.compute_grid_size(param_.max_range, param_.min_range, param_.voxel_size);
  param_.max_voxels = config["max_voxels"].as<int>();
  param_.max_points_per_voxel = config["max_points_per_voxel"].as<int>();
  param_.num_feature = config["num_feature"].as<int>();
  param_.num_voxel_feature = config["num_voxel_feature"].as<int>();
  std::cout << "grid_size: "
            << param_.grid_size.x << " "
            << param_.grid_size.y << " "
            << param_.grid_size.z << std::endl;
  std::cout << "================================" << std::endl;
}