#ifndef _VOXELIZATION_CUH_
#define _VOXELIZATION_CUH_

#include "centerpoint/base.hpp"

static __global__ void generateVoxels_random_kernel(const float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        unsigned int *mask, float *voxels);
cudaError_t generateVoxels_random_launch(const float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        unsigned int *mask, float *voxels,
        cudaStream_t stream);
static __global__ void generateBaseFeatures_kernel(unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs);
cudaError_t generateBaseFeatures_launch(unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs,
        cudaStream_t stream);
static __global__ void generateFeatures_kernel(float* voxel_features,
    unsigned int* voxel_num, unsigned int* voxel_idxs, unsigned int *params,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    float* features);
cudaError_t generateFeatures_launch(float* voxel_features,
    unsigned int * voxel_num,
    unsigned int* voxel_idxs,
    unsigned int *params, unsigned int max_voxels,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    float* features,
    cudaStream_t stream);

struct VoxelizationParameter {
    float3 min_range;
    float3 max_range;
    float3 voxel_size;
    int3 grid_size;
    int max_voxels;
    int max_points_per_voxel;
    int max_points;
    int num_feature;
    int num_voxel_feature;

    static int3 compute_grid_size(const float3& max_range, const float3& min_range,
                                        const float3& voxel_size);
};

class Voxelization {
public:
  Voxelization(const YAML::Node& config) {
    init(config);
  }
  ~Voxelization() {
      if (voxel_features_) checkRuntime(cudaFree(voxel_features_));
      if (voxel_num_) checkRuntime(cudaFree(voxel_num_));
      if (voxel_idxs_) checkRuntime(cudaFree(voxel_idxs_));

      if (features_input_) checkRuntime(cudaFree(features_input_));
      if (params_input_) checkRuntime(cudaFree(params_input_));

      if (mask_) checkRuntime(cudaFree(mask_));
      if (voxels_) checkRuntime(cudaFree(voxels_));
      // if (voxelsList_) checkRuntime(cudaFree(voxelsList_));
  }

  const float *features() { return features_input_; }

  const unsigned int *coords() { return voxel_idxs_; }

  const unsigned int *nums() { return params_input_; }

  void init(const YAML::Node& config) {
    setParams(config);
    unsigned int mask_size = param_.grid_size.z * param_.grid_size.y
                * param_.grid_size.x * sizeof(unsigned int);
    unsigned int voxels_size = param_.grid_size.z * param_.grid_size.y * param_.grid_size.x
                * param_.max_points_per_voxel * 4 * sizeof(float);
    unsigned int voxel_features_size = param_.max_voxels * param_.max_points_per_voxel * 4 * sizeof(float);

    unsigned int voxel_num_size = param_.max_voxels * sizeof(unsigned int);
    unsigned int voxel_idxs_size = param_.max_voxels * 4 * sizeof(unsigned int);
    unsigned int features_input_size = param_.max_voxels * param_.max_points_per_voxel * 10 * sizeof(float);

    checkRuntime(cudaMalloc((void **)&voxel_features_, voxel_features_size));
    checkRuntime(cudaMalloc((void **)&voxel_num_, voxel_num_size));

    checkRuntime(cudaMalloc((void **)&features_input_, features_input_size));
    checkRuntime(cudaMalloc((void **)&voxel_idxs_, voxel_idxs_size));
    checkRuntime(cudaMalloc((void **)&params_input_, sizeof(unsigned int)));

    checkRuntime(cudaMalloc((void **)&mask_, mask_size));
    checkRuntime(cudaMalloc((void **)&voxels_, voxels_size));

    checkRuntime(cudaMemset(voxel_features_, 0, voxel_features_size));
    checkRuntime(cudaMemset(voxel_num_, 0, voxel_num_size));

    checkRuntime(cudaMemset(mask_, 0, mask_size));
    checkRuntime(cudaMemset(voxels_, 0, voxels_size));

    checkRuntime(cudaMemset(features_input_, 0, features_input_size));
    checkRuntime(cudaMemset(voxel_idxs_, 0, voxel_idxs_size));
  }

  void forward(const float *_points, int num_points, void *stream) {
    cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

    checkRuntime(cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), _stream));

    checkRuntime(generateVoxels_random_launch(_points, num_points,
                param_.min_range.x, param_.max_range.x,
                param_.min_range.y, param_.max_range.y,
                param_.min_range.z, param_.max_range.z,
                param_.voxel_size.x, param_.voxel_size.y, param_.voxel_size.z,
                param_.grid_size.y, param_.grid_size.x,
                mask_, voxels_, _stream));

    checkRuntime(generateBaseFeatures_launch(mask_, voxels_,
                param_.grid_size.y, param_.grid_size.x,
                params_input_,
                voxel_features_,
                voxel_num_,
                voxel_idxs_, _stream));

    checkRuntime(generateFeatures_launch(voxel_features_,
                voxel_num_,
                voxel_idxs_,
                params_input_, param_.max_voxels,
                param_.voxel_size.x, param_.voxel_size.y, param_.voxel_size.z,
                param_.min_range.x, param_.min_range.y, param_.min_range.z,
                features_input_, _stream));
}
private:
  void setParams(const YAML::Node& config) {
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


public:
  VoxelizationParameter param_;
private:
  unsigned int *mask_ = nullptr;
  float *voxels_ = nullptr;
  // int *voxelsList_ = nullptr;
  float *voxel_features_ = nullptr;
  unsigned int *voxel_num_ = nullptr;

  float *features_input_ = nullptr;
  unsigned int *voxel_idxs_ = nullptr;
  unsigned int *params_input_ = nullptr;

};

#endif