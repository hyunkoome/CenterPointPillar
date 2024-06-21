#ifndef _VOXELIZATION_CUH_
#define _VOXELIZATION_CUH_

#include <cmath>

#include "centerpoint/base.hpp"

__global__ void generateVoxels_random_kernel(const float *points, size_t points_size,
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
__global__ void generateBaseFeatures_kernel(unsigned int *mask, float *voxels,
                                            int grid_y_size, int grid_x_size, unsigned int max_voxels,
                                            unsigned int *pillar_num,
                                            float *voxel_features,
                                            unsigned int *voxel_num,
                                            unsigned int *voxel_idxs);
cudaError_t generateBaseFeatures_launch(unsigned int *mask, float *voxels,
                                        int grid_y_size, int grid_x_size, unsigned int max_voxels,
                                        unsigned int *pillar_num,
                                        float *voxel_features,
                                        unsigned int *voxel_num,
                                        unsigned int *voxel_idxs,
                                        cudaStream_t stream);
__global__ void generateFeatures_kernel(float* voxel_features,
                                        unsigned int* voxel_num, unsigned int* voxel_idxs, unsigned int *params,
                                        float voxel_x, float voxel_y, float voxel_z, unsigned int max_voxels,
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
  Voxelization(const YAML::Node& config);
  ~Voxelization();
  void forward(const float *_points, int num_points, void *stream);
  const float *features() const { return dev_features_input_; }
  const unsigned int *coords() const { return dev_voxel_idxs_; }
  const unsigned int *nums() const { return dev_params_input_; }
  const VoxelizationParameter& param() const { return param_; }

private:
  void memoryInit(const YAML::Node& config);
  void setParams(const YAML::Node& config);

private:
  VoxelizationParameter param_;

  unsigned int *dev_mask_ = nullptr;
  unsigned int *dev_voxel_num_ = nullptr;
  unsigned int *dev_voxel_idxs_ = nullptr;
  unsigned int *dev_params_input_ = nullptr;

  float *dev_voxels_ = nullptr;
  float *dev_voxel_features_ = nullptr;
  float *dev_features_input_ = nullptr;
};

#endif