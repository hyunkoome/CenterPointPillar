#ifndef _POSTPROCESS_HPP_
#define _POSTPROCESS_HPP_

#include <algorithm>

#include "pycenterpoint/base.hpp"

#define DIVUP(x, y) (x + y - 1) / y
const int NMS_THREADS_PER_BLOCK = sizeof(uint64_t) * 8;
const int BOX_CHANNEL = 9;

struct PostProcessParameter {
  int feature_x_size;
  int feature_y_size;
  int out_size_factor;
  float pillar_x_size;
  float pillar_y_size;
  float min_x_range;
  float min_y_range;
};

void generateBoxes(const float* reg, const float* height, const float* dim, const float* rot,
                   const float* score, const int32_t* cls, const float* iou, const float* iou_rectify,
                   float* detections, unsigned int* detections_num, PostProcessParameter param,
                   void* stream);
__global__ void generateBoxesKernel(const float* reg, const float* height, const float* dim, const float* rot,
                                    const float* score, const int32_t* cls, const float* iou, const float* iou_rectify,
                                    PostProcessParameter param, float* detections, unsigned int* detections_num);
void nmsLaunch(unsigned int boxes_num, float nms_iou_threshold,
               float *boxes_sorted, uint64_t* mask, cudaStream_t stream);
__global__ void nmsLaunchKernel(const int n_boxes, const float iou_threshold,
                                const float *dev_boxes, uint64_t *dev_mask);
__device__ inline bool devIoU(float const *const box_a, float const *const box_b, const float nms_thresh);
__device__ inline float cross(const float2 p1, const float2 p2, const float2 p0);
__device__ inline int checkBox2d(float const *const box, const float2 p);
__device__ inline bool intersection(const float2 p1, const float2 p0, const float2 q1, const float2 q0, float2 &ans);
__device__ inline void rotateAroundCenter(const float2 &center, const float angle_cos, const float angle_sin, float2 &p);

class PostProcess {
public:
PostProcess(const YAML::Node& config);
~PostProcess();
int forward(const float* reg, const float* height, const float* dim, const float* rot,
            const float* score, const int32_t* cls, const float* iou,
            void * stream);
std::vector<Box> getBoxes();

private:
void memoryInit();
void setParams(const YAML::Node& config);

private:
unsigned int* host_detections_num_ = nullptr;
uint64_t* host_mask_               = nullptr;
float* dev_detections_             = nullptr;
float* dev_iou_rectify_            = nullptr;

unsigned int host_mask_size_;
unsigned int nms_pre_max_size_;
unsigned int nms_post_max_size_;

float score_threshold_;
float iou_threshold_;

std::vector<float> iou_rectify_;
std::vector<Box> boxes_pre_nms_;
std::vector<Box> boxes_post_nms_;

PostProcessParameter param_;
};

#endif // _POSTPROCESS_HPP_