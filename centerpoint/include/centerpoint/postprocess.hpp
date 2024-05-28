#ifndef _POSTPROCESS_HPP_
#define _POSTPROCESS_HPP_

#include "centerpoint/base.hpp"
#include "centerpoint/config.h"

#include <algorithm>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define THREADS_PER_BLOCK_NMS (sizeof(unsigned long long) * 8)

struct Point {
  float x, y;
  __device__ Point();
  __device__ Point(double _x, double _y);
  __device__ void set(float _x, float _y);
  __device__ Point operator+(const Point& b) const;
  __device__ Point operator-(const Point& b) const;
};

int find_valid_score_num(float* score, int feature_x_size, int feature_y_size, float threshold);
void sort_by_score(float* keys, int* values, int size);
int _raw_nms_gpu(const float* reg, const float* height, const float* dim,
                 const float* rot, const int* indexs, long* host_keep_data,
                 unsigned long long* mask_cpu, unsigned long long* remv_cpu,
                 int boxes_num, float nms_overlap_thresh);
void rawNmsLauncher(const float* reg, const float* height, const float* dim,
                    const float* rot, const int* indexs, unsigned long long* mask,
                    int boxes_num, float nms_overlap_thresh);
__global__ void raw_nms_kernel(const int boxes_num,
                               const float nms_overlap_thresh, const float* reg,
                               const float* height, const float* dim,
                               const float* rot, const int* indexs,
                               unsigned long long* mask);
__device__ float iou_bev(const float *box_a, const float *box_b);
__device__ float box_overlap(const float* box_a, const float* box_b);
__device__ void rotate_around_center(const Point& center,
                                     const float angle_cos,
                                     const float angle_sin, Point& p);
__device__ int intersection(const Point& p1, const Point& p0,
                            const Point& q1, const Point& q0,
                            Point& ans);
__device__ int check_rect_cross(const Point& p1, const Point& p2,
                                const Point& q1, const Point& q2);
__device__ float cross(const Point& a, const Point& b);
__device__ float cross(const Point& p1, const Point& p2, const Point& p0);
__device__ int check_in_box2d(const float* box, const Point& p);
__device__ int point_cmp(const Point& a, const Point& b, const Point& center);
void _gather_all(float* host_boxes, int* host_label, float* reg, float* height,
                 float* dim, float* rot, float* sorted_score, int32_t* label,
                 int* dev_indexs, long* host_keep_indexs, int boxSizeBef,
                 int boxSizeAft);

class PostProcess {
public:
PostProcess(const YAML::Node& config){
  int num_class = config["num_class"].as<int>();
  iou_rectify_ = config["iou"].as<std::vector<float>>();

  feature_y_size_ = config["feature_size"]["y"].as<unsigned int>();
  feature_x_size_ = config["feature_size"]["x"].as<unsigned int>();
  feature_size_ = feature_x_size_ * feature_y_size_;
  score_threshold_ = config["nms"]["score_threshold"].as<float>();
  iou_threshold_ = config["nms"]["iou_threshold"].as<float>();
  nms_pre_max_size_ = config["nms"]["pre_max"].as<unsigned int>();
  nms_post_max_size_ = config["nms"]["post_max"].as<unsigned int>();

  boxes_.reserve(nms_post_max_size_);
  init(config);
}

~PostProcess() {
  checkRuntime(cudaFree(dev_score_idx_));
  checkRuntime(cudaFreeHost(host_score_idx_));
  checkRuntime(cudaFreeHost(host_label_));
  checkRuntime(cudaFreeHost(host_boxes_));
  checkRuntime(cudaFreeHost(host_keep_data_));
  checkRuntime(cudaFreeHost(mask_cpu_));
  checkRuntime(cudaFreeHost(remv_cpu_));
  checkRuntime(cudaFree(dev_iou_rectify_));
}

private:
void init(const YAML::Node& config) {

  unsigned int score_idx_size = feature_size_ * sizeof(int);
  unsigned int label_size = nms_post_max_size_ * sizeof(int);
  unsigned int boxes_size = nms_post_max_size_ * 9 * sizeof(float);
  unsigned int keep_data_size = nms_pre_max_size_ * sizeof(long);
  unsigned int mask_size = nms_pre_max_size_ *
                           DIVUP(nms_pre_max_size_, THREADS_PER_BLOCK_NMS) *
                           sizeof(unsigned long long);
  unsigned int remv_size = THREADS_PER_BLOCK_NMS * sizeof(unsigned long long);

  checkRuntime(cudaMalloc((void**)&dev_score_idx_, score_idx_size));
  checkRuntime(cudaMallocHost((void**)&host_score_idx_, score_idx_size));
  checkRuntime(cudaMallocHost((void**)&host_label_, label_size));
  checkRuntime(cudaMallocHost((void**)&host_boxes_, boxes_size));
  checkRuntime(cudaMallocHost((void**)&host_keep_data_, keep_data_size));
  checkRuntime(cudaMallocHost((void**)&mask_cpu_, mask_size));
  checkRuntime(cudaMallocHost((void**)&remv_cpu_, remv_size));
  checkRuntime(cudaMalloc((void**)&dev_iou_rectify_, iou_rectify_.size() * sizeof(float)));

  checkRuntime(cudaMemset(dev_score_idx_, -1, score_idx_size));
  checkRuntime(cudaMemset(host_score_idx_, -1, score_idx_size));
  checkRuntime(cudaMemset(host_label_, -1, label_size));
  checkRuntime(cudaMemset(host_boxes_, 0, boxes_size));
  checkRuntime(cudaMemset(host_keep_data_, -1, keep_data_size));
  checkRuntime(cudaMemset(mask_cpu_, 0, mask_size));
  checkRuntime(cudaMemset(remv_cpu_, 0, remv_size));
  checkRuntime(cudaMemcpy(dev_iou_rectify_, iou_rectify_.data(), iou_rectify_.size() * sizeof(float), cudaMemcpyHostToDevice));
}


public:
int forward(float* reg, float* height, float* dim, float* rot,
             float* score, int32_t* cls, float* iou,
             void * stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  boxes_.clear();

  int box_pre_size = find_valid_score_num(score, feature_x_size_, feature_y_size_, score_threshold_);
  box_pre_size = std::min(box_pre_size, static_cast<int>(nms_pre_max_size_));
  std::cout << "box_pre_size: " << box_pre_size << std::endl;

  sort_by_score(score, dev_score_idx_, feature_size_);

  int box_post_size = _raw_nms_gpu(reg, height, dim,
                                   rot, dev_score_idx_, host_keep_data_,
                                   mask_cpu_, remv_cpu_,
                                   box_pre_size, iou_threshold_);
  box_post_size = std::min(box_post_size, static_cast<int>(nms_post_max_size_));
  std::cout << "box_post_size: " << box_post_size << std::endl;

  _gather_all(host_boxes_, host_label_,
              reg, height, dim, rot, score, cls,
              dev_score_idx_, host_keep_data_, box_pre_size, box_post_size);

  checkRuntime(cudaMemcpy(host_score_idx_, dev_score_idx_, box_post_size * sizeof(int), cudaMemcpyDeviceToHost));
  for (auto i = 0; i < box_post_size; i++) {
    int ii = host_keep_data_[i];
    int idx = host_score_idx_[ii];
    int xIdx = idx % feature_x_size_;
    int yIdx = idx / feature_x_size_;
    Box box;

    box.x = (host_boxes_[i + 0 * box_post_size] + xIdx) * OUT_SIZE_FACTOR * PILLAR_X_SIZE + MIN_X_RANGE;
    box.y = (host_boxes_[i + 1 * box_post_size] + yIdx) * OUT_SIZE_FACTOR * PILLAR_Y_SIZE + MIN_Y_RANGE;
    box.z = host_boxes_[i + 2 * box_post_size];
    box.l = host_boxes_[i + 3 * box_post_size];
    box.w = host_boxes_[i + 4 * box_post_size];
    box.h = host_boxes_[i + 5 * box_post_size];
    float theta_c = host_boxes_[i + 6 * box_post_size];
    float theta_s = host_boxes_[i + 7 * box_post_size];
    box.rt = atan2(theta_s, theta_c);
    box.score = host_boxes_[i + 8 * box_post_size];
    box.id = host_label_[i];
    boxes_.push_back(box);
  }

  return box_post_size;
}

std::vector<Box>& getBoxes() {
  return boxes_;
}

private:
int* dev_score_idx_;
int* host_score_idx_;
int* host_label_;
float* host_boxes_;
long* host_keep_data_;
unsigned long long* mask_cpu_;
unsigned long long* remv_cpu_;

unsigned int feature_x_size_;
unsigned int feature_y_size_;
unsigned int feature_size_;
float score_threshold_;
float iou_threshold_;
unsigned int nms_pre_max_size_;
unsigned int nms_post_max_size_;
std::vector<float> iou_rectify_;
float* dev_iou_rectify_;

std::vector<Box> boxes_;


};

#endif // _POSTPROCESS_HPP_