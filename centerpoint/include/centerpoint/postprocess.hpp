#ifndef _POSTPROCESS_HPP_
#define _POSTPROCESS_HPP_

#include "centerpoint/base.hpp"

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define THREADS_PER_BLOCK_NMS (sizeof(unsigned long long) * 8)

class PostProcess {
public:
PostProcess(const YAML::Node& config){
  int num_class = config["num_class"].as<int>();
  iou_rectify_ = config["iou"].as<std::vector<float>>();
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
  unsigned int feature_y_size = config["feature_size"]["y"].as<unsigned int>();
  unsigned int feature_x_size = config["feature_size"]["x"].as<unsigned int>();
  unsigned int nms_pre_max_size = config["nms"]["pre_max"].as<unsigned int>();
  unsigned int nms_post_max_size = config["nms"]["post_max"].as<unsigned int>();
  unsigned int feature_size = feature_x_size * feature_y_size;

  unsigned int score_idx_size = feature_size * sizeof(int);
  unsigned int label_size = nms_post_max_size * sizeof(int);
  unsigned int boxes_size = nms_post_max_size * 9 * sizeof(float);
  unsigned int keep_data_size = nms_pre_max_size * sizeof(long);
  unsigned int mask_size = nms_pre_max_size *
                           DIVUP(nms_pre_max_size, THREADS_PER_BLOCK_NMS) *
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
void forward() {
  // do something
}

private:
int* dev_score_idx_;
int* host_score_idx_;
int* host_label_;
float* host_boxes_;
long* host_keep_data_;
unsigned long long* mask_cpu_;
unsigned long long* remv_cpu_;

std::vector<float> iou_rectify_;
float* dev_iou_rectify_;


};

#endif // _POSTPROCESS_HPP_