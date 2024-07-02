#include "centerpoint/postprocess.hpp"

PostProcess::PostProcess(const YAML::Node& config)
{
  setParams(config);
  boxes_pre_nms_.reserve(nms_pre_max_size_);
  boxes_post_nms_.reserve(nms_post_max_size_);
  memoryInit();
}

PostProcess::~PostProcess()
{
  checkRuntime(cudaFree(dev_detections_));
  checkRuntime(cudaFreeHost(host_detections_num_));
  checkRuntime(cudaFree(dev_iou_rectify_));
  checkRuntime(cudaFreeHost(host_mask_));
}

void PostProcess::setParams(const YAML::Node& config)
{
  std::cout << "=== PostProcess Parameters ===" << std::endl << config << std::endl;
  iou_rectify_ = config["iou"].as<std::vector<float>>();
  iou_threshold_ = config["nms"]["iou_threshold"].as<float>();
  nms_pre_max_size_ = config["nms"]["pre_max"].as<unsigned int>();
  nms_post_max_size_ = config["nms"]["post_max"].as<unsigned int>();

  param_.feature_x_size = config["feature_x_size"].as<int>();
  param_.feature_y_size = config["feature_y_size"].as<int>();
  param_.out_size_factor = config["out_size_factor"].as<int>();
  param_.pillar_x_size = config["pillar_x_size"].as<float>();
  param_.pillar_y_size = config["pillar_y_size"].as<float>();
  param_.min_x_range = config["min_x_range"].as<float>();
  param_.min_y_range = config["min_y_range"].as<float>();
  std::cout << "================================" << std::endl;
}

void PostProcess::memoryInit()
{
  host_mask_size_ = nms_pre_max_size_ * DIVUP(nms_pre_max_size_, NMS_THREADS_PER_BLOCK) * sizeof(uint64_t);

  checkRuntime(cudaMalloc((void**)&dev_detections_, nms_pre_max_size_ * BOX_CHANNEL * sizeof(float)));
  checkRuntime(cudaMallocHost((void**)&host_detections_num_, sizeof(unsigned int)));
  checkRuntime(cudaMalloc((void**)&dev_iou_rectify_, iou_rectify_.size() * sizeof(float)));
  checkRuntime(cudaMallocHost((void**)&host_mask_, host_mask_size_));

  checkRuntime(cudaMemset(dev_detections_, 0, nms_pre_max_size_ * BOX_CHANNEL * sizeof(float)));
  checkRuntime(cudaMemcpy(dev_iou_rectify_, iou_rectify_.data(), iou_rectify_.size() * sizeof(float), cudaMemcpyHostToDevice));
  checkRuntime(cudaMemset(host_mask_, 0, host_mask_size_));
}

int PostProcess::forward(const float* reg, const float* height, const float* dim, const float* rot,
                         const float* score, const int32_t* cls, const float* iou,
                         void * stream)
{
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  boxes_pre_nms_.clear();
  boxes_post_nms_.clear();

  checkRuntime(cudaMemsetAsync(host_detections_num_, 0, sizeof(unsigned int), _stream));
  generateBoxes(reg, height, dim, rot,
                score, cls, iou, dev_iou_rectify_,
                dev_detections_, host_detections_num_, param_,
                stream);

  boxes_pre_nms_.resize(*host_detections_num_);
  if (*host_detections_num_ == 0) {
    std::cerr << "[ERR] No boxs detected." << std::endl;
    return 0;
  }
  if (*host_detections_num_ > nms_pre_max_size_) {
    std::cerr << "[ERR] Boxs num exceeds:" << *host_detections_num_ << std::endl;
  }

  checkRuntime(cudaMemcpyAsync(boxes_pre_nms_.data(), dev_detections_,
                               *host_detections_num_ * BOX_CHANNEL * sizeof(float),
                               cudaMemcpyDeviceToHost, _stream));
  std::sort(boxes_pre_nms_.begin(), boxes_pre_nms_.end(),
            [](const Box& a, const Box& b) { return a.val[7] > b.val[7]; });
  checkRuntime(cudaMemcpyAsync(dev_detections_, boxes_pre_nms_.data(),
                               *host_detections_num_ * BOX_CHANNEL * sizeof(float),
                               cudaMemcpyHostToDevice, _stream));

  checkRuntime(cudaMemsetAsync(host_mask_, 0, host_mask_size_, _stream));
  nmsLaunch(*host_detections_num_, iou_threshold_, dev_detections_, host_mask_, _stream);

  int col_blocks = DIVUP(*host_detections_num_, NMS_THREADS_PER_BLOCK);
  std::vector<uint64_t> remv(col_blocks, 0);
  std::vector<bool> keep(*host_detections_num_, false);
  int max_keep_size = 0;
  for (unsigned int i_nms = 0; i_nms < *host_detections_num_; i_nms++) {
    unsigned int nblock = i_nms / NMS_THREADS_PER_BLOCK;
    unsigned int inblock = i_nms % NMS_THREADS_PER_BLOCK;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[i_nms] = true;
      if (max_keep_size++ < nms_post_max_size_) {
        boxes_post_nms_.push_back(boxes_pre_nms_[i_nms]);
      }
      uint64_t* p = host_mask_ + i_nms * col_blocks;
      for (int j_nms = nblock; j_nms < col_blocks; j_nms++) {
        remv[j_nms] |= p[j_nms];
      }
    }
  }

  return boxes_post_nms_.size();
}

std::vector<Box>* PostProcess::getBoxes()
{
  return &boxes_post_nms_;
}

