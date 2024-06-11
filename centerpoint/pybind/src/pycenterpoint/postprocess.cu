#include "pycenterpoint/postprocess.hpp"

void generateBoxes(const float* reg, const float* height, const float* dim, const float* rot,
                   const float* score, const int32_t* cls, const float* iou, const float* iou_rectify,
                   float* detections, unsigned int* detections_num, PostProcessParameter param,
                   void* stream)
{
  int FEATURE_X_SIZE = param.feature_x_size;
  int FEATURE_Y_SIZE = param.feature_y_size;
  cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
  dim3 threads(32);
  dim3 blocks((FEATURE_X_SIZE * FEATURE_Y_SIZE + threads.x - 1) / threads.x);
  generateBoxesKernel<<<blocks, threads, 0, stream_>>>(reg, height, dim, rot,
                                                       score, cls, iou, iou_rectify,
                                                       param, detections, detections_num);

  checkRuntime(cudaStreamSynchronize(stream_));
}

__global__ void generateBoxesKernel(const float* reg, const float* height, const float* dim, const float* rot,
                                    const float* score, const int32_t* cls, const float* iou, const float* iou_rectify,
                                    PostProcessParameter param, float* detections, unsigned int* detections_num)
{
  int FEATURE_X_SIZE = param.feature_x_size;
  int FEATURE_Y_SIZE = param.feature_y_size;
  int OUT_SIZE_FACTOR = param.out_size_factor;
  float PILLAR_X_SIZE = param.pillar_x_size;
  float PILLAR_Y_SIZE = param.pillar_y_size;
  float MIN_X_RANGE = param.min_x_range;
  float MIN_Y_RANGE = param.min_y_range;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int HW = FEATURE_X_SIZE * FEATURE_Y_SIZE;
  if (x >= HW) return;

  int h = x / FEATURE_X_SIZE;
  int w = x % FEATURE_X_SIZE;

  float score_ = score[h * FEATURE_X_SIZE + w];
  if (score_ < 0.2) return;

  float iou_ = iou[h * FEATURE_X_SIZE + w];
  iou_ = fminf((fmaxf(iou_, 0.0f)), 1.0f);
  int32_t cls_ = cls[h * FEATURE_X_SIZE + w];
  float final_score = powf(score_, 1 - iou_rectify[cls_]) * powf(iou_, iou_rectify[cls_]);

  float xs = reg[h * FEATURE_X_SIZE + w] + w;
  float ys = reg[HW + h * FEATURE_X_SIZE + w] + h;
  float zs = height[h * FEATURE_X_SIZE + w];

  xs = xs * OUT_SIZE_FACTOR * PILLAR_X_SIZE + MIN_X_RANGE;
  ys = ys * OUT_SIZE_FACTOR * PILLAR_Y_SIZE + MIN_Y_RANGE;


  unsigned int curDet = 0;
  curDet = atomicAdd(detections_num, 1);

  float3 dim_;
  dim_.x = dim[0 * HW + h * FEATURE_X_SIZE + w];
  dim_.y = dim[1 * HW + h * FEATURE_X_SIZE + w];
  dim_.z = dim[2 * HW + h * FEATURE_X_SIZE + w];

  float rs = atan2(rot[HW + h * FEATURE_X_SIZE + w], rot[h * FEATURE_X_SIZE + w]);

  *(float3 *)(&detections[curDet * BOX_CHANNEL + 0]) = make_float3(xs, ys, zs);
  *(float3 *)(&detections[curDet * BOX_CHANNEL + 3]) = dim_;
  *(float3 *)(detections + curDet * BOX_CHANNEL + 6) = make_float3(rs, final_score, cls_);
}

void nmsLaunch(unsigned int boxes_num, float nms_iou_threshold,
               float *boxes_sorted, uint64_t* mask, cudaStream_t stream)
{
  int col_blocks = DIVUP(boxes_num, NMS_THREADS_PER_BLOCK);

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(NMS_THREADS_PER_BLOCK);

  nmsLaunchKernel<<<blocks, threads, 0, stream>>>(boxes_num, nms_iou_threshold, boxes_sorted, mask);
  cudaStreamSynchronize(stream);
}

__global__ void nmsLaunchKernel(const int n_boxes, const float iou_threshold,
                                const float *dev_boxes, uint64_t *dev_mask)
{
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int tid = threadIdx.x;

  if (row_start > col_start) return;

  const int row_size = fminf(n_boxes - row_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);
  const int col_size = fminf(n_boxes - col_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);

  __shared__ float block_boxes[NMS_THREADS_PER_BLOCK * 7];

  if (tid < col_size) {
  block_boxes[tid * 7 + 0] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * BOX_CHANNEL + 0];
  block_boxes[tid * 7 + 1] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * BOX_CHANNEL + 1];
  block_boxes[tid * 7 + 2] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * BOX_CHANNEL + 2];
  block_boxes[tid * 7 + 3] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * BOX_CHANNEL + 3];
  block_boxes[tid * 7 + 4] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * BOX_CHANNEL + 4];
  block_boxes[tid * 7 + 5] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * BOX_CHANNEL + 5];
  block_boxes[tid * 7 + 6] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * BOX_CHANNEL + 6];
  }
  __syncthreads();

  if (tid < row_size) {
    const int cur_box_idx = NMS_THREADS_PER_BLOCK * row_start + tid;
    const float *cur_box = dev_boxes + cur_box_idx * BOX_CHANNEL;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = tid + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 7, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    dev_mask[cur_box_idx * gridDim.y + col_start] = t;
  }
}

__device__ inline bool devIoU(float const *const box_a, float const *const box_b, const float nms_thresh) {
  float a_angle = box_a[6], b_angle = box_b[6];
  float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
  float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
  float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
  float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
  float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;
  float2 box_a_corners[5];
  float2 box_b_corners[5];

  float2 center_a = float2 {box_a[0], box_a[1]};
  float2 center_b = float2 {box_b[0], box_b[1]};

  float2 cross_points[16];
  float2 poly_center =  {0, 0};
  int cnt = 0;
  bool flag = false;

  box_a_corners[0] = float2 {a_x1, a_y1};
  box_a_corners[1] = float2 {a_x2, a_y1};
  box_a_corners[2] = float2 {a_x2, a_y2};
  box_a_corners[3] = float2 {a_x1, a_y2};

  box_b_corners[0] = float2 {b_x1, b_y1};
  box_b_corners[1] = float2 {b_x2, b_y1};
  box_b_corners[2] = float2 {b_x2, b_y2};
  box_b_corners[3] = float2 {b_x1, b_y2};

  float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
  float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

  for (int k = 0; k < 4; k++) {
    rotateAroundCenter(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
    rotateAroundCenter(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
  }

  box_a_corners[4] = box_a_corners[0];
  box_b_corners[4] = box_b_corners[0];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                          box_b_corners[j + 1], box_b_corners[j],
                          cross_points[cnt]);
      if (flag) {
          poly_center = {poly_center.x + cross_points[cnt].x, poly_center.y + cross_points[cnt].y};
          cnt++;
      }
    }
  }

  for (int k = 0; k < 4; k++) {
    if (checkBox2d(box_a, box_b_corners[k])) {
      poly_center = {poly_center.x + box_b_corners[k].x, poly_center.y + box_b_corners[k].y};
      cross_points[cnt] = box_b_corners[k];
      cnt++;
    }
    if (checkBox2d(box_b, box_a_corners[k])) {
      poly_center = {poly_center.x + box_a_corners[k].x, poly_center.y + box_a_corners[k].y};
      cross_points[cnt] = box_a_corners[k];
      cnt++;
    }
  }

  poly_center.x /= cnt;
  poly_center.y /= cnt;

  float2 temp;
  for (int j = 0; j < cnt - 1; j++) {
    for (int i = 0; i < cnt - j - 1; i++) {
      if (atan2(cross_points[i].y - poly_center.y, cross_points[i].x - poly_center.x) >
          atan2(cross_points[i+1].y - poly_center.y, cross_points[i+1].x - poly_center.x)
          ) {
        temp = cross_points[i];
        cross_points[i] = cross_points[i + 1];
        cross_points[i + 1] = temp;
      }
    }
  }

  float area = 0;
  for (int k = 0; k < cnt - 1; k++) {
    float2 a = {cross_points[k].x - cross_points[0].x,
                cross_points[k].y - cross_points[0].y};
    float2 b = {cross_points[k + 1].x - cross_points[0].x,
                cross_points[k + 1].y - cross_points[0].y};
    area += (a.x * b.y - a.y * b.x);
  }

  float s_overlap = fabs(area) / 2.0;;
  float sa = box_a[3] * box_a[4];
  float sb = box_b[3] * box_b[4];
  float iou = s_overlap / fmaxf(sa + sb - s_overlap, 1e-8);

  return iou >= nms_thresh;
}

__device__ inline float cross(const float2 p1, const float2 p2, const float2 p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ inline int checkBox2d(float const *const box, const float2 p) {
  const float MARGIN = 1e-2;
  float center_x = box[0];
  float center_y = box[1];
  float angle_cos = cos(-box[6]);
  float angle_sin = sin(-box[6]);
  float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
  float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

  return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline bool intersection(const float2 p1, const float2 p0, const float2 q1, const float2 q0, float2 &ans) {

  if (( fmin(p0.x, p1.x) <= fmax(q0.x, q1.x) &&
        fmin(q0.x, q1.x) <= fmax(p0.x, p1.x) &&
        fmin(p0.y, p1.y) <= fmax(q0.y, q1.y) &&
        fmin(q0.y, q1.y) <= fmax(p0.y, p1.y) ) == 0)
    return false;


  float s1 = cross(q0, p1, p0);
  float s2 = cross(p1, q1, p0);
  float s3 = cross(p0, q1, q0);
  float s4 = cross(q1, p1, q0);

  if (!(s1 * s2 > 0 && s3 * s4 > 0))
    return false;

  float s5 = cross(q1, p1, p0);
  if (fabs(s5 - s1) > 1e-8) {
    ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
    ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

  } else {
    float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
    float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
    float D = a0 * b1 - a1 * b0;

    ans.x = (b0 * c1 - b1 * c0) / D;
    ans.y = (a1 * c0 - a0 * c1) / D;
  }

  return true;
}

__device__ inline void rotateAroundCenter(const float2 &center, const float angle_cos, const float angle_sin, float2 &p) {
  float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
  float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
  p = float2 {new_x, new_y};
  return;
}