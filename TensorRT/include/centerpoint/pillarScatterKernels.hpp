/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __PILLAR_SCATTER_KERNELS_HPP__
#define __PILLAR_SCATTER_KERNELS_HPP__

#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

namespace nvinfer1
{
namespace plugin
{
const int PILLARS_PER_BLOCK = 64;
const int PILLAR_FEATURE_SIZE = 64;

__global__ void scatterBEV_kernel(const half *pillar_features_data,
          const unsigned int *coords_data, const unsigned int *params_data,
          unsigned int featureX, unsigned int featureY,
          half *spatial_feature_data);

__global__ void scatterBEV_kernel(const float *pillar_features_data,
          const unsigned int *coords_data, const unsigned int *params_data,
          unsigned int featureX, unsigned int featureY,
          float *spatial_feature_data);

int pillarScatterKernelLaunch(
  int batch_size,
  int max_pillar_num,
  int num_features,
  const half *pillar_features_data,
  const unsigned int *coords_data,
  const unsigned int *params_data,
  unsigned int featureX, unsigned int featureY,
  half *spatial_feature_data,
  cudaStream_t stream);

int pillarScatterKernelLaunch(
  int batch_size,
  int max_pillar_num,
  int num_features,
  const float *pillar_features_data,
  const unsigned int *coords_data,
  const unsigned int *params_data,
  unsigned int featureX, unsigned int featureY,
  float *spatial_feature_data,
  cudaStream_t stream);

} // namespace plugin
} // namespace nvinfer1
#endif // __PILLAR_SCATTER_KERNELS_HPP__
