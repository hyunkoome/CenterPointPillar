#ifndef _CENTERPOINT_HPP_
#define _CENTERPOINT_HPP_

#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <yaml-cpp/yaml.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pycenterpoint/voxelization.hpp"
#include "pycenterpoint/network.hpp"
#include "pycenterpoint/postprocess.hpp"
#include "pycenterpoint/npy.hpp"

namespace py = pybind11;
template <typename T>
using PyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

class CenterPoint
{
public:
  CenterPoint(const std::string& config_path, const std::string& model_path);
  ~CenterPoint();
  std::vector<Box> forward(PyArray<float> np_array);
  std::vector<Box> npy_forward(const std::string& npy_path);

private:
  void memoryInit(const std::string& model_path);


private:
  // Config
  YAML::Node config_;

  // PCL Library
  std::minstd_rand0 rng_ = std::default_random_engine{};

  // CenterPoint Pipeline
  std::shared_ptr<Voxelization> voxelization_ = nullptr;
  std::shared_ptr<Network> network_           = nullptr;
  std::shared_ptr<PostProcess> postprocess_   = nullptr;

  // Point Cloud Input
  std::vector<float> points_;
  size_t max_points;
  float* input_points_ = nullptr;
  float* dev_input_points_ = nullptr;

  // CUDA Stream
  cudaStream_t stream_;

  // Output
  float score_threshold_;
  std::vector<Box>* boxes_;


};

#endif // _CENTERPOINT_HPP_