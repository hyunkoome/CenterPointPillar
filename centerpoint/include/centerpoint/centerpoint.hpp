#ifndef _CENTERPOINT_HPP_
#define _CENTERPOINT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <yaml-cpp/yaml.h>

#include <centerpoint/voxelization.cuh>
#include <centerpoint/network.hpp>

class CenterPoint : public rclcpp::Node
{
public:
  CenterPoint() : Node("centerpoint")
  {
    this->declare_parameter("config_path");
    this->declare_parameter("model_path");
    config_ = YAML::LoadFile(this->get_parameter("config_path").as_string());
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/lidar/top/pointcloud", 1, std::bind(&CenterPoint::callback, this, std::placeholders::_1));
    pcl_cloud_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  }
  ~CenterPoint() {
    checkRuntime(cudaStreamDestroy(stream_));
    checkRuntime(cudaFree(input_points_device_));
  };

  void init() {
    checkRuntime(cudaStreamCreate(&stream_));
    voxelization_ = std::make_shared<Voxelization>(config_["voxelization"]);
    checkRuntime(cudaDeviceSynchronize());
    network_ = std::make_shared<Network>(this->get_parameter("model_path").as_string(), config_["network"],
                                         voxelization_->features(), voxelization_->coords(), voxelization_->params());
    checkRuntime(cudaDeviceSynchronize());
    size_t capacity_points_ = config_["centerpoint"]["max_points"].as<size_t>();
    size_t bytes_capacity_points_ = capacity_points_ * voxelization_->param_.num_feature * sizeof(float);
    checkRuntime(cudaMalloc(&input_points_device_, bytes_capacity_points_));
    checkRuntime(cudaDeviceSynchronize());
  }

private:
  size_t getPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_cloud)
  {
    points_.clear();
    int num_points = pcl_cloud->size();
    points_.reserve(num_points * voxelization_->param_.num_feature);
    for (int i = 0; i < num_points; i++)
    {
      points_.push_back(pcl_cloud->points[i].x);
      points_.push_back(pcl_cloud->points[i].y);
      points_.push_back(pcl_cloud->points[i].z);
      points_.push_back(pcl_cloud->points[i].intensity);
    }
    input_points_ = points_.data();

    return num_points;
  }

  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::fromROSMsg(*msg, *pcl_cloud_);
    std::shuffle(pcl_cloud_->begin(), pcl_cloud_->end(), rng);
    size_t num_points = getPoints(pcl_cloud_);
    std::cout << "num_points: " << num_points << std::endl;

    forward(num_points);
  }

  void forward(size_t num_points)
  {
    size_t bytes_points = num_points * voxelization_->param_.num_feature * sizeof(float);
    checkRuntime(cudaMemcpyAsync(input_points_device_, input_points_, bytes_points, cudaMemcpyHostToDevice, stream_));
    voxelization_->forward(input_points_device_, num_points, stream_);
    checkRuntime(cudaStreamSynchronize(stream_));
    bool status =  network_->forward(stream_);
    if (!status) {
      std::cerr << "Failed to forward" << std::endl;
      return;
    }
    checkRuntime(cudaStreamSynchronize(stream_));
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_;

  std::minstd_rand0 rng = std::default_random_engine{};

  std::chrono::steady_clock::time_point start, end;
  std::chrono::milliseconds duration;

  std::vector<float> points_;
  float* input_points_ = nullptr;
  float* input_points_device_ = nullptr;

  YAML::Node config_;

  std::shared_ptr<Voxelization> voxelization_ = nullptr;
  std::shared_ptr<Network> network_           = nullptr;

  cudaStream_t stream_;

};

#endif // _CENTERPOINT_HPP_