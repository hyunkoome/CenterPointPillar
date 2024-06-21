#ifndef _CENTERPOINT_HPP_
#define _CENTERPOINT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/quaternion.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <yaml-cpp/yaml.h>

#include <centerpoint/voxelization.hpp>
#include <centerpoint/network.hpp>
#include <centerpoint/postprocess.hpp>

class CenterPoint : public rclcpp::Node
{
public:
  CenterPoint();
  ~CenterPoint();

private:
  void memoryInit();
  size_t getPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_cloud);
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void pubishBoxes();
  void forward(size_t num_points);


private:
  // ROS2 & Config
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
  YAML::Node config_;

  // PCL Library
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud_;
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