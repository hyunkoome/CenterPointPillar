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
  CenterPoint() : Node("centerpoint")
  {
    this->declare_parameter("config_path");
    this->declare_parameter("model_path");
    config_ = YAML::LoadFile(this->get_parameter("config_path").as_string());
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      config_["centerpoint"]["sub"].as<std::string>(), 1, std::bind(&CenterPoint::callback, this, std::placeholders::_1));
    pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(config_["centerpoint"]["pub"].as<std::string>(), 1);
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

    network_ = std::make_shared<Network>(this->get_parameter("model_path").as_string());
    checkRuntime(cudaDeviceSynchronize());

    postprocess_ = std::make_shared<PostProcess>(config_["postprocess"]);

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

    start = std::chrono::steady_clock::now();
    forward(num_points);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "duration: " << duration.count() << " ms" << std::endl;
    pubishBoxes();
  }

  void pubishBoxes() {
    visualization_msgs::msg::MarkerArray msg;
    for (int i = 0; i < boxes_.size(); i++) {
      visualization_msgs::msg::Marker marker;
      marker.id = i;
      marker.header.frame_id = "base_link";
      marker.header.stamp = this->get_clock()->now();
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;

      marker.pose.position.x = boxes_[i].x;
      marker.pose.position.y = boxes_[i].y;
      marker.pose.position.z = boxes_[i].z;

      tf2::Quaternion q;
      q.setRPY(0, 0, boxes_[i].rt);
      marker.pose.orientation.w = q.w();
      marker.pose.orientation.x = q.x();
      marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z();

      marker.scale.x = boxes_[i].l;
      marker.scale.y = boxes_[i].w;
      marker.scale.z = boxes_[i].h;

      if (boxes_[i].id == 0) {
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.5;
      }
      else if (boxes_[i].id == 1){
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.5;
      }
      else {
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 0.5;
      }

      msg.markers.push_back(marker);
    }
    pub_->publish(msg);
  }

  void forward(size_t num_points)
  {
    size_t bytes_points = num_points * voxelization_->param_.num_feature * sizeof(float);
    checkRuntime(cudaMemcpyAsync(input_points_device_, input_points_, bytes_points, cudaMemcpyHostToDevice, stream_));
    voxelization_->forward(input_points_device_, num_points, stream_);
    network_->forward(voxelization_->features(), voxelization_->coords(), voxelization_->nums(), stream_);
    int box_num = postprocess_->forward(network_->center(), network_->center_z(), network_->dim(), network_->rot(),
                                        network_->score(), network_->label(), network_->iou(), stream_);
    boxes_ = std::move(postprocess_->getBoxes());
    checkRuntime(cudaStreamSynchronize(stream_));
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
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
  std::shared_ptr<PostProcess> postprocess_   = nullptr;

  cudaStream_t stream_;

  std::vector<Box> boxes_;

};

#endif // _CENTERPOINT_HPP_