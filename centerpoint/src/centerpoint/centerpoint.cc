#include "centerpoint/centerpoint.hpp"

CenterPoint::CenterPoint() : Node("centerpoint")
{
  this->declare_parameter("config_path");
  this->declare_parameter("model_path");
  config_ = YAML::LoadFile(this->get_parameter("config_path").as_string());
  score_threshold_ = config_["centerpoint"]["score_threshold"].as<float>();
  std::string sub_topic_name = config_["centerpoint"]["sub"].as<std::string>();
  std::string pub_topic_name = config_["centerpoint"]["pub"].as<std::string>();

  sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(sub_topic_name, 1,
                std::bind(&CenterPoint::callback, this, std::placeholders::_1));
  pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(pub_topic_name, 1);
  pcl_cloud_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);

  checkRuntime(cudaStreamCreate(&stream_));

  memoryInit();
}

CenterPoint::~CenterPoint()
{
  checkRuntime(cudaStreamDestroy(stream_));
  checkRuntime(cudaFree(dev_input_points_));
}

void CenterPoint::memoryInit()
{
  max_points = config_["centerpoint"]["max_points"].as<size_t>();
  size_t max_points_feature = max_points * config_["voxelization"]["num_feature"].as<size_t>();
  points_.reserve(max_points_feature);
  size_t bytes_points_capacity = max_points_feature * sizeof(float);
  checkRuntime(cudaMalloc(&dev_input_points_, bytes_points_capacity));
  checkRuntime(cudaDeviceSynchronize());

  voxelization_ = std::make_shared<Voxelization>(config_["voxelization"]);
  checkRuntime(cudaDeviceSynchronize());

  network_ = std::make_shared<Network>(this->get_parameter("model_path").as_string());
  checkRuntime(cudaDeviceSynchronize());

  postprocess_ = std::make_shared<PostProcess>(config_["postprocess"]);
  checkRuntime(cudaDeviceSynchronize());
}

size_t CenterPoint::getPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcl_cloud)
{
  points_.clear();
  int num_points = pcl_cloud->size();
  for (int i = 0; i < num_points; i++) {
    points_.push_back(pcl_cloud->points[i].x);
    points_.push_back(pcl_cloud->points[i].y);
    points_.push_back(pcl_cloud->points[i].z);
    points_.push_back(pcl_cloud->points[i].intensity);
  }
  input_points_ = points_.data();

  return num_points;
}

void CenterPoint::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  pcl::fromROSMsg(*msg, *pcl_cloud_);
  std::shuffle(pcl_cloud_->begin(), pcl_cloud_->end(), rng_);
  size_t num_points = getPoints(pcl_cloud_);

  forward(num_points);

  pubishBoxes();
}

void CenterPoint::pubishBoxes()
{
  visualization_msgs::msg::MarkerArray msg;
  for (int i = 0; i < boxes_->size(); i++) {
    Box& box = (*boxes_)[i];
    if (box.score() < score_threshold_) {
      continue;
    }
    visualization_msgs::msg::Marker marker;
    marker.id = i;
    marker.header.frame_id = "base_link";
    marker.header.stamp = this->get_clock()->now();
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = box.x();
    marker.pose.position.y = box.y();
    marker.pose.position.z = box.z();

    marker.scale.x = box.l();
    marker.scale.y = box.w();
    marker.scale.z = box.h();

    tf2::Quaternion q;
    q.setRPY(0, 0, box.yaw());
    marker.pose.orientation.w = q.w();
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();

    int cls = box.cls();
    if (cls == 0) {
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 0.5;
    }
    else if (cls == 1){
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
    marker.lifetime = rclcpp::Duration(0, 300000000);
    msg.markers.push_back(marker);
  }
  pub_->publish(msg);
}

void CenterPoint::forward(size_t num_points)
{
  if (num_points > max_points) {
    std::cout << "Max Points Over: " << num_points << std::endl;
    num_points = max_points;
  }
  size_t bytes_points = num_points * voxelization_->param().num_feature * sizeof(float);
  checkRuntime(cudaMemcpyAsync(dev_input_points_, input_points_, bytes_points, cudaMemcpyHostToDevice, stream_));

  voxelization_->forward(dev_input_points_, num_points, stream_);

  network_->forward(voxelization_->features(), voxelization_->coords(), voxelization_->nums(), stream_);

  int box_num = postprocess_->forward(network_->center(), network_->center_z(), network_->dim(), network_->rot(),
                                      network_->score(), network_->label(), network_->iou(), stream_);

  boxes_ = postprocess_->getBoxes();
  checkRuntime(cudaStreamSynchronize(stream_));
}