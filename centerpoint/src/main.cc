# include <iostream>
#include "centerpoint/centerpoint.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto centerpoint_node = std::make_shared<CenterPoint>();
  rclcpp::spin(centerpoint_node);
  rclcpp::shutdown();

  return 0;
}