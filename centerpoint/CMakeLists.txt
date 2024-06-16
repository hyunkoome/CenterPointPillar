cmake_minimum_required(VERSION 3.5)
project(centerpoint LANGUAGES CXX CUDA)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# CUDA Flag settings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80") # A100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# find dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(pcl_conversions REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(YAML-CPP REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

add_executable(centerpoint_node
  src/main.cc
  src/centerpoint/centerpoint.cc
  src/centerpoint/network.cc
  src/centerpoint/voxelization.cu
  src/centerpoint/voxelization.cc
  src/centerpoint/postprocess.cu
  src/centerpoint/postprocess.cc
  src/pillarscatter/pillarscatter-kernel.cu
  src/pillarscatter/pillarscatter.cc
)

target_link_libraries(centerpoint_node
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TensorRT_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  stdc++fs
)

ament_target_dependencies(centerpoint_node
  "rclcpp"
  "sensor_msgs"
  "visualization_msgs"
  "pcl_conversions")

install(TARGETS centerpoint_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY model
DESTINATION share/${PROJECT_NAME}
)

ament_package()
