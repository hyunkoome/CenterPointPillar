cmake_minimum_required(VERSION 3.5)
project(centerpoint_pybind LANGUAGES CXX CUDA)

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
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(YAML-CPP REQUIRED)
find_package(PythonLibs REQUIRED)

function(copy_output target_name to_path)
add_custom_command(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:${target_name}>"
    "${PROJECT_SOURCE_DIR}/${to_path}/$<TARGET_FILE_NAME:${target_name}>"
    )
endfunction()


include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

pybind11_add_module(cp MODULE
  module.cc
  src/pycenterpoint/centerpoint.cc
  src/pycenterpoint/network.cc
  src/pycenterpoint/voxelization.cu
  src/pycenterpoint/voxelization.cc
  src/pycenterpoint/postprocess.cu
  src/pycenterpoint/postprocess.cc
  src/pillarscatter/pillarscatter-kernel.cu
  src/pillarscatter/pillarscatter.cc)

target_link_libraries(cp PRIVATE
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TensorRT_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  stdc++fs
)
copy_output(cp ../../tools)

add_executable(npy_forward
  npy_forward.cc
  src/pycenterpoint/centerpoint.cc
  src/pycenterpoint/network.cc
  src/pycenterpoint/voxelization.cu
  src/pycenterpoint/voxelization.cc
  src/pycenterpoint/postprocess.cu
  src/pycenterpoint/postprocess.cc
  src/pillarscatter/pillarscatter-kernel.cu
  src/pillarscatter/pillarscatter.cc
)

target_link_libraries(npy_forward
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TensorRT_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${PYTHON_LIBRARIES}
  stdc++fs
)