project(ppscatter LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA found (${CUDA_VERSION_STRING})")
    message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -O3 -gencode arch=compute_61,code=sm_61)
else ()
    message(FATAL_ERROR "CUDA not found")
endif ()
find_package(TensorRT REQUIRED)

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
)

####################### libppscatter.so ##########################
add_library(ppscatter SHARED
    pointpillar/pointpillar-scatter.cpp
    pointpillar/pillarscatter-kernel.cu
)

target_link_libraries(ppscatter
    ${CUDA_LIBRARIES}
	${TensorRT_LIBRARY}
)