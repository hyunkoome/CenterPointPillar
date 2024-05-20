# include <iostream>
# include "centerpoint/pillarScatter.hpp"

// using PillarScatterPluginCreator = nvinfer1::plugin::PillarScatterPluginCreator;

int main() {
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  nvinfer1::plugin::PillarScatterPlugin plugin(1, 1);
  std::cout << "Hello, TensorRT!" << std::endl;

  return 0;
}