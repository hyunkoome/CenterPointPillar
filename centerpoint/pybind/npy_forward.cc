#include <iostream>
#include <algorithm>
#include <experimental/filesystem>

#include "pycenterpoint/centerpoint.hpp"

namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
  CenterPoint detector(argv[1], argv[2]);

  fs::path npy_dir(argv[3]);

  std::vector<std::string> npy_files;
  npy_files.reserve(250);

  for (const auto& entry : fs::directory_iterator(npy_dir)) {
    if (entry.path().extension() != ".npy") continue;
    npy_files.push_back(entry.path().string());
  }

  std::sort(npy_files.begin(), npy_files.end());
  for (const auto& npy_path : npy_files) {
    std::cout << "Processing: " << npy_path << std::endl;
    std::vector<Box> boxes = detector.npy_forward(npy_path);
    std::cout << "Boxes: " << boxes.size() << std::endl;
  }


  return 0;
}