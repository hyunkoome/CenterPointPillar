#ifndef __PILLAR_SCATTER_COMMON_HPP__
#define __PILLAR_SCATTER_COMMON_HPP__

#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>
#include <set>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <assert.h>

namespace nvinfer1
{
namespace plugin
{
template <typename ToType, typename FromType>
ToType* toPointer(FromType* ptr)
{
    return static_cast<ToType*>(static_cast<void*>(ptr));
}
template <typename ToType, typename FromType>
ToType const* toPointer(FromType const* ptr)
{
    return static_cast<ToType const*>(static_cast<void const*>(ptr));
}
// Helper function for serializing plugin
template <typename ValType, typename BufferType>
void writeToBuffer(BufferType*& buffer, ValType const& val)
{
    *toPointer<ValType>(buffer) = val;
    buffer += sizeof(ValType);
}

// Helper function for deserializing plugin
template <typename ValType, typename BufferType>
ValType readFromBuffer(BufferType const*& buffer)
{
    auto val = *toPointer<ValType const>(buffer);
    buffer += sizeof(ValType);
    return val;
}

// void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, PluginFieldCollection const* fc)
// {
//     for (int32_t i = 0; i < fc->nbFields; i++)
//     {
//         requiredFieldNames.erase(fc->fields[i].name);
//     }
//     if (!requiredFieldNames.empty())
//     {
//         std::stringstream msg{};
//         msg << "PluginFieldCollection missing required fields: {";
//         char const* separator = "";
//         for (auto const& field : requiredFieldNames)
//         {
//             msg << separator << field;
//             separator = ", ";
//         }
//         msg << "}";
//         std::string msg_str = msg.str();
//         std::cout << msg_str << std::endl;
//     }
// }

} // namespace plugin
} // namespace nvinfer1
#endif // _PILLAR_SCATTER_COMMON_H_