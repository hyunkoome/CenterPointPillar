# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import numpy as np
import onnx_graphsurgeon as gs

@gs.Graph.register()
def replace_with_clip(self, inputs, outputs, grid_y_size, grid_x_size):
    for inp in inputs:
        inp.outputs.clear()

    for out in outputs:
        out.inputs.clear()

    op_attrs = dict()
    op_attrs["dense_shape"] = np.array([grid_y_size, grid_x_size])

    return self.layer(name="PPScatter_0", op="PPScatterPlugin",
                      inputs=inputs, outputs=outputs, attrs=op_attrs)

def pillarscatter_surgeon(onnx_model):
    graph = gs.import_onnx(onnx_model)
    grapth_tensors = graph.tensors()

    # Input shapes
    voxel_idxs_shape = grapth_tensors["voxel_idxs"].shape
    voxel_num_shape = grapth_tensors["voxel_num"].shape

    # Pillar Feature Net Input
    input_new_node = [node for node in graph.nodes if node.op == "MatMul"][0]
    input_new_shape = input_new_node.inputs[0].shape

    # PillarScatter Input
    voxels = gs.Variable(name="voxels", dtype=np.float32, shape=input_new_shape)
    voxels_idxs = gs.Variable(name="voxel_idxs", dtype=np.int32, shape=voxel_idxs_shape)
    voxels_num = gs.Variable(name="voxel_num", dtype=np.int32, shape=voxel_num_shape)

    pillar_feature_input = [node for node in graph.nodes if node.name == '/vfe/pfn_layers.1/ReduceMax'][0]
    pillar_feature_input.attrs["keepdims"] = 0
    output_tensor = graph.tensors()[pillar_feature_input.outputs[0].name]
    output_tensor.shape.remove(1)

    # Graph surgery
    conv_op = [node for node in graph.nodes if node.op == "Conv"][0]
    # graph.inputs.append(voxels_num)
    inputs = [pillar_feature_input.outputs[0], voxels_idxs, voxels_num]
    outputs = [conv_op.inputs[0]]
    grid_y_size , grid_x_size = conv_op.inputs[0].shape[2:]

    graph.replace_with_clip(inputs, outputs, grid_y_size, grid_x_size)
    graph.cleanup().toposort()

    graph.inputs = [voxels, voxels_idxs, voxels_num]
    input_new_node.inputs[0] = voxels
    graph.outputs = [grapth_tensors[output.name] for output in graph.outputs]

    graph.cleanup().toposort()
    modified_model = gs.export_onnx(graph)

    return modified_model
