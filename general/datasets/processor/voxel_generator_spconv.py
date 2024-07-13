import cumm.tensorview as tv
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator_spconv


class VoxelGeneratorSPConv():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        self._voxel_generator = VoxelGenerator_spconv(
            vsize_xyz=vsize_xyz,
            coors_range_xyz=coors_range_xyz,
            num_point_features=num_point_features,
            max_num_points_per_voxel=max_num_points_per_voxel,
            max_num_voxels=max_num_voxels
        )

    def generate(self, points):
        assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
        voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
        tv_voxels, tv_coordinates, tv_num_points = voxel_output
        # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points
