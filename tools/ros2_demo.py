import _init_path
import rclpy
import rclpy.duration
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from tf_transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

import torch, numpy as np
import array
from pathlib import Path
from demo import DemoDataset, parse_config
# from pcdet.models import build_network, load_data_to_gpu
from object_detection.detectors3d import build_network
from general.utilities.data_utils import load_data_to_gpu
# from pcdet.utils import common_utils
from general.utilities import common_utils

DUMMY_FIELD_PREFIX = '__'
type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)

def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(
                ('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_to_nptype[f.datatype].itemsize * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray

    Reshapes the returned array to have shape (height, width), even if the
    height is 1.

    The reason for using np.frombuffer rather than struct.unpack is
    speed... especially for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (
            fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

class CenterpointNode(Node):
    def __init__(self):
        super().__init__('centerpoint_node')
        self.get_logger().info('CenterpointNode has been started')
        self.dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG,class_names=cfg.CLASS_NAMES,
                                   training=False, root_path=Path(args.data_path),
                                   ext=args.ext, logger=logger)
        self.iter = iter(self.dataset)
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.dataset)
        self.model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

        self.pc_sub = self.create_subscription(PointCloud2, '/lidar/top/pointcloud', self.pc_callback, 2)
        self.vis_pub = self.create_publisher(MarkerArray, '/boxes', 2)

    def pc_callback(self, msg: PointCloud2):
        start = self.get_clock().now()
        data = pointcloud2_to_array(msg)
        pc_ny = np.vstack([data['x'], data['y'], data['z'], np.zeros(data['x'].shape[0])]).T
        input_dict = self.dataset.data_processor.forward(data_dict={'points': pc_ny,
                                                                    'use_lead_xyz': True})
        with torch.no_grad():
            data_dict = self.dataset.collate_batch([input_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
        vis_msg = self.get_marker(pred_dicts)
        self.vis_pub.publish(vis_msg)
        end = self.get_clock().now()
        elapsed = end - start
        self.get_logger().info(f'Elapsed time: {elapsed.nanoseconds/1e6} ms')

    def get_marker(self, pred_dicts: list) -> MarkerArray:
        marker_array_msg = MarkerArray()
        ref_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        ref_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        ref_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

        for i, [box, score, label] in enumerate(zip(ref_boxes, ref_scores, ref_labels)):
            if score < 0.5:
                continue
            marker = Marker()
            marker.id = i
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(box[0])
            marker.pose.position.y = float(box[1])
            marker.pose.position.z = float(box[2])
            quat = quaternion_from_euler(0, 0, box[6])
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            marker.scale.x = float(box[3])
            marker.scale.y = float(box[4])
            marker.scale.z = float(box[5])
            if label == 1:
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
            elif label == 2:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)
            else:
                marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
            marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
            marker_array_msg.markers.append(marker)
        return marker_array_msg

    def np2ros(self, point_cloud: np.ndarray) -> PointCloud2:
        pc_msg = PointCloud2()
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        pc_msg.header.frame_id = 'map'
        pc_msg.height = 1
        pc_msg.width = point_cloud.shape[0]

        pc_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step = 16
        pc_msg.row_step = pc_msg.point_step * pc_msg.width

        memory_view = memoryview(point_cloud)
        if memory_view.nbytes > 0:
            array_bytes = memory_view.cast("B")
        else:
            # Casting raises a TypeError if the array has no elements
            array_bytes = b""
        as_array = array.array("B")
        as_array.frombytes(array_bytes)
        pc_msg.data = as_array

        return pc_msg


def main(args=None):
    rclpy.init(args=None)
    centerpoint_node = CenterpointNode()
    rclpy.spin(centerpoint_node)
    rclpy.shutdown()

if __name__ == '__main__':
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    main()

