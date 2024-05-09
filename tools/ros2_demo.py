import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf_transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

import torch, numpy as np
import array
from pathlib import Path
from demo import DemoDataset, parse_config
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class CenterpointNode(Node):
    def __init__(self):
        super().__init__('centerpoint_node')
        self.get_logger().info('CenterpointNode has been started')
        self.dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG,class_names=cfg.CLASS_NAMES,
                                   training=False, root_path=Path(args.data_path),
                                   ext=args.ext, logger=logger)
        self.iter = iter(self.dataset)
        self.get_logger().info(f'Total number of samples: \t{len(self.dataset)}')
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.dataset)
        self.model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

        self.pc_pub = self.create_publisher(PointCloud2, '/waymo', 10)
        self.vis_pub = self.create_publisher(MarkerArray, '/boxes', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
    
    def timer_callback(self):
        start = self.get_clock().now()
        with torch.no_grad():
            data_dict = next(self.iter, 0)
            if data_dict == 0:
                self.iter = iter(self.dataset)
                data_dict = next(self.iter)
            pc = data_dict['points']
            data_dict = self.dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)

        pc_msg = self.np2ros(pc)
        vis_msg = self.get_marker(pred_dicts)

        self.vis_pub.publish(vis_msg)
        self.pc_pub.publish(pc_msg)
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
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(box[0])
            marker.pose.position.y = float(box[1])
            marker.pose.position.z = float(box[2])
            quaternion = quaternion_from_euler(0, 0, box[6])
            # marker.pose.orientation.x = quaternion[0]
            # marker.pose.orientation.y = quaternion[1]
            # marker.pose.orientation.z = quaternion[2]
            # marker.pose.orientation.w = quaternion[3]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(box[3])
            marker.scale.y = float(box[4])
            marker.scale.z = float(box[5])
            marker.color.a = 0.0
            
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
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

