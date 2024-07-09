import _init_path
import argparse
import torch
import onnx
import onnxsim
import yaml
from pathlib import Path

from torch.ao.quantization import fuse_modules

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate
from modify_onnx import pillarscatter_surgeon

class DummyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return {}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/waymo_models/centerpoint_pillar_inference.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='../ckpts/waymo_iou_branch.pth', help='specify the pretrained model')
    parser.add_argument('--max_voxels', type=int, default='25000', help='specify max number of voxels')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def convert_onnx():
    logger = common_utils.create_logger()
    logger.info("------ Convert OpenPCDet model to ONNX ------")
    dataset = DummyDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=args.data_path, logger=logger
    )

    # Build the model
    cfg.MODEL.DENSE_HEAD.POST_PROCESSING.EXPORT_ONNX = True
    cfg.MODEL.POST_PROCESSING.EXPORT_ONNX = True
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                          dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # Fuse the ConvTranspose2d + BN layers
    for deblock in model.backbone_2d.deblocks:
        if deblock[0].__class__.__name__ == "ConvTranspose2d":
            fuse_modules(deblock, [['0', '1']], inplace=True)

    # Prepare the input
    with torch.no_grad():
        for item in cfg.DATA_CONFIG.DATA_PROCESSOR:
            if item.NAME == "transform_points_to_voxels":
                # max_voxels = item.MAX_NUMBER_OF_VOXELS['train']
                max_voxels = args.max_voxels
                max_points_per_voxel = item.MAX_POINTS_PER_VOXEL
        num_point_features = 4

        dummy_voxels = torch.zeros(
            (max_voxels, max_points_per_voxel, num_point_features),
            dtype=torch.float32, device='cuda'
        )
        dummy_voxel_num = torch.zeros((1,), dtype=torch.int32, device='cuda')
        dummy_voxel_idxs = torch.zeros((max_voxels, 4), dtype=torch.int32, device='cuda')

        # Export the model to ONNX
        dummy_input = ({'voxels': dummy_voxels,
                       'voxel_num_points': dummy_voxel_num,
                       'voxel_coords': dummy_voxel_idxs,
                       'batch_size': 1}, {})
        input_names = ['voxels', 'voxel_num', 'voxel_idxs']
        output_names = list(cfg.MODEL.DENSE_HEAD.SEPARATE_HEAD_CFG.HEAD_DICT.keys())+["score", "label"]
        torch.onnx.export(model,
                        dummy_input,
                        onnx_raw_path,
                        export_params=True,
                        opset_version=14,
                        do_constant_folding=True,
                        keep_initializers_as_inputs=True,
                        input_names=input_names,
                        output_names=output_names)
        onnx_raw = onnx.load(onnx_raw_path)
        onnx_sim, _ = onnxsim.simplify(onnx_raw)
        onnx.save(onnx_sim, onnx_sim_path)

    logger.info("Model exported to model.onnx")

def export_config():
    centerpoint_dict = {}
    centerpoint_dict['max_points'] = 200000
    centerpoint_dict['pub'] = "/centerpoint/boxes"
    centerpoint_dict['sub'] = "/lidar/concatenated/pointcloud"
    centerpoint_dict['score_threshold'] = 0.5

    voxelization_dict = {'min_range': {},
                         'max_range': {},
                         'voxel_size': {}}
    for item in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if item.NAME == "transform_points_to_voxels":
            voxelization_config = item
    voxelization_dict['min_range']['x'] = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[0]
    voxelization_dict['min_range']['y'] = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[1]
    voxelization_dict['min_range']['z'] = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[2]
    voxelization_dict['max_range']['x'] = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3]
    voxelization_dict['max_range']['y'] = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[4]
    voxelization_dict['max_range']['z'] = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[5]
    voxelization_dict['voxel_size']['x'] = voxelization_config.VOXEL_SIZE[0]
    voxelization_dict['voxel_size']['y'] = voxelization_config.VOXEL_SIZE[1]
    voxelization_dict['voxel_size']['z'] = voxelization_config.VOXEL_SIZE[2]
    voxelization_dict['max_voxels'] = args.max_voxels
    voxelization_dict['max_points_per_voxel'] = voxelization_config.MAX_POINTS_PER_VOXEL
    voxelization_dict['num_feature'] = len(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list)
    voxelization_dict['num_voxel_feature'] = 10

    postprocess_dict = {'nms': {}}
    postprocess_config = cfg.MODEL.DENSE_HEAD.POST_PROCESSING
    postprocess_dict['iou'] = postprocess_config.IOU_RECTIFIER
    postprocess_dict['nms']['pre_max'] = postprocess_config.NMS_CONFIG.NMS_PRE_MAXSIZE
    postprocess_dict['nms']['post_max'] = postprocess_config.NMS_CONFIG.NMS_POST_MAXSIZE
    postprocess_dict['nms']['iou_threshold'] = postprocess_config.NMS_CONFIG.NMS_THRESH
    postprocess_dict['out_size_factor'] = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    postprocess_dict['feature_x_size'] = round((voxelization_dict['max_range']['x'] - voxelization_dict['min_range']['x']) / voxelization_dict['voxel_size']['x'] / postprocess_dict['out_size_factor'])
    postprocess_dict['feature_y_size'] = round((voxelization_dict['max_range']['y'] - voxelization_dict['min_range']['y']) / voxelization_dict['voxel_size']['y'] / postprocess_dict['out_size_factor'])
    postprocess_dict['pillar_x_size'] = voxelization_dict['voxel_size']['x']
    postprocess_dict['pillar_y_size'] = voxelization_dict['voxel_size']['y']
    postprocess_dict['min_x_range'] = voxelization_dict['min_range']['x']
    postprocess_dict['min_y_range'] = voxelization_dict['min_range']['y']

    save_path = Path("../") / "config.yaml"
    cfg_output = {'centerpoint': centerpoint_dict,
                    'voxelization': voxelization_dict,
                    'postprocess': postprocess_dict}
    with open(save_path, 'w') as f:
        yaml.dump(cfg_output, f, default_flow_style=False)
    print(f"Config exported to {save_path}")

if __name__ == '__main__':
    args, cfg = parse_config()

    save_dir = Path("../onnx")
    onnx_raw_path = save_dir / "model_raw.onnx"
    onnx_sim_path = save_dir / "model_sim.onnx"
    onnx_path = save_dir / "model.onnx"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    if not onnx_sim_path.exists():
        print("Exporting model to ONNX...")
        convert_onnx()

    if not onnx_path.exists():
        print("Model optimization...")
        onnx_model = onnx.load(onnx_sim_path)
        modified_model = pillarscatter_surgeon(onnx_model)
        onnx.save(modified_model, onnx_path)
        print("Model exported to model.onnx")

    export_config()
