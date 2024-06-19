import os 
from tools import cp
from glob import glob
import numpy as np

model_path = "{}/OpenPCDet/centerpoint/model/model.trt".format(os.path.expanduser('~'))
config_path = "{}/OpenPCDet/centerpoint/config/config.yaml".format(os.path.expanduser('~'))
centerpoint = cp.CenterPoint(config_path, model_path)

def getBox(box):
    x, y, z = box.x(), box.y(), box.z()
    l, w, h = box.l(), box.w(), box.h()
    yaw, score, cls = box.yaw(), box.score(), box.cls()

    return x, y, z, l, w, h, yaw, score, cls

pc_dir = sorted(glob("{}/OpenPCDet/data/waymo/waymo_processed_data_v0_5_0/segment-10335539493577748957_1372_870_1392_870_with_camera_labels/*.npy".format(os.path.expanduser('~'))))
for pc_path in pc_dir:
    pc = np.load(pc_path)[:,:4]
    pc[:,3] = np.tanh(pc[:,3])
    print(pc.shape)
    print(pc_path)
    boxes = centerpoint.forward(pc)
    for box in boxes:
        x, y, z, l, w, h, yaw, score, cls = getBox(box)
        