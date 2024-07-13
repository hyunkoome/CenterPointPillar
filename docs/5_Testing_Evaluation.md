## Testing a model and Evaluation

### 1. Test and evaluate the pretrained `pytorch` models
* Test with a pretrained model:
```shell script
python object_detection/test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```
- for examples
```shell script
python object_detection/test.py --cfg_file tools/cfgs/waymo_models/centerpoint_pillar_train_refactoring.yaml --ckpt /home/hyunkoo/DATA/HDD8TB/LiDAR/CenterPointPillarLocal/ckpt/checkpoint_epoch_24.pth
python object_detection/test.py --cfg_file tools/cfgs/waymo_models/centerpoint_pillar_train_refactoring.yaml --ckpt checkpoint_epoch_24.pth --ckpt_dir /home/hyunkoo/DATA/HDD8TB/LiDAR/CenterPointPillarLocal/ckpt
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument:
```shell script
python object_detection/test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```
- for examples
```shell script
python object_detection/test.py --cfg_file tools/cfgs/waymo_models/centerpoint_pillar_train_refactoring.yaml --ckpt_dir /home/hyunkoo/DATA/HDD8TB/LiDAR/CenterPointPillarLocal/ckpt --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

### 2. Evaluation with a pytorch model
``` shell
docker exec -it centerpoint bash
cd ~/CenterPointPillar
python object_detection/test.py --cfg_file tools/cfgs/waymo_models/centerpoint_pillar_inference.yaml --ckpt /home/lidar/CenterPointPillar/ckpt/checkpoint_epoch_24.pth
```

- Results as shown:
```
2024-07-08 07:59:21,802   INFO  
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP: 0.6204 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH: 0.6137 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APL: 0.6204 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP: 0.5417 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH: 0.5358 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APL: 0.5417 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP: 0.5329 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH: 0.2887 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APL: 0.5329 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP: 0.4553 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH: 0.2468 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APL: 0.4553 
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/AP: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APH: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APL: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/AP: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APH: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APL: 0.0000 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP: 0.3267 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH: 0.2730 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APL: 0.3267 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP: 0.3141 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH: 0.2625 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APL: 0.3141 
```

- If you set `test: 25000` of `MAX_NUMBER_OF_VOXELS` at the `cfgs/waymo_models/centerpoint_pillar_inference.yaml` like TensorRT (`centerpoint/config.yaml`),
- You can get more similar results as shown:
```
2024-07-08 09:57:04,120   INFO  
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP: 0.6199 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH: 0.6132 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APL: 0.6199 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP: 0.5413 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH: 0.5353 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APL: 0.5413 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP: 0.5327 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH: 0.2885 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APL: 0.5327 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP: 0.4552 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH: 0.2466 
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APL: 0.4552 
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/AP: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APH: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APL: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/AP: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APH: 0.0000 
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APL: 0.0000 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP: 0.3262 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH: 0.2729 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APL: 0.3262 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP: 0.3137 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH: 0.2625 
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APL: 0.3137
```

## [Return to the main page.](../README.md)