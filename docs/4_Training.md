## Training a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters.

### 1. Train using Multi-GPUs
- Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_NUM}
# or
sh scripts/torch_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_NUM}
# or
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```
- If you use pytorch 1.x, you have to use `python -m torch.distributed.launch` i.e., `tools/scripts/dist_X.sh`
- If you use pytorch 2.x, you have to use `torchrun` i.e., `tools/scripts/torch_train_X.sh`
``` shell script
cd ~/CenterPointPillar
# you can link as `output` directory from `/Dataset/Train_Results/CenterPoint/` 
ln -s /Dataset/Train_Results/CenterPoint/ output   
cd tools/
sh scripts/torch_train.sh 2 --cfg_file ./cfgs/waymo_models/centerpoint_pillar_train.yaml --batch_size 24
```

### 2. Train using Single-GPU
* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```

``` shell
cd ~/CenterPointPillar
# you can link as `output` directory from `/Dataset/Train_Results/CenterPoint/`
ln -s /Dataset/Train_Results/CenterPoint/ output
cd tools/
CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file ./cfgs/waymo_models/centerpoint_pillar_train.yaml --batch_size 16  # you can replace `CUDA_VISIBLE_DEVICES=1` with gpu's number you want
```

### 3. Pretrained Models
- If you would like to train [CaDDN](../tools/cfgs/kitti_models/CaDDN.yaml), download the pretrained [DeepLabV3 model](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and place within the `checkpoints` directory. 
- Please make sure the [kornia](https://github.com/kornia/kornia) is installed since it is needed for `CaDDN`.
```
CenterPointPillar
├── checkpoints
│   ├── deeplabv3_resnet101_coco-586e9e4e.pth
├── data
├── pcdet
├── tools
```

## [Return to the main page.](../README.md)