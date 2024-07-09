

# Prepare Waymo Dataset!
- The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets.
- If you prepare another datasets such as KITTI, NuScenes, Lyft and Pandaset, please refer to the [GETTING_STARTED.md](../docs/GETTING_STARTED.md).
- If you want to use a custom dataset, Please refer to our [custom dataset template](CUSTOM_DATASET_TUTORIAL.md).

### Waymo Open Dataset
🔥 **You just do it on the `Host Env`, not in the `Container Env`.**
- Please download the official [Waymo Open Dataset](https://waymo.com/open/download/),
- If you download `waymo_open_dataset_v_1_4_3/archived_files` files, including the training data `training_0000.tar~training_0031.tar` and the validation data `validation_0000.tar~validation_0007.tar`.
  - Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):

```
CenterPointPillar
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional, old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
|   |   |── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0 (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_multiframe_-4_to_0.pkl (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0_global.np  (new, for single/multi-frame)

├── pcdet
├── tools
```

* If you want to download `waymo_open_dataset_v_1_4_3/individual_files` using `gsutil` as follows:
```shell
gsutil -m cp -r \
  "gs://waymo_open_dataset_v_1_4_3/individual_files/training" \
  "gs://waymo_open_dataset_v_1_4_3/individual_files/validation" \
  "gs://waymo_open_dataset_v_1_4_3/individual_files/testing" \
  "gs://waymo_open_dataset_v_1_4_3/individual_files/testing_3d_camera_only_detection" \
  "gs://waymo_open_dataset_v_1_4_3/individual_files/domain_adaptation" \
  .
```

- if you want only to do training and validation, just download `training` set and `validation` one, and merge to `raw_data` directory as follows: 
```shell
mkdir -p waymo_open_dataset_v_1_4_3/raw_data
cd waymo_open_dataset_v_1_4_3
gsutil -m cp -r \
  "gs://waymo_open_dataset_v_1_4_3/individual_files/training" \
  "gs://waymo_open_dataset_v_1_4_3/individual_files/validation" \
  .
cp -v training/* raw_data/
cp -v validation/* raw_data/
```

- For Waymo datasets, Install the official `waymo-open-dataset` by running the following command:
``` shell
docker exec -it centerpointpillar bash
pip install --upgrade pip
sudo apt install python3-testresources
pip install waymo-open-dataset-tf-2-12-0==1.6.4
```

- Extract point cloud data from tfrecord and generate data infos by running the following command
  - (it takes several hours, and you could refer to `data/waymo/waymo_processed_data_v0_5_0` to see how many records that have been processed):
``` shell
# only for single-frame setting: without 'elongation' in the 'used_feature_list'
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset_use_feature_no_elongation.yaml
    
# only for single-frame setting
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml

# for single-frame or multi-frame setting
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset_multiframe.yaml
# Ignore 'CUDA_ERROR_NO_DEVICE' error as this process does not require GPU.
```
Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics.

## [Return to the main page.](../README.md)






