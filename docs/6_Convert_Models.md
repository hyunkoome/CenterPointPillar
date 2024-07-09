## Convert ONNX model

- Convert ONNX model file from Pytorch 'pth' model file

``` shell
docker exec -it centerpointpillar bash

pip install onnx==1.16.0 onnxsim==0.4.36 onnxruntime

cd ~/CenterPointPillar/
mkdir onnx
cd ~/CenterPointPillar/tools
python export_onnx.py --cfg_file cfgs/waymo_models/centerpoint_pillar_inference.yaml --ckpt ../ckpt/checkpoint_epoch_24.pth

```
<img src="../sources/cmd_onnx.png" align="center" width="100%">

- As a result, create 3 onnx files on the `CenterPointPillar/onnx`
  - model_raw.onnx: pth를 onnx 로 변환한 순수 버전
  - model_sim.onnx: onnx 그래프 간단화해주는 라이브러리 사용한 버전
  - model.onnx: sim 모델을 gragh surgeon으로 수정한 최종 버전, tensorRT plugin 사용하려면 gragh surgeon이 필수임.

<img src="../sources/three_onnx_models.png" align="center" width="100%">

- Copy Onnx file to the `centerpoint/model` folder for `TensorRT` Process  
``` shell
cd ~/CenterPointPillar/
cp onnx/model.onnx centerpoint/model/
```

## [Return to the main page.](../README.md)
