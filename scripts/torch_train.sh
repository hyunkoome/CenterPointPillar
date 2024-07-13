#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} object_detection/train.py --launcher pytorch ${PY_ARGS}

# torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:46342 object_detection/train.py --launcher pytorch --cfg_file tools/cfgs/waymo_models/centerpoint_pillar_train_refactoring.yaml