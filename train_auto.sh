#!bin/bash
echo "script name: $0"
echo "devices: $1"
echo "gpu num: $2"
echo "config path: $3"
echo "log path: $4"
# 暂时有个bug， 最后一个参数shell解析出来会多一个/r 不知道为什么， 所以添加一个无用参数
python3 data_preprocess.py --project_config_path "$3" --log_path "$4" --local_rank 0
CUDA_VISIBLE_DEVICES="$1" python3 -m torch.distributed.launch --nproc_per_node="$2" train_bisenetv2.py --project_config_path "$3" --log_path "$4"