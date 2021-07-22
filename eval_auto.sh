#!/bin/bash
# train_auto.sh
echo "script name: $0"
# "0,1,2"
echo "devices: $1"
# /home/liusheng/data/images4code/ai_model_ckpt/sjht/test_semantic_whyf_224/1.1/config_project.yaml
echo "config path: $2"
# /home/liusheng/data/images4code/ai_model_ckpt/sjht/test_semantic_whyf_224/1.1/model_and_temp_file/logs
echo "log path: $3"
#python3 data_preprocess.py --project_config_path "$2" --log_path "$3"
python3 eval_bisenetv2.py --project_config_path "$2" --log_path "$3"