#!/bin/bash

# source activate pwm
# export PYTHONPATH=/path/to/root/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --config_file /lpai/volumes/ad-agent-vol-ga/wujunjie/project/Policy-World-Model/accelerate_configs/2_gpus_deepspeed_zero2.yaml \
  --main_process_port=8924 /lpai/volumes/ad-agent-vol-ga/wujunjie/project/Policy-World-Model/training/fine-tune_navsim.py
