#!/bin/bash

export PYTHONPATH="$(pwd):${PYTHONPATH}"
accelerate launch --config_file ./accelerate_configs/8_gpus_deepspeed_zero2.yaml \
  --main_process_port=8924 ./training/fine-tune_navsim.py
