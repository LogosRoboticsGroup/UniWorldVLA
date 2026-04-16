#!/bin/bash
# source activate pwm
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_ALGO=ring
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_NSOCKS_PERTHREAD=4
# export CUDA_LAUNCH_BLOCKING=0  # 禁用同步调试
# export NCCL_ASYNC_ERROR_HANDLING=1
if [ -z "$LPAI_MASTER_0_HOST" ]; then
  export LPAI_CODE_DIR_1="/lpai/volumes/ad-agent-vol-ga/xhh/PWM-DA3"
  export EVAL_ONLY="1"
  export LOCAL_RUN_PWM="0"
  export RESUME_FROM_PRETRAIN="/lpai/volumes/ad-agent-vol-ga/lqq/PWM/Policy_World_Model/pre-training/ckpt_w_DFL"
  export EVAL_FROM_CHECKPOINT="0"
  export OUTPUT_DIR="/lpai/volumes/ad-agent-vol-ga/xhh/PWM-DA3/Local_8"
  export PYTHONPATH="$(pwd):${PYTHONPATH}"
  export MAX_TRAIN_STEPS=160000
  export BATCH_SIZE_TRAIN_NUS=2
  export BATCH_SIZE_VAL_NUS=7
  export LOG_EVERY=50
  export LPAI_ALL_NUM=1
  export LOAD_LX_TRAINING=0
  export EVAL_DIR="/lpai/volumes/ad-agent-vol-ga/xhh/PWM-DA3/Policy_World_Model/ckpt_sft_navsim"
fi
if [ "$LPAI_ALL_NUM" -eq 1 ]; then
  echo "========================>8_gpus_deepspeed_zero2.yaml"
  accelerate launch --config_file ./accelerate_configs/8_gpus_deepspeed_zero2.yaml \
    --main_process_port=8924 ./training/fine-tune_navsim.py
else
  echo "========================>auto_gpus_deepspeed_zero2.yaml"
  accelerate launch --config_file ./accelerate_configs/auto_gpus_deepspeed_zero2.yaml \
    --main_process_ip=${LPAI_MASTER_0_HOST} \
    --main_process_port=${LPAI_MASTER_0_PORT} \
    --machine_rank=${LPAI_RANK} \
    --num_processes=8 \
    ./training/fine-tune_navsim.py
fi

# export OUTPUT_DIR="/lpai/volumes/ad-agent-vol-ga/xhh/PWM-DA3/${LPAI_TASK_NAME}"
# export LPAI_CODE_DIR_1="/lpai/volumes/ad-agent-vol-ga/xhh/PWM-DA3"
# export EVAL_ONLY="0"
# export LOCAL_RUN_PWM="0"
# export RESUME_FROM_PRETRAIN="/lpai/volumes/ad-agent-vol-ga/lqq/PWM/Policy_World_Model/pre-training/ckpt_w_DFL/unwrapped_model/pytorch_model.bin"
# export EVAL_FROM_CHECKPOINT="0"
# export MAX_TRAIN_STEPS="160000"
# export BATCH_SIZE_TRAIN_NUS="5"
# export BATCH_SIZE_VAL_NUS="20"
# export LOG_EVERY="50"