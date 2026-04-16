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

# ============================================================
# 训练阶段控制 (环境变量 TRAINING_STAGE)
# ============================================================
# 可选值:
#   不设置     - 自动切换模式（使用配置文件中的 enable 设置）
#   stage1     - 只训练 stage1 (context)
#   stage2     - 只训练 stage2 (dynamic)
#   stage3     - 只训练 stage3 (context + dynamic)
#   stage4     - 只训练 stage4 (showo only)
#
# 示例:
#   export TRAINING_STAGE=stage1  # 只训练 context 模块
#   export TRAINING_STAGE=stage2  # 只训练 dynamic 模块
#   export TRAINING_STAGE=       # 自动切换
# ============================================================

export TRAINING_STAGE=""

# if [ -z "$LPAI_MASTER_0_HOST" ]; then
export LPAI_CODE_DIR_1="/lpai/volumes/ad-agent-vol-ga/xhh/PWM-DA3"
export EVAL_ONLY="0"
export LOCAL_RUN_PWM="0"
export RESUME_FROM_PRETRAIN="/lpai/volumes/ad-agent-vol-ga/xhh/PWM-DA3/pwm-xxh-train-cd481606/checkpoint_step_31721"
export EVAL_FROM_CHECKPOINT="0"
export OUTPUT_DIR="/lpai/volumes/ad-agent-vol-ga/xhh/PWM-DA3/Local_8"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export MAX_TRAIN_STEPS=160000
export BATCH_SIZE_TRAIN_NUS=6
export BATCH_SIZE_VAL_NUS=4
export LOG_EVERY=1
export LPAI_ALL_NUM=1
export LOAD_LX_TRAINING=1
export CUDA_LAUNCH_BLOCKING=1
export EVAL_DIR="/lpai/volumes/ad-agent-vol-ga/xhh/PWM-DA3/Policy_World_Model/ckpt_sft_navsim"
export RUN_FLASH_DATA_LOADER=0
LPAI_ALL_NUM=1
export CUDA_VISIBLE_DEVICES="3,7"
export TRAINING_STAGE="4"
# fi
if [ "$LPAI_ALL_NUM" -eq 1 ]; then
  echo "========================>8_gpus_deepspeed_zero2.yaml"
  NCCL_ALGO=Ring  NCCL_COMPRESS=0 ACCELERATE_DEBUG=1 PYTHONFAULTHANDLER=1 accelerate launch --config_file ./accelerate_configs/2_gpus_deepspeed_zero2.yaml \
    --main_process_port=8924 ./training/fine-tune_navsim.py
else
  echo "========================>auto_gpus_deepspeed_zero2.yaml"
  ACCELERATE_DEBUG=1 PYTHONFAULTHANDLER=1 accelerate launch --config_file ./accelerate_configs/auto_gpus_deepspeed_zero2.yaml \
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