# limitations under the License.
# Copyright 2024 HuggingFace, NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import moviepy
from moviepy.editor import ImageSequenceClip
import imageio
import tempfile
import os
import warnings

# from data_utils.sequence_visualizer import plot_attention_mask
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path  # coding=utf-8

from typing import Union
import copy
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
import random
import sys

from models.video_metric import Evaluator
from torch.optim import AdamW
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from models.video_metric import FeatureStats
from data_utils.pwm_dataset import DatasetNavsim
from models import Showo
from models.modeling_showo import get_vq_model_class
from models.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_t2d, create_attention_mask_for_nusc
from models.lr_schedulers import get_scheduler
from models.depth_encoder import DepthEncoder
from models.logging import set_verbosity_info, set_verbosity_error
from torch.utils.data import DataLoader,Subset
from torch.utils.data.distributed import DistributedSampler

from training.utils import flatten_omega_conf, AverageMeter
from training.staged_training_manager import StagedTrainingManager
SYSTEM_PROMPT_LEN = 28
from navsim.visualization.camera import visualize_pred_gt_camera_traj
from models.depth_encoder import DepthEncoder
# 导入重构后的token处理器
from data_utils.prepare_input_ids import InputIDsPreparer
from navsim.pdsm_test_utils import PDSM_eval, pdsm_score_process
# 导入可视化和评测相关函数
from training.visualization_and_evaluation import (
    img_token2pixel, depth_token2pixel, video_concate, visualize_predictions,
    save_as_webm, video_display_and_visualization, depth_visualization_draw,
    depth_display_and_visualization,
    batch_forward, process_images, video_metrics_process, depth_metrics_process,
    video_metrics_evaluation, depth_metrics_evaluation, action_metrics_evaluation
)

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) and 'mm_projector' not in name.split('.'):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1]) #unique linear layer name

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    if 'embed_tokens' in lora_module_names:
        lora_module_names.remove('embed_tokens')
    return list(lora_module_names)

def configure_experiment_from_env(config):
    """从环境变量配置实验参数
    
    参数:
        config: OmegaConf配置对象
        
    返回:
        修改后的config对象
    """
    config.experiment.eval_only = bool(int(os.getenv("EVAL_ONLY", 0)))
    local_test = bool(int(os.getenv("LOCAL_RUN_PWM", 0)))
    if not local_test:
        config.model.showo.resume_from_pretrain = os.getenv("RESUME_FROM_PRETRAIN", "")
        config.experiment.eval_from_checkpoint = bool(
            int(os.getenv("EVAL_FROM_CHECKPOINT", 0))
        )
        config.experiment.output_dir = os.getenv("OUTPUT_DIR", "")

        # 新增训练相关环境变量配置
        config.training.max_train_steps = int(
            os.getenv("MAX_TRAIN_STEPS", config.training.max_train_steps)
        )
        config.training.batch_size_train_nus = int(
            os.getenv("BATCH_SIZE_TRAIN_NUS", config.training.batch_size_train_nus)
        )
        config.training.batch_size_val_nus = int(
            os.getenv("BATCH_SIZE_VAL_NUS", config.training.batch_size_val_nus)
        )
        config.experiment.log_every = int(
            os.getenv("LOG_EVERY", config.experiment.log_every)
        )
        config.model.showo.load_lx_train = bool(
            int(os.getenv("LOAD_LX_TRAINING", config.model.showo.load_lx_train))
        )
        config.experiment.eval.eval_dir = os.getenv("EVAL_DIR", config.experiment.eval.eval_dir)
        
        # 从环境变量读取训练超参数
        config.optimizer.params.learning_rate = float(
            os.getenv("LEARNING_RATE", config.optimizer.params.learning_rate)
        )
        config.training.max_grad_norm = float(
            os.getenv("MAX_GRAD_NORM", config.training.max_grad_norm)
        )
        config.experiment.nfp_loss.depth_coffe = float(
            os.getenv("DEPTH_COFFE", config.experiment.nfp_loss.depth_coffe)
        )
        config.experiment.nfp_loss.frame_coffe = float(
            os.getenv("FRAME_COFFE", config.experiment.nfp_loss.frame_coffe)
        )
        config.training.video_coeff = float(
            os.getenv("VIDEO_COEFF", config.training.video_coeff)
        )
        config.training.tj_coeff = float(
            os.getenv("TJ_COEFF", config.training.tj_coeff)
        )
        config.training.eval_start_epoch = int(
            os.getenv("EVAL_START_EPOCH", config.training.eval_start_epoch)
        )
        config.training.use_residual = bool(
            os.getenv("USE_RESIDUAL", config.training.use_residual)
        )
    else:
        config.training.batch_size_train_nus = 1
        config.training.batch_size_val_nus = 3

    return config, local_test

def setup_accelerator_and_logging():
    """初始化加速器和日志配置
    
    参数:
        config: OmegaConf配置对象
        
    返回:
        tuple: (accelerator对象, local_test标志, config, total_batch_size_per_gpu)
    """
    config_path = "configs/sft_navsim/navsim.yaml"
    config = OmegaConf.load(config_path)
    config.worker = OmegaConf.load(config.worker)
    # Enable TF32 on Ampere GPU
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config, local_test = configure_experiment_from_env(config)
    log_dir = os.path.join(config.experiment.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=log_dir,
        split_batches=True,
    )
    # 核心：打印节点相关关键信息
    print("="*50)
    print(f"✅ 当前节点rank（machine_rank）: {accelerator.process_index}")  # 当前节点序号
    print(f"✅ 单节点进程数（num_processes per node）: {accelerator.num_processes}")  # 每节点GPU数
    time_str = ""
    if accelerator.is_local_main_process:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir,f"{time_str}_train.log") if accelerator.is_local_main_process else None
    if accelerator.is_local_main_process:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "%m/%d/%Y %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.getLogger().addHandler(stream_handler)

    os.environ["WANDB_MODE"] = os.getenv("WANDB_MODE", "offline")  # 可通过环境变量 WANDB_MODE=online 覆盖，默认离线模式
    total_batch_size_per_gpu = config.training.batch_size_train_nus #must have context frame tokens
    total_batch_size = config.training.batch_size_train_nus * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = total_batch_size_per_gpu
    if accelerator.mixed_precision == "fp16":
        images_feat_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        images_feat_dtype = torch.bfloat16
    else:
        images_feat_dtype = torch.float32
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # Initialize trackers and store configuration
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.output_dir,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.eval_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    return accelerator, local_test, config, total_batch_size_per_gpu

def setup_models_and_optimizers(config, accelerator,val_dataloader_navsim):
    """初始化模型、优化器和学习率调度器
    
    参数:
        config: OmegaConf配置对象
        accelerator: 加速器对象
        
    返回:
        tuple: (model, optimizer, lr_scheduler, uni_prompting)
    """

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    # 初始化tokenizer
    logger.info("Loading models and optimizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.showo.llm_model_path, padding_side="left"
    )

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                      special_tokens=(
                                          "<|soi|>", "<|eoi|>", "<|sod|>", "<|eod|>", "<|t2i|>",
                                          "<|mmu|>", "<|t2d|>", "<|act|>", "<|lvg|>"
                                      ),
                                      ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob,
                                      skip_len=config.model.showo.vocab_size)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # Initialize model
    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(accelerator.device)
        if config.model.showo.vocab_size != model.vocab_size:
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = model.config.vocab_size - 1
            model.mask_token_id = model.config.vocab_size - 1
    else:
        model = Showo(**config.model.showo).to(accelerator.device)

    # embedding expand
    if config.model.showo.dynamic_size:
        dynamic_size = config.model.showo.dynamic_size
        model.resize_dynamic_size(dynamic_size, 'sft', config)
        evaluator = Evaluator(config.model.eval.i3d_path, max_batchsize=config.training.batch_size_val_nus)
        evaluator.eval()
        evaluator.requires_grad_(False)
        vq_name = get_vq_model_class(config.model.vq_model.type)
        vq_model = vq_name(config_exps=config,
                          num_vq_embeddings=config.model.vq_model.num_vq_embeddings,
                          num_dyn_embeddings=config.model.vq_model.num_dyn_embeddings)

        if config.model.vq_model.get("pretrained_model_path", None):
            from safetensors.torch import load_file
            state_dict = load_file(config.model.vq_model.pretrained_model_path)  # ['model']
            vq_model.load_state_dict(state_dict, strict=True)

        vq_model.eval()
        vq_model.requires_grad_(False)

    if not config.experiment.eval_only:
        load_path = config.model.showo.resume_from_pretrain
        strict = False
    else:
        load_path = config.experiment.eval.eval_dir
        strict = False
     # 初始化 depth encoder（从 fine-tune_navsim.py 迁移到 modeling_showo.py）
    vq_model = vq_model.to(accelerator.device)
    evaluator = evaluator.to(accelerator.device)
    model.init_depth_encoder(
        vq_model=vq_model,
        condition_len=config.dataset.ctd.context_length,
        uni_prompting=uni_prompting,
        device=accelerator.device,
        args=config.dataset.depth_cache,
        data_loader=val_dataloader_navsim.dataset.dataset if hasattr(val_dataloader_navsim.dataset, 'dataset') else val_dataloader_navsim.dataset,
        contex_norm=getattr(config.dataset.ctd, "c_resolution",  [128,224]),
        dynamic_norm=getattr(config.dataset.ctd, "d_resolution",  [256,448]),
        config=config,
        # mask_dtype=mask_dtype
    )
    load_path = os.path.join(load_path, "unwrapped_model", "pytorch_model.bin") 
    print(f"load from ckpt:{load_path}")
    # if  config.model.showo.load_lx_train:
    #     model.resize_dynamic_size(2, "sft", config)
    state_dict = torch.load(load_path, map_location="cpu")   
    # state_dict = torch.load(load_path)   
    model.load_state_dict(state_dict, strict=strict)
    del state_dict
    # if not config.model.showo.load_lx_train:
    #     model.resize_dynamic_size(2, "sft", config)

    ##################################
    #   Optimizer and LR scheduler   #
    ##################################
    optimizer_config = config.optimizer.params
    if config.training.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32*2,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.01,
            bias= "none",
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

   
    # 初始化分阶段训练管理器
    staged_training_manager = StagedTrainingManager(
        config=config,
        model=model,
        accelerator=accelerator
    )

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                      p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                      p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


    # Debug: 检查参数组是否为空
    logger.info("="*80)
    logger.info(f"🔧 [DEBUG] 优化器参数组检查（在冻结设置之后）")
    total_trainable = 0
    for i, group in enumerate(optimizer_grouped_parameters):
        param_count = len(group["params"])
        numel_count = sum(p.numel() for p in group["params"])
        total_trainable += numel_count
        logger.info(f"   组 {i+1}: {param_count} 个参数, {numel_count:,} 个数值")
        
        # 检查参数的 device 和 dtype
        if param_count > 0:
            first_param = group["params"][0]
            logger.info(f"           第一个参数: device={first_param.device}, dtype={first_param.dtype}, shape={first_param.shape}")
    
    if total_trainable == 0:
        logger.error("❌ 警告：优化器没有找到任何可训练参数！")
        logger.info("   检查当前模型的可训练参数：")
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        total_count = sum(1 for _ in model.parameters())
        logger.info(f"   模型总参数数: {total_count}")
        logger.info(f"   可训练参数数: {trainable_count}")
        
        # 列出所有被冻结的参数
        logger.info("   ❌ 以下参数被冻结:")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                logger.info(f"      {name}: {param.shape}, numel={param.numel():,}")
    else:
        logger.info(f"✅ 优化器将优化 {total_trainable:,} 个参数")
    logger.info("="*80)
    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        base_lr=optimizer_config.learning_rate,
    )
    
    logger.info("Preparing model, optimizer and dataloaders")
    
    
    
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    return model, optimizer, lr_scheduler, uni_prompting, vq_model, evaluator, staged_training_manager


def create_dataloaders(config, accelerator, local_test):
    """创建训练和验证数据加载器
    
    参数:
        config: OmegaConf配置对象
        accelerator: 加速器对象
        local_test: 是否本地测试模式
        
    返回:
        tuple: (train_dataloader, val_dataloader, num_update_steps_per_epoch)
    """
    logger.info("Creating dataloaders")
    dataset_config = config.dataset.params

    if config.dataset.dataset_use == "sft_navsim":
        total_batch_size_without_accum = config.training.batch_size_train_nus * accelerator.num_processes
        total_batch_size = (total_batch_size_without_accum * config.training.gradient_accumulation_steps)

        # 创建训练数据集
        if not config.experiment.eval_only:
            cur_loader_type = (
                "test" if bool(int(os.getenv("LOCAL_RUN_PWM", 0))) else "train"
            )
            full_dataset_navsim_train = DatasetNavsim(
                config=config, split=cur_loader_type, aug_enable=False
            )
            if local_test:
                dataset_navsim_train = Subset(
                    full_dataset_navsim_train,
                    range(min(200, len(full_dataset_navsim_train))),
                )
            else:
                dataset_navsim_train = full_dataset_navsim_train
        else:
            full_dataset_navsim_train = None
            dataset_navsim_train = None

        # 创建验证数据集
        cur_loader_type = (
            "train" if bool(int(os.getenv("RUN_FLASH_DATA_LOADER", 0))) else "test"
        )
        full_dataset_navsim_val = DatasetNavsim(config=config, split=cur_loader_type, aug_enable=False)
        dataset_navsim_val = (
            Subset(
                full_dataset_navsim_val, range(min(200, len(full_dataset_navsim_val)))
            )
            if local_test
            else full_dataset_navsim_val
        )

        print('process index : ',
            accelerator.process_index, ', total_gpus:', accelerator.num_processes,
            "Length of dataset_train:", len(dataset_navsim_train) if dataset_navsim_train else [], "samples",
            "Length of dataset_val:", len(dataset_navsim_val), "samples")

        # 配置分布式采样器
        if accelerator.num_processes > 1:
            sampler_nusc = (
                DistributedSampler(
                    dataset_navsim_train,
                    num_replicas=accelerator.num_processes,
                    rank=accelerator.process_index,
                    shuffle=True,
                    seed=config.training.seed
                ) if not config.experiment.eval_only else None
            )
            sampler_nusc_val = DistributedSampler(
                dataset_navsim_val,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
                seed=config.training.seed
            )
            shuffle_train = False
            shuffle_val = False
        else:
            sampler_nusc = None
            sampler_nusc_val = None
            shuffle_train = True
            shuffle_val = False

        # 创建数据加载器
        if not config.experiment.eval_only:
            train_dataloader_navsim = DataLoader(
                dataset_navsim_train,
                batch_size=config.training.batch_size_train_nus,
                sampler=sampler_nusc,
                collate_fn=(
                    dataset_navsim_train.dataset.collate_fn
                    if hasattr(dataset_navsim_train, "dataset")
                    else dataset_navsim_train.collate_fn
                ),
                shuffle=shuffle_train,
                num_workers=dataset_config.num_workers,
                prefetch_factor=2 if dataset_config.num_workers > 0 else None,
            )
            num_update_steps_per_epoch = math.ceil(len(dataset_navsim_train) / total_batch_size)
        else:
            train_dataloader_navsim = None
            num_update_steps_per_epoch = 0

        val_dataloader_navsim = DataLoader(
            dataset_navsim_val,
            batch_size=config.training.batch_size_val_nus,
            sampler=sampler_nusc_val,
            collate_fn=(
                dataset_navsim_val.dataset.collate_fn
                if hasattr(dataset_navsim_val, "dataset")
                else dataset_navsim_val.collate_fn
            ),
            shuffle=shuffle_val,
            num_workers=dataset_config.num_workers,
            prefetch_factor=2 if dataset_config.num_workers > 0 else None,
        )

        return (
            train_dataloader_navsim,
            val_dataloader_navsim,
            num_update_steps_per_epoch,
            total_batch_size,
        )
    else:
        raise ValueError(f"Unsupported dataset")


def main():
    # torch.cuda.empty_cache()
    #########################
    # SETUP Accelerator     #
    #########################

    # 初始化加速器和日志
    accelerator, local_test, config, total_batch_size_per_gpu = setup_accelerator_and_logging()
 
    # 调用封装好的函数创建数据加载器
    (
        train_dataloader_navsim,
        val_dataloader_navsim,
        num_update_steps_per_epoch,
        total_batch_size,
    ) = create_dataloaders(config, accelerator, local_test)
    # 调用封装好的函数初始化模型、优化器和学习率调度器
    # 模型在dataer loaer 之前还是还是之后
    model, optimizer, lr_scheduler, uni_prompting, vq_model, evaluator,staged_training_manager = setup_models_and_optimizers(
        config, accelerator,val_dataloader_navsim
    )

    global_step = 0
    first_epoch = 0
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    
    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype
 
    
    # 实例化输入ID准备器

    # 实例化输入ID准备器
    input_ids_preparer = InputIDsPreparer(uni_prompting, vq_model, model.depth_encoder, accelerator)

    ####################
    #     Training     #
    ####################

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    name = [n for n, p in model.named_parameters() if p.requires_grad]
    num_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    # print("Parameters require gradients: {}".format(name))
    print("Num of Parameters require gradients: {}M".format(num_params / 1e6))

    end = time.time()
    min_mean_fvd = 1000
    if config.experiment.eval_only:
        try:
            eval_logs = evaluate(model,
                                vq_model,
                                config,
                                mask_dtype,
                                accelerator,
                                global_step,
                                uni_prompting,
                                val_dataloader_navsim,
                                evaluator,
                                input_ids_preparer)
            return eval_logs
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise
    num_train_epochs = num_update_steps_per_epoch

    # 执行训练循环
    global_step = training_loop(model, vq_model, config, mask_dtype, accelerator, global_step, uni_prompting,
                               train_dataloader_navsim, val_dataloader_navsim, evaluator, input_ids_preparer,
                               optimizer, lr_scheduler, first_epoch, num_train_epochs, total_batch_size_per_gpu,
                               batch_time_m, data_time_m, end, depth_encoder=model.depth_encoder if hasattr(model, 'depth_encoder') else None,staged_training_manager=staged_training_manager)
    accelerator.wait_for_everyone()
    accelerator.end_training()


@torch.no_grad()
def evaluate(model,
             vq_model,
             config,
             mask_dtype,
             accelerator,
             global_step,
             uni_prompting,
             eval_dataloader,
             evaluator,
             input_ids_preparer):

    model.eval()
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
    
    losses = []
    future_seconds = 4
    scoring_params = OmegaConf.load(config.dataset.scoring_path)
    cache_path = os.path.join(config.experiment.base_root, 'dataset/navsim/cache', config.dataset.scene_filter.test_cache)
    score_rows = []
    mse_values, psnr_values, ssim_values, lpips_values, fvds, desc_a_pair, action_pair, fvd = [], [], [], [], [], [], [], None
    real_feats, gen_feats = FeatureStats(capture_mean_cov=True), FeatureStats(capture_mean_cov=True)
    eval_iters = min(len(eval_dataloader), config.experiment.eval.max_eval_iters)
    bar = tqdm(range(eval_iters), desc="validation", disable=not accelerator.is_local_main_process)
    logger.info("validation PDMS...")
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    num_visual = 0
    local_test=bool(int(os.getenv("LOCAL_RUN_PWM", 0)))
    test_len = int(os.getenv("TEST_LEN", 0))
    # if not local_test:                
    dynamic_tok_num = 30 
    # else:
    #     dynamic_tok_num = test_len + 30
    output_root = config.experiment.output_dir
    os.makedirs(output_root, exist_ok=True)
    final_i = 0  # 用于记录最后的迭代次数

    # 初始化深度指标列表
    depth_mse_values = []
    depth_abs_rel_values = []
    depth_rmse_values = []

    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            final_i = i  # 更新最后的迭代次数

            if i == config.experiment.eval.max_eval_iters:
                break
            
            if bool(int(os.getenv("RUN_FLASH_DATA_LOADER", 0))):
                input_ids_preparer.prepare_one_batch_test(
                batch, config, mask_dtype, create_attention_mask_for_nusc
                )
                continue
            
            try:
                input_ids, labels, image_tokens_ori, attention_mask, depth_feat = input_ids_preparer.prepare_one_batch_test(
                    batch, config, mask_dtype, create_attention_mask_for_nusc
                )      
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
            # 从batch中提取需要的变量
            future_trajectories = batch['future_trajectory'].to(accelerator.device)
            ego_status = batch['ego_status'].to(accelerator.device) if config.experiment.add_ego else None
            next_img_dynamic = batch['next_img_dynamic'].to(accelerator.device)
            token = batch['token']
            token_encode = torch.tensor([tok.encode("utf-8") for tok in token], device=accelerator.device, dtype=torch.uint8)

            sod_input_idx = torch.where(input_ids == uni_prompting.sptids_dict['<|sod|>'].to(input_ids.device))[1].unique()
            eod_input_idx = torch.where(input_ids == uni_prompting.sptids_dict['<|eod|>'].to(input_ids.device))[1].unique()
            eot_input_idx = torch.where(input_ids == uni_prompting.sptids_dict['<|eot|>'].to(input_ids.device))[1].unique()
            action_len = future_trajectories.shape[1]
            context_length = config.dataset.ctd.context_length
            # add cmd token
            input_embed = model.showo.model.embed_tokens(input_ids[:, :eot_input_idx[-1]+dynamic_tok_num+1])
            if ego_status is not None:
                ego_token = model.ego_forward(ego_status.to(input_embed.dtype))
                input_embed[:, eot_input_idx[0]-1, :] = ego_token
            # check(input_embed,"input_embed")
            input_attention_mask = attention_mask[:, :, :eot_input_idx[-1]+dynamic_tok_num+1, :eot_input_idx[-1]+dynamic_tok_num+1]
            # plot_attention_mask(input_attention_mask,save_path=f"attention_mask_{i}.png")
            time_0 = time.time()
            with torch.autocast("cuda", dtype=torch.float32, enabled=accelerator.mixed_precision != "no"):
                gen_image_token_ids, gen_trj = accelerator.unwrap_model(model).navsim_alt_gen(input_embed=input_embed,
                                            attention_mask=input_attention_mask,
                                            config=config,
                                            action_len=action_len,
                                            uni_prompting=uni_prompting,
                                            ego_status=ego_status,
                                            depth_embeddings=depth_feat,
                                            input_ids=input_ids,
                                            )
                # check(gen_image_token_ids, "gen_image_token_ids")
                # check(gen_trj, "gen_trj")

            batch_size = batch["next_img_context"].shape[0]
            #####################################
            # ---------video metrics------------#
            #####################################
            predicted_images, recons_images, pixel_values, mse_values, psnr_values, ssim_values, lpips_values, fvds, real_feats, gen_feats,fvd = video_metrics_evaluation(
                config, accelerator, evaluator, uni_prompting, vq_model,
                image_tokens_ori, next_img_dynamic, gen_image_token_ids,
                batch_size, mse_values, psnr_values, ssim_values,
                lpips_values, fvds, real_feats, gen_feats, time_0,
                batch["next_img_context"].to(accelerator.device),context_length
            )

            score_rows = action_metrics_evaluation(config, accelerator, gen_trj, token_encode, scoring_params, cache_path, score_rows, logger,output_root,eval_dataloader)
            num_visual = video_display_and_visualization(accelerator, num_visual, pixel_values, recons_images, predicted_images,
                                                        context_length, output_root, global_step, token, batch_size, i)
            bar.update(1)
    eval_logs = {}  # 初始化 eval_logs
    if accelerator.is_main_process and not bool(int(os.getenv("RUN_FLASH_DATA_LOADER", 0))):
        # video val
        # if len(fvds) > 0:
        #     fvd = torch.cat(fvds, 0).mean().item() if isinstance(fvds[0], torch.Tensor) else fvds[0]
        # else:
        #     fvd = 0.0
        # fvd_mean = fvd if isinstance(fvd, (int, float)) else (torch.mean(fvd).item() if isinstance(fvd, torch.Tensor) else 0.0)
        eval_logs = video_metrics_process(config, mse_values, fvd, psnr_values, ssim_values, lpips_values)
        accelerator.log(eval_logs, step=global_step+final_i)

        # traj test
        trj_logs = pdsm_score_process(config, score_rows, global_step, logger)
        trj_logs = {f"PDMS/{k}":v for k,v in trj_logs.items() if k not in ['token','valid']}
        accelerator.log(trj_logs, step=global_step+final_i)
    model.train()
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
  
    logger.info("validation finished...")
    if accelerator.is_main_process:
        return eval_logs
    else:
        return None

def training_loop(model, vq_model, config, mask_dtype, accelerator, global_step, uni_prompting,
                 train_dataloader_navsim, val_dataloader_navsim, evaluator, input_ids_preparer,
                 optimizer, lr_scheduler, first_epoch, num_train_epochs, total_batch_size_per_gpu,
                 batch_time_m, data_time_m, end, depth_encoder=None,staged_training_manager=None):
    """
    执行训练循环，包含前向传播、反向传播、日志记录、检查点保存和评估
    
    Args:
        model: 模型
        vq_model: VQ模型
        config: 配置对象
        mask_dtype: 掩码数据类型
        accelerator: 加速器对象
        global_step: 全局步数
        uni_prompting: 统一提示对象
        train_dataloader_navsim: 训练数据加载器
        val_dataloader_navsim: 验证数据加载器
        evaluator: 评估器
        input_ids_preparer: 输入ID准备器
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        first_epoch: 第一个epoch
        num_train_epochs: 训练epoch数
        total_batch_size_per_gpu: 每个GPU的批次大小
        batch_time_m: 批次时间平均计
        data_time_m: 数据时间平均计
        end: 结束时间
    
    Returns:
        int: 更新后的 global_step
    """
    cur_epoch = -1
    # 监控 step_loss_tj 连续小于阈值的计数器
    loss_tj_below_threshold_count = 0
    loss_tj_threshold = 0.2
    loss_tj_target_count = 20
    showo_trainable_enabled = False  # 标记是否已经开启 showo 可训练

    

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()
        for batch_idx, batch in enumerate(train_dataloader_navsim):
            try:
                input_ids, labels, image_tokens_ori, attention_mask,depth_features_dict = input_ids_preparer.prepare_one_batch_train(
                    batch, config, data_time_m, end, model, mask_dtype, create_attention_mask_for_nusc
                )
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
                
            action_len = batch['future_trajectory'].shape[1]
            with (accelerator.accumulate(model)):
                logits, loss_video, loss_tj, mmu_index, eod_img_d = model.navsim_forward(
                    inputs=input_ids,
                    input_embeddings=None,
                    attention_mask=attention_mask,
                    labels=labels,
                    batch_size=batch["next_img_context"].shape[0],
                    action_len=action_len,
                    sptids_dict=uni_prompting.sptids_dict,
                    gt_tj=batch['future_trajectory'].to(accelerator.device),
                    motion_weight=config.training.motion_weight,
                    ego_status=batch['ego_status'].to(accelerator.device) if config.experiment.add_ego else None,
                    nfp_coffe=config.experiment.nfp_loss,
                    depth_embeddings=depth_features_dict,
                    mode="navsim_alt",
                )

                avg_loss_video = accelerator.gather(loss_video.repeat(total_batch_size_per_gpu)).mean()
                avg_loss_tj = accelerator.gather(loss_tj.repeat(total_batch_size_per_gpu)).mean()
                loss = config.training.video_coeff * loss_video + config.training.tj_coeff * loss_tj

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    if accelerator.is_main_process:
                        # print(f"grad_norm: {grad_norm}")
                        pass
                    # emb_param = next(model.showo.get_input_embeddings().parameters())
                    # print(f"embedding梯度均值: {emb_param.grad.mean().item()}")  # 非0才正常

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_video": avg_loss_video.item(),
                        "step_loss_tj": avg_loss_tj.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                        "loss_tj_below_threshold_count": loss_tj_below_threshold_count,
                        "showo_trainable_enabled": showo_trainable_enabled,
                        "training_stage": staged_training_manager.get_current_stage_info().get('name', 'unknown'),
                        # "stage_switched_staged": stage_switched if 'stage_switched' in locals() else False,
                        # "loss_token":batch_group_losses["pair_0"]["loss"].item(),
                        # "loss_agressvie":batch_group_losses["pair_1"]["loss"].item()
                        "loss_all": loss.item()
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"epoch: {epoch} "
                        f"Step: {global_step + 1} "
                        f"Loss_video: {avg_loss_video.item():0.4f} "
                        f"Loss_tj: {avg_loss_tj.item():0.4f} "
                        f"Loss_tj<0.2计数: {loss_tj_below_threshold_count}/{loss_tj_target_count} "
                        f"Showo可训练: {'是' if showo_trainable_enabled else '否'} "
                        f"训练阶段: {staged_training_manager.get_current_stage_info().get('name', 'unknown')} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.8f} "
                        # f"loss_token:{batch_group_losses['pair_0']['loss'].item():0.4f} "
                        # f"loss_agressvie:{batch_group_losses['pair_1']['loss'].item():0.4f}"
                        f"loss_all:{loss.item():0.4f}"
                    )
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if accelerator.is_main_process and cur_epoch!=epoch and cur_epoch!=-1:
                    # and epoch>=config.training.save_start_epoch:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                # Evaluation
                if cur_epoch != epoch  and (epoch>=config.training.eval_start_epoch or epoch==5):
                    if accelerator.num_processes > 1:
                        accelerator.wait_for_everyone()
                    try:
                        torch.cuda.empty_cache()
                        total_mem = torch.cuda.get_device_properties(accelerator.device).total_memory
                        used_mem = torch.cuda.memory_allocated(accelerator.device)
                        logger.info(f"GPU memory cleared - Used: {used_mem/1024**3:.2f}GB, Free: {(total_mem-used_mem)/1024**3:.2f}GB")
                        eval_logs = evaluate(model,
                                            vq_model,
                                            config,
                                            mask_dtype,
                                            accelerator,
                                            global_step,
                                            uni_prompting,
                                            val_dataloader_navsim,
                                            evaluator,
                                            input_ids_preparer)
                    except Exception as e:
                        save_checkpoint(model, config, accelerator, global_step + 1)
                        logger.error(f"Evaluation failed: {e}")
                        import traceback
                        traceback.print_stack()

                cur_epoch = epoch
                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

        if global_step >= config.training.max_train_steps:
            break

    return global_step

def save_checkpoint(model, config, accelerator, global_step,min_step=None):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)
    if min_step:
        save_path = Path(output_dir) / f"checkpoint_w_min_fvd_{global_step}"
    else:
        save_path = Path(output_dir) / f"checkpoint_step_{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / (grads.numel()).item() + 1e-12)
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)
            print(f"grad_norm/{name}: {grad_norm}")


if __name__ == "__main__":

    main()
