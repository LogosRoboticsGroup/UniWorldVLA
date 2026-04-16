"""
分阶段训练管理器

该模块提供了分阶段训练的管理功能，支持根据训练进度自动切换不同的训练阶段。
主要用于控制 DepthEncoder 的 context 和 dynamic 部分的训练策略。

支持通过环境变量 TRAINING_STAGE 指定训练阶段：
- TRAINING_STAGE=stage1: 只训练 stage1 (context)
- TRAINING_STAGE=stage2: 只训练 stage2 (dynamic)
- TRAINING_STAGE=stage3: 只训练 stage3 (context + dynamic)
- TRAINING_STAGE=stage4: 只训练 stage4 (showo only)
如果未设置环境变量，则使用自动切换模式。
"""

import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class StagedTrainingManager:
    """
    分阶段训练管理器
    
    管理不同训练阶段之间的切换逻辑，支持基于loss阈值和训练步数的自动切换。
    阶段按配置顺序依次执行，跳过未启用的阶段。
    
    Attributes:
        config: 配置对象
        model: 模型对象
        accelerator: 加速器对象
        depth_encoder: 深度编码器对象（可选）
        enable_staged: 是否启用分阶段训练
        stages: 训练阶段列表（按配置顺序）
        current_stage_index: 当前阶段索引
        stage_switched: 是否发生了阶段切换
        below_threshold_count: 满足阈值的连续次数计数器
    """
    
    def __init__(self, config: Any, model: Any, accelerator: Any, depth_encoder: Optional[Any] = None):
        """
        初始化分阶段训练管理器
        
        Args:
            config: 配置对象，需包含 training.staged_training 配置
            model: 模型对象
            accelerator: 加速器对象（用于分布式训练和日志记录）
            depth_encoder: 深度编码器对象（可选，如果不提供则无法控制训练状态）
        """
        self.config = config
        self.model = model
        self.accelerator = accelerator
        self.depth_encoder = depth_encoder
        
        # 检查是否启用分阶段训练
        self.staged_config = getattr(config.training, 'staged_training', None) if hasattr(config, 'training') else None
        self.enable_staged = False
        
        if self.staged_config is not None and self.staged_config.get('enable', False):
            self.enable_staged = True
        
        # 阶段状态
        self.stages: list[Dict[str, Any]] = []
        self.current_stage_index = 0
        self.stage_switched = False
        self.initialized = False  # 是否已经初始化并应用了首个阶段配置
        self.below_threshold_count = 0
        
        # 初始化阶段配置
        if self.enable_staged:
            self._initialize_stages()
    
    def _build_stage_dict(self, stage_config: Dict[str, Any],stage_name: str, auto_switch: bool) -> Dict[str, Any]:
        """构建阶段配置字典

        Args:
            stage_config: 配置文件中的阶段配置
            auto_switch: 是否启用自动切换

        Returns:
            阶段配置字典
        """
        return {
            'name': stage_config.get('name', 'unknown'),
            'enable_context': stage_config.get('enable_context', True),
            'enable_dynamic': stage_config.get('enable_dynamic', False),
            'auto_switch': auto_switch,
            'threshold_metrics': stage_config.get('threshold_metrics', {}),
            'freeze_depth_encoder': stage_config.get('freeze_depth_encoder', False),
            'freeze_showo': stage_config.get('freeze_showo', True),
            'stage':stage_name
        }

    def _initialize_stages(self) -> None:
        """初始化训练阶段配置（按配置顺序组织成列表）"""
        if self.staged_config is None:
            return

        # 检查环境变量，支持手动指定训练阶段
        training_stage_env = os.getenv("TRAINING_STAGE", "").lower()

        # 规范化：'1' -> 'stage1', 'stage1' -> 'stage1'
        if training_stage_env and not training_stage_env.startswith('stage'):
            training_stage_env = f"stage{training_stage_env}"

        # 从配置读取阶段列表（支持任意数量的阶段）
        stage_names = [key for key in self.staged_config.keys() if key.startswith('stage')]
        stage_names.sort()  # 按 stage1, stage2, stage3 顺序排序

        self.stages = []
        for stage_name in stage_names:
            stage_config = self.staged_config.get(stage_name, {})

            # 如果指定了环境变量，只添加匹配的阶段（忽略enable配置）
            if training_stage_env:
                if stage_name == training_stage_env:
                    self.stages.append(self._build_stage_dict(stage_config, stage_name=stage_name, auto_switch=False))
                continue

            # 自动模式：只添加启用的阶段到列表
            if not stage_config.get('enable', True):
                continue

            self.stages.append(self._build_stage_dict(stage_config, stage_name, auto_switch=stage_config.get('auto_switch', True)))

        # 设置当前阶段为第一个启用的阶段
        self.current_stage_index = 0 if self.stages else -1

        if self.current_stage_index < 0:
            logger.warning("⚠️  所有训练阶段都未启用，分阶段训练将不会生效")

        # 注意:不在这里应用配置,而是在首次调用 check_and_switch_stage 时应用
        # 这样可以在训练开始前触发日志,确保能够知道当前训练模式
        self.check_and_switch_stage(1,10000)
    
    def _apply_stage_config(self) -> None:
        """应用当前阶段的训练配置
        
        ⚠️ 重要：此方法必须在所有进程中执行，确保所有进程的训练模式一致
        只有日志记录在主进程中进行，但模型状态切换必须在所有进程中完成
        """
        if self.current_stage_index < 0 or self.current_stage_index >= len(self.stages):
            return
        
        current_stage = self.stages[self.current_stage_index]
        
        # 🚨 关键修复：移除 is_main_process 条件！！！
        # 所有进程都必须应用模型配置，否则不同进程会在不同模式下训练
        
        # 主进程记录日志
        if self.accelerator.is_main_process:
            logger.info(f"\n{'='*70}")
            logger.info(f"🔄 切换到训练阶段 ({self.current_stage_index + 1}/{len(self.stages)}): {current_stage['name']} {current_stage['stage']}")
            logger.info(f"📋 Context模块: {'✅ 启用' if current_stage['enable_context'] else '❌ 跳过'}")
            logger.info(f"📋 Dynamic模块: {'✅ 启用' if current_stage['enable_dynamic'] else '❌ 跳过'}")
            
            # 新增：显示 Depth Encoder 是否冻结
            if 'freeze_depth_encoder' in current_stage:
                logger.info(f"📋 Depth Encoder: {'❌ 冻结' if current_stage['freeze_depth_encoder'] else '✅ 训练'}")
            
            logger.info(f"{'='*70}")
        
        # 🚨 所有进程都必须执行模型配置切换
        if self.model is not None:
            # 1. 设置 depth_cross_attention 的训练阶段（context/dynamic 启用状态）
            if hasattr(self.model.depth_encoder, 'set_depth_cross_attention_training_stage'):
                self.model.depth_encoder.set_depth_cross_attention_training_stage(
                    enable_context=current_stage['enable_context'],
                    enable_dynamic=current_stage['enable_dynamic'],
                    verbose=True  # 只有主进程记录日志，避免重复
                )
            else:
                if self.accelerator.is_main_process:
                    logger.warning(f"⚠️  模型没有 set_depth_cross_attention_training_stage 方法，无法设置训练阶段")
                    
            # 3. 处理 freeze_showo 配置（独立控制 ShowO 模型）
            if 'freeze_showo' in current_stage:
                freeze_showo = current_stage['freeze_showo']
                
                # 冻结或解冻 ShowO 模型
                if hasattr(self.model, 'set_showo_trainable'):
                    self.model.set_showo_trainable(
                        trainable=not freeze_showo,
                        verbose=True
                    )
                    if self.accelerator.is_main_process:
                        status = "🔒 冻结" if freeze_showo else "✅ 训练"
                        logger.info(f"   {status} ShowO 模型")
            
            # 2. 处理 freeze_depth_encoder 配置（独立控制 Depth Encoder）
            if 'freeze_depth_encoder' in current_stage:
                freeze_depth = current_stage['freeze_depth_encoder']
                
                # 冻结或解冻 Depth Encoder
                if hasattr(self.model, 'depth_encoder'):
                    self.model.depth_encoder.set_trainable(
                        trainable=not freeze_depth,
                        stage=current_stage['stage'],
                        verbose=True
                    )
                    if self.accelerator.is_main_process:
                        status = "🔒 冻结" if freeze_depth else "✅ 训练"
                        logger.info(f"   {status} Depth Encoder 的参数")
            
            
    
    def check_and_switch_stage(self, global_step: int, loss_tj: Optional[float] = None) -> bool:
        """
        检查并切换训练阶段
        
        Args:
            global_step: 当前全局步数
            loss_tj: 当前tj loss值（可选，用于基于loss的切换条件）
        
        Returns:
            bool: 是否发生了阶段切换或初始化完成
        
        Example:
            >>> manager = StagedTrainingManager(config, model, accelerator, depth_encoder)
            >>> for step in range(max_steps):
            ...     loss = train_step()
            ...     if manager.check_and_switch_stage(step, loss):
            ...         print(f"阶段切换到: {manager.get_current_stage_info()['name']}")
        """
        if not self.enable_staged:
            return False
        
        if self.current_stage_index < 0 or self.current_stage_index >= len(self.stages):
            return False
        
        # 首次调用:应用首个阶段配置并返回true
        if not self.initialized:
            self._apply_stage_config()
            self.initialized = True
            # 确保所有进程都完成初始化后再继续
            self.accelerator.wait_for_everyone()
            return True
        
        current_stage = self.stages[self.current_stage_index]
        
        if not current_stage.get('auto_switch', False):
            return False
        
        switch_to_next = False
        
        # 检查切换条件
        threshold_metrics = current_stage.get('threshold_metrics', {})
        
        # 检查loss阈值条件（两个阶段都支持）
        if threshold_metrics.get('loss_tj_below_threshold', False):
            threshold_value = threshold_metrics.get('loss_tj_threshold', 0.1)
            target_count = threshold_metrics.get('loss_tj_target_count', 50)
            
            if loss_tj is not None and loss_tj < threshold_value:
                self.below_threshold_count += 1
                if self.accelerator.is_main_process:
                    logger.info(f"📊 tj loss < {threshold_value}: {self.below_threshold_count}/{target_count}")
                
                if self.below_threshold_count >= target_count:
                    switch_to_next = True
            else:
                self.below_threshold_count = 0
        
        # 检查步数阈值条件（两个阶段都支持）
        elif 'step_threshold' in threshold_metrics:
            step_threshold = threshold_metrics['step_threshold']
            if global_step >= step_threshold:
                switch_to_next = True
        
        # 执行阶段切换 - 所有进程需要同步执行
        if switch_to_next:
            # 确保所有进程都达成一致后再切换
            self.accelerator.wait_for_everyone()
            self._switch_to_next_stage()
            # 切换完成后再次同步，确保所有进程都完成了切换
            self.accelerator.wait_for_everyone()
            return True
        
        return False
    
    def _switch_to_next_stage(self) -> None:
        """切换到下一个阶段"""
        if self.current_stage_index < 0:
            return
        
        # 重置计数器
        self.below_threshold_count = 0
        
        # 移动到下一阶段
        self.current_stage_index += 1
        
        # 检查是否还有下一阶段
        if self.current_stage_index >= len(self.stages):
            if self.accelerator.is_main_process:
                logger.info("✅ 已完成所有训练阶段")
        else:
            self._apply_stage_config()
            self.stage_switched = True
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """
        获取当前阶段的信息
        
        Returns:
            dict: 包含当前阶段信息的字典，包括：
                - name: 阶段名称
                - enable_context: 是否训练context
                - enable_dynamic: 是否训练dynamic
                - stage_switched: 是否发生过切换
                - stage_index: 当前阶段索引
                - total_stages: 总阶段数
        """
        if self.current_stage_index < 0 or self.current_stage_index >= len(self.stages):
            return {
                'name': 'unknown',
                'enable_context': False,
                'enable_dynamic': False,
                'stage_switched': False,
                'stage_index': -1,
                'total_stages': len(self.stages)
            }
        
        current_stage = self.stages[self.current_stage_index]
        return {
            'name': current_stage['name'],
            'enable_context': current_stage['enable_context'],
            'enable_dynamic': current_stage['enable_dynamic'],
            'stage_switched': self.stage_switched,
            'stage_index': self.current_stage_index,
            'total_stages': len(self.stages)
        }