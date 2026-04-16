"""
深度 Cross Attention 工具函数
从 inputs 中查找 soi/eoi 和 sod/eod 区间，提取对应 embeddings，与 depth_embeddings 进行 cross attention

支持一个 batch 中有多组区间的情况，会将所有区间的 embeddings cat 在一起
使用 group mask 实现组内可见、组间不可见的 attention
depth_embeddings 形状为 (B, T, seq_len, depth_feature_dim)，在外部已与提取的embeddings对齐
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from .depth_feature_extractor import DepthImageCrossAttention
  

class BaseDepthCrossAttentionModule(nn.Module):
    """
    深度 Cross Attention 模块
    从 inputs 中查找 soi/eoi 和 sod/eod 区间，提取对应 embeddings，与 depth_embeddings 进行 cross attention
    
    支持一个 batch 中有多组区间的情况，会将所有区间的 embeddings cat 在一起
    使用 group mask 实现组内可见、组间不可见的 attention
    depth_embeddings 形状为 (B, T, seq_len, depth_feature_dim)，在外部已与提取的embeddings对齐
    
    分阶段训练支持：
    - context 阶段：只执行 context_attended，跳过 dynamic_attended
    - dynamic 阶段：跳过 context_attended，只执行 dynamic_attended（但 context_attended 的特征仍然可用）
    - both 阶段：同时执行 context_attended 和 dynamic_attended
    """
    def __init__(
        self,
        depth_feature_dim: int,
        image_feature_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_adaptive_projection: bool = False,
        model_name: str = "depth_cross_attention",
    ):
        super().__init__()
        
        self.depth_feature_dim = depth_feature_dim
        self.image_feature_dim = image_feature_dim
        self.num_heads = num_heads
        self.use_adaptive_projection = use_adaptive_projection
        self.model_name = model_name
        
        # Context Cross Attention 模块（用于 soi/eoi）
        self.context_cross_attention = DepthImageCrossAttention(
            depth_feature_dim=depth_feature_dim,
            image_feature_dim=image_feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_adaptive_projection=use_adaptive_projection,
        )
        
        # Dynamic Cross Attention 模块（用于 sod/eod）
        self.dynamic_cross_attention = DepthImageCrossAttention(
            depth_feature_dim=depth_feature_dim,
            image_feature_dim=image_feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_adaptive_projection=use_adaptive_projection,
        )
        
        self.context_projector = torch.nn.Sequential(
                torch.nn.Linear(2048, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )
        self.dynamic_projector = torch.nn.Sequential(
                torch.nn.Linear(2048, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )
        
        # 模块启用状态控制
        self.enable_context = True   # 是否启用 context 模块（soi/eoi）
        self.enable_dynamic = True   # 是否启用 dynamic 模块（sod/eod）
    
    def _find_all_token_indices(
        self,
        inputs: torch.Tensor,
        sptids_dict: dict,
        start_token: str,
        end_token: str
    ) -> List[List[Tuple[int, int]]]:
        """
        在 inputs 中查找所有的 start_token 和 end_token 区间对
        
        Args:
            inputs: 输入 token IDs，形状 (B, seq_len)
            sptids_dict: 特殊 token ID 字典
            start_token: 起始特殊 token 名称
            end_token: 结束特殊 token 名称
            
        Returns:
            all_ranges: 每个 batch 的所有区间列表
        """
        batch_size = inputs.shape[0]
        
        start_id = sptids_dict[start_token].item()
        end_id = sptids_dict[end_token].item()
        
        all_ranges = []
        
        for batch_idx in range(batch_size):
            input_batch = inputs[batch_idx]
            
            start_positions = torch.where(input_batch == start_id)[0]
            end_positions = torch.where(input_batch == end_id)[0]
            
            batch_ranges = []
            
            if len(start_positions) == 0 or len(end_positions) == 0:
                all_ranges.append(batch_ranges)
                continue
            
            for start_idx in start_positions:
                end_after_start = end_positions[end_positions > start_idx]
                if len(end_after_start) > 0:
                    end_idx = end_after_start[0].item()
                    batch_ranges.append((start_idx.item(), end_idx))
            
            all_ranges.append(batch_ranges)
        
        return all_ranges
    
    def _extract_batch_segments(
        self,
        input_embeddings: torch.Tensor,
        batch_idx: int,
        batch_ranges: List[Tuple[int, int]],
        device: torch.device
    ) -> Tuple[torch.Tensor, List[int], int]:
        """
        提取单个batch中所有区间的embeddings并拼接
        #review
        Args:
            input_embeddings: 输入embeddings (B, seq_len, feature_dim)
            batch_idx: 当前batch索引
            batch_ranges: 当前batch的所有区间 [(start1, end1), (start2, end2), ...]
            device: 设备
            
        Returns:
            concatenated: 拼接后的embeddings (total_len, feature_dim)
            interval_lengths: 每个区间的长度列表 [len1, len2, ...]
            total_len: 总长度
        """
        if len(batch_ranges) == 0:
            empty_tensor = torch.zeros(0, input_embeddings.shape[2], device=device)
            return empty_tensor, [], 0
        
        segments = []
        interval_lengths = []
        
        for start_idx, end_idx in batch_ranges:
            # 提取区间内的embeddings（不包含特殊token）
            segment = input_embeddings[batch_idx, start_idx + 1:end_idx]
            segments.append(segment)
            interval_lengths.append(segment.shape[0])
        
        concatenated = torch.cat(segments, dim=0)
        total_len = concatenated.shape[0]
        
        return concatenated, interval_lengths, total_len
    
    def _pad_to_max_length(
        self,
        embeddings_list: List[torch.Tensor],
        max_total_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        将所有batch的embeddings padding到相同长度
        #checked
        Args:
            embeddings_list: 各batch的embeddings列表 [emb1, emb2, ...]
            max_total_len: 最大总长度
            device: 设备
            
        Returns:
            padded_embeddings: padding后的embeddings (B, max_total_len, feature_dim)
        """
        if max_total_len == 0:
            return torch.zeros(len(embeddings_list), 0, embeddings_list[0].shape[1], device=device)
        
        padded_list = []
        for emb in embeddings_list:
            current_len = emb.shape[0]
            
            if current_len < max_total_len:
                padding_length = max_total_len - current_len
                padding = torch.zeros(padding_length, emb.shape[1], device=device)
                padded_emb = torch.cat([emb, padding], dim=0)
            else:
                padded_emb = emb
            
            padded_list.append(padded_emb)
        
        return torch.stack(padded_list, dim=0)
    
    def _extract_and_concat_embeddings(
        self,
        input_embeddings: torch.Tensor,
        all_ranges: List[List[Tuple[int, int]]],
        device: torch.device
    ) -> Tuple[torch.Tensor, List[int], List[List[int]]]:
        """
        提取所有区间的embeddings并拼接，记录每个区间的长度
        #checked
        该函数处理流程：
        1. 遍历每个batch，提取所有区间的embeddings
        2. 将每个batch中的所有区间连接在一起
        3. 将不同batch的连接结果padding到相同长度
        
        Args:
            input_embeddings: 输入embeddings (B, seq_len, image_feature_dim)
            all_ranges: 每个batch的所有区间列表
            device: 设备
            
        Returns:
            concatenated_embeddings: 拼接后的embeddings (B, max_total_len, image_feature_dim)
            total_lengths: 每个batch的总长度列表
            interval_lengths: 每个batch的每个区间长度列表 [[len1, len2, ...], ...]
        """
        batch_size = input_embeddings.shape[0]
        
        # 提取每个batch的embeddings
        embeddings_list = []
        total_lengths = []
        all_interval_lengths = []
        
        for batch_idx in range(batch_size):
            batch_ranges = all_ranges[batch_idx]
            
            # 提取并拼接当前batch的所有区间
            concatenated, interval_lengths, total_len = self._extract_batch_segments(
                input_embeddings, batch_idx, batch_ranges, device
            )
            
            embeddings_list.append(concatenated)
            total_lengths.append(total_len)
            all_interval_lengths.append(interval_lengths)
        
        # 计算最大总长度
        max_total_len = max(total_lengths) if len(total_lengths) > 0 else 0
        
        # 将所有batch的embeddingspadding到相同长度
        concatenated_embeddings = self._pad_to_max_length(
            embeddings_list, max_total_len, device
        )
        
        return concatenated_embeddings, total_lengths, all_interval_lengths
    
    def _create_group_attention_mask(
        self,
        interval_lengths: List[List[int]],
        max_total_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        创建 group attention mask（内存优化版本）
        
        每个区间内的 tokens 可以互相看见，但不同区间的 tokens 互相不可见
        
        内存优化：使用bool类型而非float类型，节省内存（约为原来的1/4）
        
        Args:
            interval_lengths: 每个 batch 的每个区间长度列表
            max_total_len: 最大总长度
            device: 设备
            
        Returns:
            attention_mask: attention mask (B, 1, max_total_len, max_total_len)
                          True 表示不可见，False 表示可见
                          注意：第2个维度为1，用于广播到多头attention
        """
        batch_size = len(interval_lengths)
        
        # 使用bool类型而非float类型，节省约75%内存
        attention_mask = torch.ones(
            (batch_size, 1, max_total_len, max_total_len),
            dtype=torch.bool,
            device=device
        )
        
        for batch_idx in range(batch_size):
            batch_intervals = interval_lengths[batch_idx]
            
            if len(batch_intervals) == 0:
                continue
            
            start_pos = 0
            for interval_len in batch_intervals:
                if interval_len == 0:
                    continue
                
                end_pos = start_pos + interval_len
                # 四维索引：[batch_idx, head_dim, seq_start:seq_end, seq_start:seq_end]
                attention_mask[batch_idx, :, start_pos:end_pos, start_pos:end_pos] = False
                start_pos = end_pos
        
        return attention_mask
    
    def _replace_with_attended_features(
        self,
        input_embeddings: torch.Tensor,
        attended_features: torch.Tensor,
        all_ranges: List[List[Tuple[int, int]]],
        total_lengths: List[int],
        projector: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        将 attended 的 depth features 替换回 original embeddings 的对应位置
        #checked
        Args:
            input_embeddings: 原始 embeddings (B, seq_len, image_feature_dim)
            attended_features: attended后的depth features (B, max_total_len, depth_feature_dim)
            all_ranges: 每个 batch 的所有区间列表
            total_lengths: 每个 batch 的总长度列表
            projector: 可选的 projector MLP，用于特征分布对齐
            
        Returns:
            updated_embeddings: 更新后的embeddings (B, seq_len, image_feature_dim)
        """
        device = input_embeddings.device
        batch_size = input_embeddings.shape[0]
        
        # 投影到 image_feature_dim
        if attended_features.shape[-1] != input_embeddings.shape[-1]:
            if not hasattr(self, 'output_projection'):
                self.output_projection = nn.Linear(attended_features.shape[-1], input_embeddings.shape[-1])
                self.output_projection.to(device)
            attended_features = self.output_projection(attended_features)
        
        updated_embeddings = input_embeddings.clone()
        
        for batch_idx in range(batch_size):
            batch_ranges = all_ranges[batch_idx]
            batch_total_len = total_lengths[batch_idx]
            
            if batch_total_len == 0 or len(batch_ranges) == 0:
                continue
            
            # 获取该batch的attended features
            batch_attended = attended_features[batch_idx, :batch_total_len, :]
            
            # 按区间顺序切片替换
            start_pos = 0
            for start_idx, end_idx in batch_ranges:
                # 原始区间（不包含特殊token）
                orig_start = start_idx + 1
                orig_end = end_idx
                orig_len = orig_end - orig_start
                
                if orig_len == 0:
                    continue
                
                # 从batch_attended中切片对应长度
                end_pos = start_pos + orig_len
                segment = batch_attended[start_pos:end_pos, :]
                
                # 使用MLP进行特征分布对齐（如果提供了projector）
                if projector is not None:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        segment = self.projector(updated_embeddings[batch_idx, orig_start:orig_end],segment,projector=projector)
                        # segment = projector(segment)
                
                # 替换到原始位置
                updated_embeddings[batch_idx, orig_start:orig_end] = segment
                start_pos = end_pos
        
        return updated_embeddings
    
    def _process_single_interval(
        self,
        input_embeddings: torch.Tensor,
        inputs: torch.Tensor,
        depth_embeddings: torch.Tensor,
        sptids_dict: dict,
        start_token: str,
        end_token: str,
        cross_attention_module: nn.Module,
        projector: Optional[nn.Module],
        device: torch.device
    ) -> torch.Tensor:
        """
        处理单个区间的 cross attention（使用 group mask）
        
        Args:
            input_embeddings: 输入embeddings (B, seq_len, image_feature_dim)
            inputs: 输入token IDs (B, seq_len)
            depth_embeddings: 深度特征 (B, num_tokens, depth_feature_dim)
              在外部已与 concatenated_embeddings 对齐
            sptids_dict: 特殊token ID字典
            start_token: 起始token名称
            end_token: 结束token名称
            cross_attention_module: cross attention模块
            projector: 可选的 projector MLP，用于特征分布对齐
            device: 设备
            
        Returns:
            updated_embeddings: 更新后的embeddings (B, seq_len, image_feature_dim)
        """
        # assert inputs.shape == depth_embeddings.shape
        all_ranges = self._find_all_token_indices(inputs, sptids_dict, start_token, end_token)
        
        # if len(all_ranges[0]) ==0 :
        #     return input_embeddings
        concatenated_embeddings, total_lengths, interval_lengths = self._extract_and_concat_embeddings(
            input_embeddings, all_ranges, device
        )
        
        max_total_len = concatenated_embeddings.shape[1]
        if max_total_len == 0:
            return input_embeddings
        
        # 创建 group attention mask
        attention_mask = self._create_group_attention_mask(
            interval_lengths, max_total_len, device
        )
        
        # depth_embeddings 在外部已经与 concatenated_embeddings 对齐
        # 直接使用 depth_embeddings 进行 cross attention
        
        # 内存优化：将bool mask转换为float mask（在使用时转换，避免提前占用大量内存）
        if attention_mask.dtype == torch.bool:
            # 将bool mask转换为float mask，但只在需要时转换
            attention_mask_float = attention_mask.float().masked_fill(attention_mask, float('-inf')).contiguous()
        else:
            attention_mask_float = attention_mask
        if cross_attention_module is not None:
            assert len(all_ranges)==depth_embeddings.shape[0] and len(all_ranges[0])== depth_embeddings.shape[1] and \
            (-all_ranges[0][0][0] + all_ranges[0][0][1]-1)==depth_embeddings.shape[2]
            if  cross_attention_module.training:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # 进行cross attention（使用group mask实现组内可见）
                    attended_features = cross_attention_module(
                        depth_features=concatenated_embeddings,
                        image_features=depth_embeddings.reshape(depth_embeddings.shape[0],-1,depth_embeddings.shape[-1]),
                        attention_mask=attention_mask_float
                    )
            else :
                attended_features = cross_attention_module(
                    depth_features=concatenated_embeddings,
                    image_features=depth_embeddings.reshape(depth_embeddings.shape[0],-1,depth_embeddings.shape[-1]),
                    attention_mask=attention_mask_float
                )
        else :
            # for residual equal to input_embeddings
            attended_features, total_lengths, interval_lengths = self._extract_and_concat_embeddings(
            depth_embeddings, all_ranges, device
            ) 
            check_shape_clone = attended_features[:,:,0].reshape(attended_features.shape[0],len(all_ranges[0]),-1)       
            assert len(all_ranges)==check_shape_clone.shape[0] and len(all_ranges[0])== check_shape_clone.shape[1] and \
            (-all_ranges[0][0][0] + all_ranges[0][0][1]-1)==check_shape_clone.shape[2]

        
        # 及时释放临时mask的内存
        del attention_mask_float
        
        # 替换回原始embeddings
        updated_embeddings = self._replace_with_attended_features(
            input_embeddings, attended_features, all_ranges, total_lengths, projector
        )
        
        return updated_embeddings
    
    def set_context_dynamic_enabled(self, enable_context: bool, enable_dynamic: bool):
        """
        设置 context 和 dynamic 模块的启用状态
        
        Args:
            enable_context: 是否启用 context 模块（soi/eoi区间）
            enable_dynamic: 是否启用 dynamic 模块（sod/eod区间）
        """
        self.enable_context = enable_context
        self.enable_dynamic = enable_dynamic
    
    def _process_module_with_staged_training(
        self,
        input_embeddings: torch.Tensor,
        inputs: torch.Tensor,
        depth_embeddings: torch.Tensor,
        sptids_dict: dict,
        start_token: str,
        end_token: str,
        cross_attention_module: nn.Module,
        projector: Optional[nn.Module],
        device: torch.device,
        is_enabled: bool
    ) -> torch.Tensor:
        """
        根据 training_stage 处理单个模块，支持分阶段训练
        
        Args:
            input_embeddings: 输入embeddings
            inputs: 输入token IDs
            depth_embeddings: 深度特征
            sptids_dict: 特殊token ID字典
            start_token: 起始token名称
            end_token: 结束token名称
            cross_attention_module: cross attention模块
            projector: projector MLP
            device: 设备
            is_enabled: 该模块是否在当前训练阶段启用
        
        Returns:
            updated_embeddings: 更新后的embeddings
        """
        if depth_embeddings is None:
            return input_embeddings
        if is_enabled:
            # 正常训练阶段：正常处理
            output = self._process_single_interval(
                input_embeddings=input_embeddings,
                inputs=inputs,
                depth_embeddings=depth_embeddings,
                sptids_dict=sptids_dict,
                start_token=start_token,
                end_token=end_token,
                cross_attention_module=cross_attention_module,
                projector=projector,
                device=device
            )
            return output
        else:
            # 不训练该阶段：将输出乘以0，保持架构统一，梯度为0，backward后自动释放
            with torch.enable_grad():
                output = self._process_single_interval(
                    input_embeddings=input_embeddings,
                    inputs=inputs,
                    depth_embeddings=depth_embeddings,
                    sptids_dict=sptids_dict,
                    start_token=start_token,
                    end_token=end_token,
                    cross_attention_module=cross_attention_module,
                    projector=projector,
                    device=device
                )
                # 计算差异并乘以0
                zero_output = (output - input_embeddings) * 0
                result = input_embeddings + zero_output  # 等价于不加任何东西
                del output, zero_output
                return result
            
    def projector(self, x,y,projector):
        # print("commom projector")
        return projector(y)
    
    def forward(
        self,
        input_embeddings: torch.Tensor,
        inputs: torch.Tensor,
        depth_embeddings: Dict[str, torch.Tensor],
        sptids_dict: dict,
        soi_token: str = "<|soi|>",
        eoi_token: str = "<|eoi|>",
        sod_token: str = "<|sod|>",
        eod_token: str = "<|eod|>",
        return_selected: bool = False
    ):
        """
        前向传播（支持分阶段训练控制）
        
        Args:
            input_embeddings: 输入embeddings (B, seq_len, image_feature_dim)
            inputs: 输入token IDs (B, seq_len)
            depth_embeddings: 深度特征字典
                - 'context_attended': (B, num_tokens, depth_feature_dim)
                  在外部已与提取的embeddings对齐
                - 'dynamic_attended': (B, num_tokens, depth_feature_dim)
                  在外部已与提取的embeddings对齐
              使用group mask实现组内可见、组间不可见
            sptids_dict: 特殊token ID字典
            soi_token: soi特殊token名称
            eoi_token: eoi特殊token名称
            sod_token: sod特殊token名称
            eod_token: eod特殊token名称
            return_selected: 是否返回选中的原始embeddings
        
        Returns:
            updated_embeddings: 更新后的embeddings
            （如果return_selected=True，则返回tuple(updated_embeddings, selected_embeddings)）
        
        分阶段训练行为（通过 enable_context 和 enable_dynamic 控制）：
        - enable_context=True, enable_dynamic=False: 只处理 soi/eoi 区间（context_attended）
        - enable_context=False, enable_dynamic=True: 只处理 sod/eod 区间（dynamic_attended）
        - enable_context=True, enable_dynamic=True: 同时处理两个区间
        """
        if inputs is None:
            return input_embeddings
        assert len(inputs.shape) ==2
        device = input_embeddings.device
        updated_embeddings = input_embeddings.clone()
        
        context_all_ranges = None
        dynamic_all_ranges = None
        
        if return_selected:
            context_all_ranges = self._find_all_token_indices(inputs, sptids_dict, soi_token, eoi_token)
            dynamic_all_ranges = self._find_all_token_indices(inputs, sptids_dict, sod_token, eod_token)
        
        # 定义处理模块的配置（根据启用状态决定是否处理）
        modules_to_process = [
            {
                'key': 'context_attended',
                'enable': self.enable_context,
                'start_token': soi_token,
                'end_token': eoi_token,
                'cross_attention': self.context_cross_attention,
                'projector': self.context_projector
            },
            {
                'key': 'dynamic_attended',
                'enable': self.enable_dynamic,
                # 'enable': (self.training_stage in ["dynamic", "both"]),
                'start_token': sod_token,
                'end_token': eod_token,
                'cross_attention': self.dynamic_cross_attention,
                'projector': self.dynamic_projector
            }
        ]
        
        # 统一处理所有模块
        for module_config in modules_to_process:
            if module_config['key'] in depth_embeddings:
                depth_embs = depth_embeddings[module_config['key']]
                if depth_embs is None:
                    # print(module_config['key'] + "===not use=====" + self.model_name)
                    continue
                else:
                    # print(module_config['key'] + "====use=====" + self.model_name)
                    pass
                updated_embeddings = self._process_module_with_staged_training(
                    input_embeddings=updated_embeddings,
                    inputs=inputs,
                    depth_embeddings=depth_embs,
                    sptids_dict=sptids_dict,
                    start_token=module_config['start_token'],
                    end_token=module_config['end_token'],
                    cross_attention_module=module_config['cross_attention'],
                    projector=module_config['projector'],
                    device=device,
                    is_enabled=module_config['enable']
                )
               
        if return_selected and context_all_ranges is not None and dynamic_all_ranges is not None:
            context_concatenated, _, _ = self._extract_and_concat_embeddings(
                input_embeddings, context_all_ranges, device
            )
            dynamic_concatenated, _, _ = self._extract_and_concat_embeddings(
                input_embeddings, dynamic_all_ranges, device
            )
            selected_embeddings = {
                'context': context_concatenated,
                'dynamic': dynamic_concatenated
            }
            return updated_embeddings, selected_embeddings
        
        return updated_embeddings
    
class DepthCrossAttentionModule(BaseDepthCrossAttentionModule):
    def __init__(
        self,
        depth_feature_dim: int,
        image_feature_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_adaptive_projection: bool = False,
    ):
        super().__init__(depth_feature_dim, image_feature_dim, num_heads, dropout, use_adaptive_projection,"DepthCrossAttentionModule")

    
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self,ineed_to_update,for_update):
        return self.norm1(ineed_to_update + self.mlp(for_update))  # 核心残差连接        
class DepthCrossAttentionModuleResidual(BaseDepthCrossAttentionModule):
    def __init__(
        self,
        depth_feature_dim: int,
        image_feature_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_adaptive_projection: bool = False,
    ):
        super().__init__(depth_feature_dim, image_feature_dim, num_heads, dropout, use_adaptive_projection,"DepthCrossAttentionModuleResidual") 
        # Context Cross Attention 模块（用于 soi/eoi）
        self.context_cross_attention = None
        
        # Dynamic Cross Attention 模块（用于 sod/eod）
        self.dynamic_cross_attention = None
        
        self.context_projector = ResidualMLPBlock(2048)
        self.dynamic_projector = ResidualMLPBlock(2048)
    #need_to_update ==>hidstate for_update ====> embding
    def projector(self, need_to_update,for_update,projector):
        # print("residual projector")
        return projector(need_to_update,for_update)
    