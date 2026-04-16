import torch
import torch.nn.functional as F


def _compute_token_weights(
    input_ids: torch.Tensor,
    token_pairs_weights: torch.Tensor,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    include_start_end_tokens: bool = True
) -> torch.Tensor:
    """计算token权重矩阵的辅助函数"""
    weights = torch.zeros(batch_size * seq_len, dtype=dtype, device=device)
    
    for batch_idx in range(batch_size):
        current_input_ids = input_ids[batch_idx]
        base_idx = batch_idx * seq_len
        
        for token_pair in token_pairs_weights:
            start_token_id = int(token_pair[0].item())
            end_token_id = int(token_pair[1].item())
            weight = token_pair[2].item()
            
            start_positions = torch.where(current_input_ids == start_token_id)[0]
            end_positions = torch.where(current_input_ids == end_token_id)[0]
            
            for start_pos in start_positions:
                end_pos_candidates = end_positions[end_positions > start_pos]
                if len(end_pos_candidates) > 0:
                    end_pos = end_pos_candidates[0]
                    
                    if include_start_end_tokens:
                        global_start = base_idx + start_pos
                        global_end = base_idx + end_pos + 1
                    else:
                        if end_pos > start_pos + 1:
                            global_start = base_idx + start_pos + 1
                            global_end = base_idx + end_pos
                        else:
                            continue
                    
                    weights[global_start:global_end] += weight
    
    return weights


def _compute_single_token_pair_weights(
    input_ids: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    include_start_end_tokens: bool = True
) -> torch.Tensor:
    """计算单个token对的权重矩阵"""
    weights = torch.zeros(batch_size * seq_len, dtype=dtype, device=device)
    
    for batch_idx in range(batch_size):
        current_input_ids = input_ids[batch_idx]
        base_idx = batch_idx * seq_len
        
        start_positions = torch.where(current_input_ids == start_token_id)[0]
        end_positions = torch.where(current_input_ids == end_token_id)[0]
        
        for start_pos in start_positions:
            end_pos_candidates = end_positions[end_positions > start_pos]
            if len(end_pos_candidates) > 0:
                end_pos = end_pos_candidates[0]
                
                if include_start_end_tokens:
                    global_start = base_idx + start_pos
                    global_end = base_idx + end_pos + 1
                else:
                    if end_pos > start_pos + 1:
                        global_start = base_idx + start_pos + 1
                        global_end = base_idx + end_pos
                    else:
                        continue
                
                weights[global_start:global_end] += 1.0
    
    return weights


def _create_delimiter_mask(
    input_ids: torch.Tensor,
    token_pairs_weights: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device
) -> torch.Tensor:
    """创建分隔符掩码，用于指示哪些位置是分隔符"""
    delimiter_mask = torch.ones(batch_size * seq_len, dtype=torch.bool, device=device)
    
    for batch_idx in range(batch_size):
        current_input_ids = input_ids[batch_idx]
        base_idx = batch_idx * seq_len
        
        for token_pair in token_pairs_weights:
            start_token_id = int(token_pair[0].item())
            end_token_id = int(token_pair[1].item())
            
            start_positions = torch.where(current_input_ids == start_token_id)[0]
            end_positions = torch.where(current_input_ids == end_token_id)[0]
            
            for start_pos in start_positions:
                delimiter_mask[base_idx + start_pos] = False
            
            for end_pos in end_positions:
                delimiter_mask[base_idx + end_pos] = False
    
    return delimiter_mask


def _compute_focal_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    focal_gamma: float
) -> torch.Tensor:
    """计算focal loss权重
    
    参数:
        logits: 模型输出的logits
        labels: 目标标签
        valid_mask: 有效token的掩码
        batch_size: batch大小
        seq_len: 序列长度
        focal_gamma: focal loss的gamma参数
    
    返回:
        focal权重张量
    """
    focal_weight = torch.ones(batch_size * seq_len, dtype=logits.dtype, device=logits.device)
    
    # 只在有效且属于当前位置计算focal权重
    valid_focal_mask = valid_mask & (labels.view(-1) >= 0)
    if valid_focal_mask.any():
        # 计算正确类别的预测概率
        probs = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)
        # 使用clamp确保label索引合法（ignore_index=-100会被clamp到0）
        pt = probs.gather(1, labels.view(-1, 1).clamp(min=0)).squeeze(1)
        # Focal权重：(1-pt)^gamma，预测越不准(pt小)权重越大
        focal_weight[valid_focal_mask] = (1 - pt[valid_focal_mask]) ** focal_gamma
    
    return focal_weight


def compute_average_group_loss(group_losses: dict) -> torch.Tensor:
    """
    计算所有group的平均loss（总和除以2）
    
    参数:
        group_losses: _compute_group_losses返回的字典
    
    返回:
        torch.Tensor: 所有group的loss总和除以2的结果
    
    示例:
        >>> group_losses = {
        ...     'pair_0': {'loss': tensor(1.5), ...},
        ...     'pair_1': {'loss': tensor(2.0), ...},
        ...     'summary': {...}
        ... }
        >>> avg_loss = compute_average_group_loss(group_losses)
    """
    total_loss_sum = 0.0
    count = 0
    
    for key, value in group_losses.items():
        if isinstance(value, dict) and 'loss' in value:
            total_loss_sum += value['loss'].item() * value['weighted_token_count'] * value['weight']
            # count += value['weighted_token_count'] 
            count += value['weighted_token_count'] * value['weight']

    if count == 0:
        return torch.tensor(0.0, device=next(iter(group_losses.values())).get('loss', torch.tensor(0.0)).device)
    
    return torch.tensor(total_loss_sum / (count + 1e-6) , device=next(iter(group_losses.values())).get('loss', torch.tensor(0.0)).device)


def token_weighted_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    token_pairs_weights: torch.Tensor,
    continuous_targets: torch.Tensor | None = None,
    regression_loss_type: str = 'mse',
    ignore_index: int = -100,
    reduce: bool = True,
    label_smoothing: float = 0.0,
    include_start_end_tokens: bool = True,
    include_delimiter_loss: bool = True,
    focal_gamma: float = 0.0,
    hidden_states: torch.Tensor | None = None,
    vocab_2_continues: None = None,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于token ID的加权交叉熵损失函数，支持混合监督（分类+回归）
    
    通过起始token ID和结束token ID来锁定序列中的token段，而不是使用固定的索引位置
    
    参数:
        logits: 模型输出的预测值，形状为 (batch_size, seq_len, vocab_size)
        labels: 目标标签，形状为 (batch_size, seq_len)
        input_ids: 输入token ID序列，形状为 (batch_size, seq_len)
        token_pairs_weights: token对和权重配置，形状为 (n_pairs, 6) 或 (n_pairs, 5)，每行包含:
                          [start_token_id, end_token_id, weight, alpha_coffe, beta_coffe, use_continuous_supervision]
                          - start_token_id: 起始token的ID（包含）
                          - end_token_id: 结束token的ID（包含）
                          - weight: 该段token的基础权重
                          - alpha_coffe: 当labels和input_ids对齐时使用的系数
                          - beta_coffe: 当labels和input_ids不对齐时使用的系数
                          - use_continuous_supervision: 是否使用连续监督（0=分类监督，1=连续监督，可选，默认为0）
                                                    当设置为1时，该分组会自动使用回归损失，无需手动创建mask
                          可以有多个(start_token_id, end_token_id)对，权重会叠加
        continuous_targets: 连续目标值张量，支持两种形状:
                          - 3D: (batch_size, seq_len, continuous_dim) - 每个token位置一个连续值
                          - 4D: (batch_size, seq_len, len, continuous_dim) - 每个token位置多个连续值
                            4D输入会被展平为 (batch_size*seq_len, len*dim) 进行计算
                          当需要使用回归监督时提供，对应占位位置的连续值
                          函数会根据token_pairs_weights中的use_continuous_supervision字段自动
                          为相应分组生成continuous_mask
        regression_loss_type: 回归损失类型，'mse'或'l1'，默认为'mse'
        ignore_index: 要忽略的标签索引，默认为 -100
        reduce: 是否返回标量损失，如果为 False 则返回每个样本的损失，默认为 True
        label_smoothing: 标签平滑系数，默认为 0.0
        include_start_end_tokens: 分割时是否包含开始和结束token，默认为 True
                                  True: 包含start和end位置
                                  False: 不包含start和end位置
        include_delimiter_loss: 计算loss时是否包含分隔符的loss，默认为 True
                                True: 计算所有位置的loss
                                False: 不计算分隔符（start和end token）的loss
        focal_gamma: focal loss的gamma参数，默认为 0.0（不使用focal loss）
                    设置为>0时，会根据预测概率动态调整权重：
                    预测越不准(pt小)的token权重越高，预测准确(pt大)的token权重越低
                    典型值: 2.0

    返回:
        tuple: (total_loss, group_losses)
              - total_loss: 总的加权损失
              - group_losses: 字典，包含每个token对的详细信息
                {
                    'pair_0': {'start_token_id': int, 'end_token_id': int, 'weight': float, 'loss': tensor,
                               'alpha_coffe': float, 'beta_coffe': float, 'total_tokens': int, 'weighted_token_count': float},
                    'pair_1': {'start_token_id': int, 'end_token_id': int, 'weight': float, 'loss': tensor,
                               'alpha_coffe': float, 'beta_coffe': float, 'total_tokens': int, 'weighted_token_count': float},
                    ...
                }
    
    示例:
        >>> logits = torch.randn(2, 20, 100)  # batch_size=2, seq_len=20, vocab_size=100
        >>> labels = torch.randint(0, 100, (2, 20))
        >>> input_ids = torch.tensor([
        ...     [1, 2, 3, 100, 5, 6, 101, 8, 9, 10, 100, 12, 13, 101, 15, 16, 17, 18, 19, 20],
        ...     [1, 2, 3, 100, 5, 6, 101, 8, 9, 10, 100, 12, 13, 101, 15, 16, 17, 18, 19, 20],
        ... ])
        >>> token_pairs_weights = torch.tensor([
        ...     [100, 101, 2.0, 1.0, 2.0],    # token_id 100到101之间的token权重为2.0，alpha=1.0, beta=2.0
        ...     [1, 3, 0.5, 1.0, 1.5],        # token_id 1到3之间的token权重为0.5，alpha=1.0, beta=1.5
        ... ])
        >>> total_loss, group_losses = token_weighted_cross_entropy_loss(logits, labels, input_ids, token_pairs_weights)
    """
    # ========== 第一步：输入验证和初始化 ==========
    if logits.dim() != 3:
        raise ValueError(f"logits 的维度必须是 3，但得到的是 {logits.dim()}")
    
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # 空分组处理
    if token_pairs_weights.numel() == 0:
        loss = F.cross_entropy(
            logits_flat, labels_flat,
            ignore_index=ignore_index,
            reduce=False,
            label_smoothing=label_smoothing
        )
        avg_loss = loss.mean().unsqueeze(0) if reduce else loss
        return {}, loss, torch.tensor(0.0, device=logits.device), avg_loss
    
    # ========== 第二步：提取各种掩码并收集分组信息 ==========
    
    # 基础掩码：有效位置
    valid_mask = labels_flat != ignore_index
    
    # 初始化分组信息字典
    group_info = {}
    
    # 连续监督掩码
    continuous_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=logits.device)
    if continuous_targets is not None and token_pairs_weights.shape[1] >= 6:
        for token_pair in token_pairs_weights:
            if token_pair[5].item() > 0:  # use_continuous > 0
                start_token_id = int(token_pair[0].item())
                end_token_id = int(token_pair[1].item())
                pair_mask = _compute_single_token_pair_weights(
                    input_ids, start_token_id, end_token_id,
                    batch_size, seq_len, logits.dtype, logits.device,
                    include_start_end_tokens
                ) > 0
                continuous_mask |= pair_mask.view(batch_size, seq_len)
    
    # 第二步同时收集所有分组的pair_mask
    for i, token_pair in enumerate(token_pairs_weights):
        start_token_id = int(token_pair[0].item())
        end_token_id = int(token_pair[1].item())
        weight = token_pair[2].item()
        alpha_coffe = token_pair[3].item() if token_pairs_weights.shape[1] >= 4 else 1.0
        beta_coffe = token_pair[4].item() if token_pairs_weights.shape[1] >= 5 else 1.0
        use_continuous = token_pair[5].item() if token_pairs_weights.shape[1] >= 6 else 0
        
        pair_mask = _compute_single_token_pair_weights(
            input_ids, start_token_id, end_token_id,
            batch_size, seq_len, torch.float32, logits.device,
            include_start_end_tokens
        ) > 0
        
        group_info[f"pair_{i}"] = {
            'start_token_id': start_token_id,
            'end_token_id': end_token_id,
            'weight': weight,
            'alpha_coffe': alpha_coffe,
            'beta_coffe': beta_coffe,
            'use_continuous': use_continuous,
            'pair_mask': pair_mask
        }
    
    # 分类监督掩码（非连续监督的位置）
    classification_mask = ~continuous_mask
    continuous_mask_flat = continuous_mask.view(-1)
    classification_mask_flat = classification_mask.view(-1)
    
    # 分隔符掩码
    delimiter_mask = None
    if not include_delimiter_loss:
        delimiter_mask = _create_delimiter_mask(input_ids, token_pairs_weights, batch_size, seq_len, logits.device)
    
    # ========== 第三步：计算损失 ==========
    
    per_token_loss = torch.zeros(batch_size * seq_len, dtype=logits.dtype, device=logits.device)
    
    # 1. 分类损失（离散监督位置）
    classification_positions = classification_mask_flat & valid_mask
    if classification_positions.any():
        per_token_loss[classification_positions] = F.cross_entropy(
            logits_flat[classification_positions],
            labels_flat[classification_positions],
            ignore_index=ignore_index,
            reduction='none',
            label_smoothing=label_smoothing
        )
    
    # 2. 回归损失（连续监督位置）
    regression_positions = continuous_mask_flat & valid_mask
    if regression_positions.any() and continuous_targets is not None:
        # 处理4D输入
        # if continuous_targets.dim() == 4:
        #     continuous_targets_flat = continuous_targets.view(continuous_targets.shape[0] * continuous_targets.shape[1], -1)
        # else:
        continuous_targets_flat = continuous_targets.reshape(-1, continuous_targets.shape[-1])
        hidden_states_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        # regression_targets = continuous_targets_flat[regression_positions]
        # continuous_dim = regression_targets.size(-1)
        hidden_states_flat = vocab_2_continues(hidden_states_flat[regression_positions]) if vocab_2_continues else hidden_states_flat[regression_positions]
        # hidden_states_flat = continues_norm(hidden_states_flat) if continues_norm else hidden_states_flat
        if regression_loss_type == 'mse':
            reg_loss = F.mse_loss(hidden_states_flat, continuous_targets_flat, reduction='none')
        elif regression_loss_type == 'l1':
            reg_loss = F.l1_loss(hidden_states_flat, continuous_targets_flat, reduction='none')
        else:
            raise ValueError(f"不支持的回归损失类型: {regression_loss_type}")
        
        if reg_loss.dim() > 1:
            reg_loss = reg_loss.mean(dim=-1)
        
        per_token_loss[regression_positions] = reg_loss
    
    # ========== 第四步：计算权重、加权损失和分组统计（一步到位） ==========
    
    # 基础权重
    weights = _compute_token_weights(input_ids, token_pairs_weights, batch_size, seq_len,
                                     logits.dtype, logits.device, include_start_end_tokens)
    
    # 初始化分组统计字典
    group_losses = {}
    delimiter_mask_flat = delimiter_mask.view(-1) if delimiter_mask is not None else None
    
    # 动态权重（alpha/beta系数）+ 分组统计（复用第二步收集的group_info）
    if token_pairs_weights.shape[1] >= 5:
        for group_key, info in group_info.items():
            pair_mask = info['pair_mask']
            beta_coffe = info['beta_coffe']
            alpha_coffe = info['alpha_coffe']
            use_continuous = info['use_continuous'] > 0
            
            if pair_mask.any():
                # 计算动态权重
                if not use_continuous:
                    amplify_weight = (labels_flat == input_ids.view(-1))
                    dynamic_coeff = amplify_weight.float() * beta_coffe + (~amplify_weight).float() * alpha_coffe
                else:
                    dynamic_coeff= weights
                # Focal loss
                if focal_gamma > 0:
                    if  use_continuous:
                        dynamic_coeff= weights
                    else:
                        focal_weight = _compute_focal_weights(logits, labels, valid_mask, batch_size, seq_len, focal_gamma)
                        dynamic_coeff = dynamic_coeff * focal_weight
                
                weights[pair_mask] = dynamic_coeff[pair_mask]
                
                # 计算当前group的加权损失
                all_pair_weighted_loss = (per_token_loss * weights)
                all_pair_weights = weights
                
                # 创建组的有效掩码
                group_valid_mask = valid_mask & pair_mask
                if delimiter_mask_flat is not None:
                    group_valid_mask = group_valid_mask & delimiter_mask_flat
                
                # 根据group类型选择对应的掩码（分类或回归）
                if use_continuous:
                    target_valid_mask = continuous_mask_flat & group_valid_mask
                else:
                    target_valid_mask = classification_mask_flat & group_valid_mask
                
                # 只统计对应类型的token
                # finally_target_valid_mask = target_valid_mask[target_valid_mask]
                
                # 分组统计
                total_tokens = int(target_valid_mask.sum().item())
                weighted_token_count = float(all_pair_weights[target_valid_mask].sum())
                
                if target_valid_mask.any() and weighted_token_count > 0:
                    group_loss = all_pair_weighted_loss[target_valid_mask].sum() / weighted_token_count
                else:
                    group_loss = torch.tensor(0.0, device=logits.device)
                
                # 记录group信息（只包含必要的字段）
                info=info.update({
                    'loss': group_loss,
                    'total_tokens': total_tokens,
                    'weighted_token_count': weighted_token_count,
                    'type': 'regression' if use_continuous else 'classification'
                })
    
    # 计算加权损失
    weighted_loss = per_token_loss * weights
    
    # 应用分隔符掩码
    final_valid_mask = valid_mask
    if delimiter_mask is not None:
        final_valid_mask = valid_mask & delimiter_mask_flat
   
    avg_loss = compute_average_group_loss(group_info)
    return group_info, weighted_loss, torch.tensor(0.0, device=logits.device), avg_loss




# ===== 使用示例 =====
# 详细的使用示例和文档请参考: docs/weighted_cross_entropy_usage_examples.md
# 混合监督实现文档请参考: docs/mixed_supervision_loss_implementation.md
