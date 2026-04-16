"""
深度数据处理工具模块
提供深度数据的归一化和倒数变换功能
"""

import torch
import torchvision.transforms as transforms


def tensor_norm(norm_layer: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """
    使用可训练的Layer Normalization对tensor进行归一化处理
    
    Args:
        norm_layer: LayerNorm层对象
        tensor: 输入张量，期望形状为 (batch_size, sequence_length, channels, height, width)
        
    Returns:
        torch.Tensor: Layer Normalization后的张量
    """
    # 将张量重塑为适合LayerNorm的形状
    original_shape = tensor.shape
    if len(original_shape) == 5:
        # (B, S, C, H, W) -> (B*S, C, H, W)
        bs, s, c, h, w = original_shape
        tensor_reshaped = tensor.view(-1, c, h, w)

        # 应用LayerNorm
        normalized = norm_layer(tensor_reshaped)

        # 恢复原始形状
        return normalized.view(bs, s, c, h, w)
    else:
        # 对于其他形状，直接应用LayerNorm
        return norm_layer(tensor)


def apply_depth_processing(
    tensor: torch.Tensor,
    norm_layer: torch.nn.Module,
    submode: str = "norm",
    eps: float = 1e-6
) -> torch.Tensor:
    """
    应用深度数据处理，支持不同的处理模式
    
    Args:
        tensor: 输入张量，期望形状为 (batch_size, sequence_length, channels, height, width)
        norm_layer: LayerNorm层对象
        submode: 处理模式，支持以下选项:
            - "norm": 仅归一化（原始方式）
            - "inverse": 仅取倒数
            - "inverse_norm": 先取倒数，再归一化
            - "norm_inverse": 先归一化，再取倒数
        eps: 防止除零的小常数，用于倒数计算
        
    Returns:
        torch.Tensor: 处理后的张量
        
    Raises:
        ValueError: 当submode不支持时抛出异常
    """
    if submode == "norm":
        # 方案1: 仅归一化（原始方式）
        return tensor_norm(norm_layer, tensor)
    
    elif submode == "inverse":
        # 方案2: 仅取倒数
        return 1.0 / (tensor + eps)
    
    elif submode == "inverse_norm":
        # 方案3: 先取倒数，再归一化
        inverted = 1.0 / (tensor + eps)
        return tensor_norm(norm_layer, inverted)
    
    elif submode == "norm_inverse":
        # 方案4: 先归一化，再取倒数
        normalized = tensor_norm(norm_layer, tensor)
        return 1.0 / (normalized + eps)
    elif submode == "norm_transform":
        tensor = tensor / 200.0
        return transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)(tensor)
    else:
        raise ValueError(
            f"Unsupported submode: {submode}. "
            f"Supported modes are: 'norm', 'inverse', 'inverse_norm', 'norm_inverse'"
        )


def batch_depth_processing(
    prev_tensor: torch.Tensor,
    next_tensor: torch.Tensor,
    norm_layer_c: torch.nn.Module,
    depth_norm_layer_d: torch.nn.Module,
    submode: str = "norm",
    eps: float = 1e-6
) -> tuple:
    """
    批量应用深度数据处理到prev和next数据
    
    Args:
        prev_tensor: prev数据张量
        next_tensor: next数据张量
        norm_layer: LayerNorm层对象
        submode: 处理模式
        eps: 防止除零的小常数
        
    Returns:
        tuple: (处理后的prev_tensor, 处理后的next_tensor)
    """
    prev_processed = apply_depth_processing(prev_tensor, norm_layer_c, submode, eps)
    if depth_norm_layer_d is None:
        next_processed = apply_depth_processing(next_tensor, norm_layer_c, submode, eps)
    else:
        next_processed = apply_depth_processing(next_tensor, depth_norm_layer_d, submode, eps)
    
    return prev_processed, next_processed