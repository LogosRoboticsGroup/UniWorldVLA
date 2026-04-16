"""
深度特征提取器 - 不使用VAE，使用常规CNN提取深度特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Any
from omegaconf import OmegaConf, DictConfig
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from .compressive_vq_model import Compressive_magvit_v2
from torch.utils.checkpoint import checkpoint


def nonlinearity(x):
    """激活函数"""
    return F.silu(x)


def normalization(channels):
    """归一化层"""
    return nn.GroupNorm(32, channels)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip(x) + h


class Downsample(nn.Module):
    """下采样层"""
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, x):
        return self.op(x)


class MagVITStyleDepthEncoder(nn.Module):
    """
    MagVIT风格的深度Encoder
    采用与图像Encoder相同的架构，返回多层特征用于cross-attention
    """
    def __init__(
        self,
        in_channels=1,
        ch=128,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=[2, 2, 2, 2],
        z_channels=256,
        max_att_resolution=16,
        dropout=0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.max_att_resolution = max_att_resolution
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        
        # 构建下采样模块
        self.down = nn.ModuleList()
        in_ch = ch
        for i_level in range(self.num_resolutions):
            out_ch = ch * ch_mult[i_level]
            block = nn.Module()
            block.block = nn.ModuleList()
            block.attn = nn.ModuleList()
            
            for i_block in range(num_res_blocks[i_level]):
                block.block.append(ResidualBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            
            self.down.append(block)
            
            if i_level != self.num_resolutions - 1:
                self.down[-1].downsample = Downsample(in_ch)
        
        # 中间层
        mid_ch = ch * ch_mult[-1]
        self.mid = nn.Module()
        self.mid.block_1 = ResidualBlock(mid_ch, mid_ch, dropout)
        self.mid.attn_1 = ResidualBlock(mid_ch, mid_ch, dropout)
        self.mid.block_2 = ResidualBlock(mid_ch, mid_ch, dropout)
        
        # 输出层
        self.norm_out = normalization(mid_ch)
        self.conv_out = nn.Conv2d(mid_ch, z_channels, kernel_size=3, padding=1)
        self.quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)
    
    def forward(self, x, return_features=False):
        """
        前向传播
        
        Args:
            x: 输入深度图，形状 (B*T, C, H, W)
            return_features: 是否返回中间特征（用于cross-attention）
            
        Returns:
            如果return_features=True: (h, cond_features)
            否则: h
        """
        # 初始卷积
        h = self.conv_in(x)
        
        # 收集用于条件化的特征
        cond_features = []
        
        # 下采样
        hs = [h]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(hs[-1])
                if h.shape[-2] <= self.max_att_resolution:
                    cond_features.append(h)
                hs.append(h)
        
        # 中间层
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        cond_features.append(h)
        
        # 输出
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        
        if return_features:
            return h, cond_features
        else:
            return h


class DepthFeatureExtractor(nn.Module):
    """
    深度特征提取器
    使用轻量级CNN提取深度图的特征，不依赖VAE
    """
    def __init__(
        self,
        in_channels=1,  # 深度图是单通道
        feature_dim=256,  # 输出特征维度
        hidden_dims=[64, 128, 256, 512],  # 隐藏层维度
        spatial_reduction=8,  # 空间降采样倍数 (输入尺寸 -> 输出尺寸)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.spatial_reduction = spatial_reduction
        
        # 构建编码器
        layers = []
        current_dim = in_channels
        
        for hidden_dim in hidden_dims:
            # [Conv2d -> BatchNorm -> ReLU -> MaxPool]
            layers.extend([
                nn.Conv2d(current_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 自适应池化到固定尺寸
        # 假设输入是 (B*T, 1, H, W)，输出特征图大小为 (H/spatial_reduction, W/spatial_reduction)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # 输出 16x16 特征图
        
        # 特征投影到指定维度
        final_dim = hidden_dims[-1]
        self.feature_projection = nn.Sequential(
            nn.Conv2d(final_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 归一化层
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, depth_maps):
        """
        前向传播
        
        Args:
            depth_maps: 深度图张量，形状 (B*T, 1, H, W) 或 (B, T, 1, H, W)
        
        Returns:
            depth_features: 深度特征，形状 (B*T, num_patches, feature_dim)
            其中 num_patches = 16*16 = 256
        """
        # 处理输入维度
        original_shape = depth_maps.shape
        if len(original_shape) == 5:
            # (B, T, 1, H, W) -> (B*T, 1, H, W)
            B, T, C, H, W = original_shape
            depth_maps = depth_maps.view(B * T, C, H, W)
        
        # 确保是单通道
        if depth_maps.shape[1] != 1:
            # 如果是多通道，取第一个通道
            depth_maps = depth_maps[:, :1, :, :]
        
        # 编码特征
        features = self.encoder(depth_maps)  # (B*T, C, H', W')
        
        # 自适应池化到固定尺寸
        features = self.adaptive_pool(features)  # (B*T, C, 16, 16)
        
        # 投影特征维度
        features = self.feature_projection(features)  # (B*T, feature_dim, 16, 16)
        
        # 重塑为序列格式 (B*T, num_patches, feature_dim)
        features = features.flatten(2).transpose(1, 2)  # (B*T, 256, feature_dim)
        
        # LayerNorm
        features = self.norm(features)
        
        return features


class DepthImageCrossAttention(nn.Module):
    """
    深度特征与图像特征的Cross-Attention
    深度特征作为query，图像特征作为key和value
    支持梯度检查点以减少内存占用
    """
    def __init__(
        self,
        depth_feature_dim=256,
        image_feature_dim=2048,  # 图像特征维度（来自VAE编码器）
        num_heads=8,
        dropout=0.1,
        use_adaptive_projection=True,
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        
        self.depth_feature_dim = depth_feature_dim
        self.image_feature_dim = image_feature_dim
        self.num_heads = num_heads
        self.head_dim = depth_feature_dim // num_heads
        self.use_adaptive_projection = use_adaptive_projection
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        assert depth_feature_dim % num_heads == 0, "depth_feature_dim must be divisible by num_heads"
        
        # Query投影（来自深度特征）
        self.q_proj = nn.Linear(depth_feature_dim, depth_feature_dim, bias=False)
        
        # Key和Value投影（来自图像特征）
        # 如果使用自适应投影，可以处理不同维度的图像特征
        if use_adaptive_projection:
            # 先将图像特征投影到标准维度，再投影到多头注意力维度
            self.image_feature_projection = nn.Linear(image_feature_dim, depth_feature_dim, bias=False)
            self.k_proj = nn.Linear(depth_feature_dim, depth_feature_dim, bias=False)
            self.v_proj = nn.Linear(depth_feature_dim, depth_feature_dim, bias=False)
        else:
            # 直接投影到多头注意力维度
            self.k_proj = nn.Linear(image_feature_dim, depth_feature_dim, bias=False)
            self.v_proj = nn.Linear(image_feature_dim, depth_feature_dim, bias=False)
            self.image_feature_projection = None
        
        # 输出投影
        self.out_proj = nn.Linear(depth_feature_dim, depth_feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm (Pre-norm风格)
        self.norm_q = nn.LayerNorm(depth_feature_dim)
        self.norm_k = nn.LayerNorm(depth_feature_dim)
        self.norm_v = nn.LayerNorm(depth_feature_dim)
        self.norm1 = nn.LayerNorm(depth_feature_dim)  # 用于FFN前
        self.norm2 = nn.LayerNorm(depth_feature_dim)  # 用于FFN后
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(depth_feature_dim, depth_feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(depth_feature_dim * 4, depth_feature_dim),
            nn.Dropout(dropout)
        )
        
    def _forward_core(self, depth_features, image_features, attention_mask=None):
        """
        核心前向逻辑（用于梯度检查点）
        """
        residual = depth_features
        
        # Pre-norm: 在投影Q/K/V之前先norm输入
        depth_features_norm = self.norm_q(depth_features)
        
        # 如果使用自适应投影，先将图像特征投影到标准维度
        if self.use_adaptive_projection and self.image_feature_projection is not None:
            image_features = self.image_feature_projection(image_features)
        
        # Pre-norm: K/V也先norm
        image_features_norm_k = self.norm_k(image_features)
        image_features_norm_v = self.norm_v(image_features)
        
        # Multi-head attention
        # Query来自深度特征（使用norm后的）
        q = self.q_proj(depth_features_norm)  # (B, num_depth_patches, depth_feature_dim)
        # Key和Value来自图像特征（使用norm后的）
        k = self.k_proj(image_features_norm_k)  # (B, num_image_patches, depth_feature_dim)
        v = self.v_proj(image_features_norm_v)  # (B, num_image_patches, depth_feature_dim)
        
        # 重塑为多头形式
        B, num_depth_patches, D = q.shape
        num_image_patches = image_features.shape[1]
        
        q = q.view(B, num_depth_patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, num_image_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, num_image_patches, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 内存优化：使用scaled_dot_product_attention（PyTorch 2.0+）
        # 这样可以利用Flash Attention等优化，减少内存占用
        try:
            # 使用PyTorch内置的内存优化attention实现
            attended = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        except AttributeError:
            # 降级到原始实现
            # 计算注意力分数
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # 应用注意力掩码（如果有）
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            # Softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 应用注意力权重
            attended = torch.matmul(attn_weights, v)
        
        # 合并多头
        attended = attended.transpose(1, 2).contiguous().view(B, num_depth_patches, D)
        
        # 输出投影
        attended = self.out_proj(attended)
        attended = self.dropout(attended)
        
        # Pre-norm风格: 只做残差连接，不需要norm1
        features = attended + residual
        
        # FFN (Pre-norm风格)
        residual = features
        features_norm = self.norm1(features)  # FFN前norm
        features = self.ffn(features_norm)
        features = features + residual  # 残差连接
        features = self.norm2(features)  # FFN后norm
        
        return features
    
    def forward(self, depth_features, image_features, attention_mask=None):
        """
        前向传播（优化内存版本，支持梯度检查点）
        
        Args:
            depth_features: 深度特征，形状 (B, num_depth_patches, depth_feature_dim)
            image_features: 图像特征，形状 (B, num_image_patches, image_feature_dim)
            attention_mask: 注意力掩码 (可选)
        
        Returns:
            attended_features: 交互后的特征，形状 (B, num_depth_patches, depth_feature_dim)
        """
        # 使用梯度检查点减少内存占用
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                self._forward_core,
                depth_features,
                image_features,
                attention_mask,
                use_reentrant=False
            )
        else:
            return self._forward_core(depth_features, image_features, attention_mask)


class DepthFeatureExtractorWithAttention(nn.Module):
    """
    完整的深度特征提取和交互模块
    """
    def __init__(
        self,
        depth_config,
        image_feature_dim=2048,
        num_heads=8,
        device='cuda',
    ):
        super().__init__()
        
        self.device = device
        
        # 深度特征提取器
        self.depth_extractor = DepthFeatureExtractor(
            in_channels=depth_config.get('in_channels', 1),
            feature_dim=depth_config.get('feature_dim', 256),
            hidden_dims=depth_config.get('hidden_dims', [64, 128, 256, 512]),
            spatial_reduction=depth_config.get('spatial_reduction', 8),
        )
        
        # Cross-Attention模块
        self.cross_attention = DepthImageCrossAttention(
            depth_feature_dim=depth_config.get('feature_dim', 256),
            image_feature_dim=image_feature_dim,
            num_heads=num_heads,
            dropout=depth_config.get('dropout', 0.1),
        )
        
    def forward(self, depth_maps, image_features=None, attention_mask=None):
        """
        前向传播
        
        Args:
            depth_maps: 深度图，形状 (B, T, 1, H, W) 或 (B*T, 1, H, W)
            image_features: 图像特征，形状 (B, num_image_patches, image_feature_dim)
            attention_mask: 注意力掩码
        
        Returns:
            depth_features: 提取的深度特征
            attended_features: 与图像特征交互后的特征（如果提供image_features）
        """
        # 提取深度特征
        depth_features = self.depth_extractor(depth_maps)
        
        # 如果提供了图像特征，进行cross-attention
        if image_features is not None:
            attended_features = self.cross_attention(depth_features, image_features, attention_mask)
            return depth_features, attended_features
        
        return depth_features, None


class DepthCompressiveVQ(Compressive_magvit_v2,nn.Module):
    """
    继承Compressive_magvit_v2的深度特征提取器
    专门用于深度图的特征提取，复用图像tokenizer的context和dynamic encoder逻辑
    """
    def __init__(
        self,
        config_exps: Union[dict, DictConfig],
        uni_prompting=None,
    ):
        # 调用父类初始化，从config中读取参数
        # config_exps.model.cond_enable = False
        super().__init__(
            config_exps=config_exps,  # type: ignore[arg-type]
            num_vq_embeddings=config_exps['model']['vq_model'].get('num_vq_embeddings', 256),
            num_dyn_embeddings=config_exps['model']['vq_model'].get('num_dyn_embeddings', 256)
        )
        self.context_length = config_exps.dataset.ctd.context_length
        self.uni_prompting = uni_prompting
        
        # 训练阶段控制状态
        self.enable_context = True   # 是否启用 context 模块
        self.enable_dynamic = True   # 是否启用 dynamic 模块
        
        # 修改context encoder和dynamic encoder的输入通道为1（深度图）
        # self.context_vq_model.encoder.conv_in = nn.Conv2d(1, 128, kernel_size=3, padding=1).to(device)
        # self.dynamic_vq_model.encoder.conv_in = nn.Conv2d(1, 128, kernel_size=3, padding=1).to(device)
    
    def set_context_dynamic_enabled(self, enable_context: bool, enable_dynamic: bool):
        """
        设置 context 和 dynamic 模块的启用状态
        
        Args:
            enable_context: 是否启用 context 模块（soi/eoi区间）
            enable_dynamic: 是否启用 dynamic 模块（sod/eod区间）
        """
        self.enable_context = enable_context
        self.enable_dynamic = enable_dynamic
    
    def tokenize_depth(
        self,
        context_depth: torch.Tensor,
        dynamic_depth: torch.Tensor,
        return_encoder_features: bool = True
    ):
        """
        深度tokenize方法，调用父类的tokenize实现
        根据 enable_context 和 enable_dynamic 状态决定是否计算对应的特征
        
        需要将单通道的深度图扩展为3通道以兼容父类实现
        
        训练阶段控制：
        - 如果 self.enable_context=False，则跳过 context 的特征计算（节省计算资源）
        - 如果 self.enable_dynamic=False，则跳过 dynamic 的特征计算（节省计算资源）
        
        Args:
            context_depth: context深度数据 [B*Tc, 1, H, W]
            dynamic_depth: dynamic深度数据 [B*Td, 1, H, W]
            return_encoder_features: 是否返回encoder的多层特征
            
        Returns:
            返回父类tokenize的结果 (indices, labels) 或 (indices, labels, encoder_features)
        """
        # 扩展单通道为3通道以兼容父类实现
        context_depth_3ch = context_depth.repeat(1,1, 3, 1, 1)
        dynamic_depth_3ch = dynamic_depth.repeat(1,1, 3, 1, 1)
        
        # 根据训练阶段决定是否计算对应特征
        if not self.enable_context and not self.enable_dynamic:
            # 两个都禁用：返回空结果（不推荐，但需要处理）
            import torch
            
            return None, None
        
        elif  self.enable_context and not self.enable_dynamic:
            # 只启用 context：使用 only_context_vq=True
            return super().tokenize(
                pixel_values=context_depth_3ch,  # type: ignore[arg-type]
                context_pixel_values=context_depth_3ch,
                context_length=self.context_length,
                special_token=self.uni_prompting.sptids_dict,
                only_context_vq=False,  # 只处理 context
                is_img=False,
                return_encoder_features=return_encoder_features,
                return_stage=1
            )
        else: #both
            # 启用 dynamic（context可选）：正常处理
            return super().tokenize(
                pixel_values=dynamic_depth_3ch,  # type: ignore[arg-type]  # 确保是Float类型
                context_pixel_values=context_depth_3ch ,
                # context_pixel_values=context_depth_3ch if self.enable_context else None,
                context_length=self.context_length,
                special_token=self.uni_prompting.sptids_dict,
                only_context_vq=False,
                is_img=False,  # 标记为深度数据
                return_encoder_features=return_encoder_features,
                return_stage=2
            )


# 默认配置
def get_default_depth_config():
    """
    获取默认的深度特征提取配置
    """
    return {
        'in_channels': 1,
        'feature_dim': 256,
        'hidden_dims': [64, 128, 256, 512],
        'spatial_reduction': 8,
        'dropout': 0.1,
    }