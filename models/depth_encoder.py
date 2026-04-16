from typing import Union, Optional, TYPE_CHECKING
import os
import json
import pickle
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from matplotlib import cm
from DA3.Basic_usage import DA3Main
from data_utils.cache_utils import get_depth_cache_path, get_token_hash
from models.depth_utils import  batch_depth_processing
from models.depth_cross_attention_utils import DepthCrossAttentionModule,DepthCrossAttentionModuleResidual
if TYPE_CHECKING:
    from DA3.src.depth_anything_3.specs import Prediction

# 动态导入以避免循环依赖
def get_prediction_class():
    try:
        from DA3.src.depth_anything_3.specs import Prediction
        return Prediction
    except ImportError:
        return None


class DepthEncoder(torch.nn.Module):
    def __init__(
        self,
        vq_model,
        condition_len,
        uni_prompting,
        device=None,
        args=None,
        data_loader=None,
        contex_norm=[],
        dynamic_norm=[],
        config=None,
        use_dual_vit=True,
    ):
        super().__init__()
        self.args = args
        self.use_dual_vit = use_dual_vit  # 存储配置
        self.device = device if device is not None else 'cuda'
        self.__dict__['da3_main'] = DA3Main(device=self.device, model_config=config)
        self.__dict__['vq_model'] = vq_model
        self.condition_len = condition_len
        self.uni_prompting = uni_prompting

        # 深度数据存储配置
        self.depth_cache_dir = getattr(args, "cache_dir", "./depth_cache")
        self.use_cached_depth = getattr(args, "use_cached_depth", False)
        self.save_depth_cache = getattr(args, "use_cached_depth", False)

        # DataLoader引用，用于复用加载数据逻辑
        self.data_loader = data_loader

        # 创建缓存目录
        os.makedirs(self.depth_cache_dir, exist_ok=True)

        # 初始化可训练的Layer Normalization，使用传入的参数
        self.depth_norm_layer_c = torch.nn.LayerNorm(normalized_shape=[1, contex_norm[0], contex_norm[1]])
        self.depth_norm_layer_d = torch.nn.LayerNorm(normalized_shape=[1, dynamic_norm[0], dynamic_norm[1]])
        
        from data_utils.depth_loader import DepthloaderOneimage
        self.depth_loader = DepthloaderOneimage(device, args, data_loader, self)
        # 初始化深度特征提取器（不使用VAE）
        
       
        
        # MagVIT风格的深度tokenizer（独立参数，不与图像共用）
        from models.depth_feature_extractor import DepthCompressiveVQ, DepthImageCrossAttention
        
        
        self.magvit_dual_encoder = DepthCompressiveVQ(
            config_exps=config,
            uni_prompting=uni_prompting,
        )
        if not use_dual_vit:
            from models.depth_feature_extractor import DepthFeatureExtractorWithAttention, get_default_depth_config
            # Cross-Attention模块（用于与图像特征交互）
            # 添加梯度检查点支持以减少内存占用
            self.depth_cross_attention = DepthImageCrossAttention(
                depth_feature_dim=256,
                image_feature_dim=2048,
                num_heads=8,
                dropout=0.1,
                use_adaptive_projection=False,
            )
            
            depth_config = get_default_depth_config()
            self.depth_feature_extractor = DepthFeatureExtractorWithAttention(
                depth_config=depth_config,
                image_feature_dim=2048,  # 对应mm_projector的输出维度
                num_heads=8,
                device=device
            )
        else:
            # MLP投影层：将 context_attended 和 dynamic_attended 投影到2048维
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(512, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )
            # self.context_norm = torch.nn.LayerNorm(2048)
            # self.dynamic_norm = torch.nn.LayerNorm(2048)
            self.dynamic_mlp = torch.nn.Sequential(
                torch.nn.Linear(2048, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )
                       
            self.depth_cross_attention_module = DepthCrossAttentionModule(2048,2048)
            self.residual_net = DepthCrossAttentionModuleResidual(2048,2048)
            self.use_resudial = config.training.use_residual

    def residula_net_forward(self,input_embeddings,cur_input_ids,hidden_states):
        if not self.use_resudial:
            return hidden_states
        if input_embeddings is not None and cur_input_ids is not None:
            input_embeddings_mock = self.rebuild_depth_embeddings_for_match_mock(input_embeddings)  
            hidden_states = self.residual_net(hidden_states,cur_input_ids,input_embeddings_mock, self.uni_prompting.sptids_dict)
        return hidden_states

    def depth_forward(self,depth_embeddings,cur_input_ids,input_embeddings):
        if depth_embeddings is not None and cur_input_ids is not None:
            depth_embeddings = self.rebuild_depth_embeddings(depth_embeddings)  
            # if depth_embeddings["index"] == 0:
            #     cur_input_ids = cur_input_ids[:,:1521]
            input_embeddings = self.depth_cross_attention_module(input_embeddings,cur_input_ids,depth_embeddings, self.uni_prompting.sptids_dict)
        return input_embeddings
    
    def replace_depth_tokens(self, input_ids, input_embeddings, depth_embeddings, sptids_dict,start_index):
        """
        替换深度 token 的逻辑：在 inputs 中检索到占位符时，用深度的 embedding 来替换
        
        Args:
            input_ids: 原始的 token IDs (batch_size, seq_len)
            input_embeddings: 所有 token 的 embedding (batch_size, seq_len, hidden_size)
            depth_embeddings: 深度 token 的 embedding (batch_size, 1, hidden_size)
            sptids_dict: 特殊 token 的字典，包含 <|sop|> 和 <|eop|>
        
        Returns:
            替换后的 input_embeddings
        """
        batch_size = input_ids.size(0)
        sop_token_id = sptids_dict['<|sop|>'].to(input_ids.device)
        eop_token_id = sptids_dict['<|eop|>'].to(input_ids.device)
        
        # 遍历 batch 中的每个样本
        for batch_idx in range(batch_size):
            # 找到所有的 <|sop|> 和 <|eop|> 位置
            sop_positions = torch.where(input_ids[batch_idx] == sop_token_id)[0]
            eop_positions = torch.where(input_ids[batch_idx] == eop_token_id)[0]
            
            # 确保成对出现
            num_pairs = min(sop_positions.size(0), eop_positions.size(0))
            if num_pairs == 0:
                continue
            
            assert num_pairs >= depth_embeddings.shape[1], f"depth_embeddings.shape[0]: {depth_embeddings.shape[0]}, num_pairs: {num_pairs}"
            # 对每一对 <|sop|> 和 <|eop|> 进行替换
            for pair_idx in range(depth_embeddings.shape[1]):
                sop_pos = sop_positions[pair_idx].item()
                eop_pos = eop_positions[pair_idx].item()
                
                # 验证位置关系
                if eop_pos <= sop_pos + 1:
                    assert eop_pos > sop_pos + 1 , f"assert eop_pos > sop_pos + 1"
                    continue
                if start_index > pair_idx: continue
                # 替换 <|sop|> 和 <|eop|> 之间的所有位置（不包括边界）
                # 用深度 embedding 的第一个时间步的 embedding 来填充
                depth_emb = depth_embeddings[batch_idx, pair_idx:pair_idx + 1, :].expand(1, eop_pos - sop_pos - 1, -1)
                input_embeddings[batch_idx, sop_pos + 1 : eop_pos] = depth_emb
        
        return input_embeddings
    
    def repalce_inputs_and_labeles(self, inputs,input_embeddings,depth_embeddings, sptids_dict):
        """
        替换输入和标签中的特殊 token
        """
        pre_1s_context_attended = depth_embeddings["prev_1s_context_attended"][:,1:2,...]
        pre_2s_context_attended = depth_embeddings["prev_2s_context_attended"][:,1:2,...]
        next_context_attended = depth_embeddings["next_context_attended"][:,1:2,...]
        merged_attended = torch.cat([ pre_2s_context_attended, next_context_attended], dim=1)
        next_attended = depth_embeddings["next_dynamic_attended"][:,:10]
        input_embeddings = self.replace_depth_tokens(inputs, input_embeddings, merged_attended, sptids_dict,start_index=0)
        input_embeddings = self.replace_depth_tokens(inputs, input_embeddings, next_attended, sptids_dict,start_index=2)
        
        # labels_embeddings = self.replace_depth_tokens(inputs, input_embeddings, depth_embeddings, sptids_dict)
        return input_embeddings
    
    def rebuild_depth_embeddings(self, depth_embeddings):
        """
        重建 depth embeddings
        """
        if depth_embeddings["prev_1s_context_attended"] is not None and self.enable_context:
            pre_1s_context_attended = depth_embeddings["prev_1s_context_attended"][:,1:2,...]
            pre_2s_context_attended = depth_embeddings["prev_2s_context_attended"][:,1:2,...]
            next_context_attended = depth_embeddings["next_context_attended"][:,1:2,...]
            merged_attended = torch.cat([ pre_2s_context_attended, next_context_attended], dim=1)
        else:
            merged_attended = None
        if "prev_1s_dynamic_attended" in depth_embeddings and depth_embeddings["prev_1s_dynamic_attended"] is not None and self.enable_dynamic:
            pre_1s_dynamic_attended = depth_embeddings["prev_1s_dynamic_attended"][:,:9,...]
            pre_2s_dynamic_attended = depth_embeddings["prev_2s_dynamic_attended"][:,:9,...]
            next_attended = depth_embeddings["next_dynamic_attended"][:,:8]
            dynamic_attended = torch.cat([ pre_1s_dynamic_attended,pre_2s_dynamic_attended, next_attended], dim=1)
        else:
            dynamic_attended = None
         # plot_attention_mask(attention_mask_step_input[0,0], save_path=str(len(genframe_token)) + "navsim_gen.png")
         #inference
           
        depth_embeddings["context_attended"]=merged_attended
        depth_embeddings["dynamic_attended"]=dynamic_attended
        if "dynamic_attended" in depth_embeddings.keys() and depth_embeddings["dynamic_attended"] is not None:
            if depth_embeddings["index"]== 0:
                depth_embeddings["dynamic_attended"]=depth_embeddings["dynamic_attended"][:,:19]
            elif depth_embeddings["index"] >0 and depth_embeddings["index"] <= 9:
                depth_embeddings["dynamic_attended"]=depth_embeddings["dynamic_attended"][:,18+depth_embeddings["index"]:19+depth_embeddings["index"]]
            
        if "context_attended" in depth_embeddings.keys() and depth_embeddings["context_attended"] is not None:
            if depth_embeddings["index"] > 0:
                depth_embeddings["context_attended"]=None
        return depth_embeddings
    
    def rebuild_depth_embeddings_for_match_mock(self, depth_embeddings):
        out_depth_embeddings = {}
        if self.enable_context and self.use_resudial:
            out_depth_embeddings["context_attended"]=depth_embeddings
        else:
            out_depth_embeddings["context_attended"]=None
        if self.enable_dynamic and self.use_resudial:
            out_depth_embeddings["dynamic_attended"]=depth_embeddings
        else:
            out_depth_embeddings["dynamic_attended"]=None
        return out_depth_embeddings

    @torch.no_grad()
    def depth_tokenizer_context_dynamic(
        self,
        prev_depth_context_1s: Union[torch.FloatTensor, torch.LongTensor],
        prev_depth_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor],
        prev_depth_context_2s: Union[torch.FloatTensor, torch.LongTensor],
        prev_depth_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor],
        next_depth_context: Union[torch.FloatTensor, torch.LongTensor],
        next_depth_dynamic: Union[torch.FloatTensor, torch.LongTensor],
        submode="norm_transform",
    ):
        """
        直接使用深度的context和dynamic数据进行tokenize
        tokenize 调用方式与 prepare_inputs_and_labels 保持一致
        
        Args:
            prev_depth_context_1s: 1秒前的context深度数据 [B, T_context, H, W]
            prev_depth_dynamic_1s: 1秒前的dynamic深度数据 [B, T_dynamic, H, W]
            prev_depth_context_2s: 2秒前的context深度数据 [B, T_context, H, W]
            prev_depth_dynamic_2s: 2秒前的dynamic深度数据 [B, T_dynamic, H, W]
            next_depth_context: 未来的context深度数据 [B, T_context, H, W]
            next_depth_dynamic: 未来的dynamic深度数据 [B, T_dynamic, H, W]
            submode: 处理模式，支持 "norm", "inverse", "inverse_norm", "norm_inverse"
        
        Returns:
            tuple: (input_ids_prev1s, labels_prev1s, input_ids_prev2s, labels_prev2s,
                    input_ids_next, labels_ids_next)
        """
        uni_prompting = self.uni_prompting
        device = self.device
        condition_len = self.condition_len
        
        # 1. 将数据移到设备
        prev_depth_context_1s = prev_depth_context_1s.to(device, non_blocking=True)
        prev_depth_dynamic_1s = prev_depth_dynamic_1s.to(device, non_blocking=True)
        prev_depth_context_2s = prev_depth_context_2s.to(device, non_blocking=True)
        prev_depth_dynamic_2s = prev_depth_dynamic_2s.to(device, non_blocking=True)
        next_depth_context = next_depth_context.to(device, non_blocking=True)
        next_depth_dynamic = next_depth_dynamic.to(device, non_blocking=True)
        
        # 2. 扩展通道维度 [B, T, H, W] -> [B, T, 3, H, W]
        prev_depth_context_1s = prev_depth_context_1s.repeat(1, 1, 3, 1, 1)
        prev_depth_dynamic_1s = prev_depth_dynamic_1s.repeat(1, 1, 3, 1, 1)
        prev_depth_context_2s = prev_depth_context_2s.repeat(1, 1, 3, 1, 1)
        prev_depth_dynamic_2s = prev_depth_dynamic_2s.repeat(1, 1, 3, 1, 1)
        next_depth_context = next_depth_context.repeat(1, 1, 3, 1, 1)
        next_depth_dynamic = next_depth_dynamic.repeat(1, 1, 3, 1, 1)
        
        # 3. 扩充next的dynamic数据（与原有逻辑一致）
        # next_depth_dynamic = torch.cat([prev_depth_dynamic_2s[:,-2:,...], next_depth_dynamic], dim=1)
        
        # 4. 使用 batch_depth_processing 应用归一化处理
        # 支持的submode: "norm", "inverse", "inverse_norm", "norm_inverse"
        prev_depth_context_1s, prev_depth_dynamic_1s = batch_depth_processing(
            prev_depth_context_1s,
            prev_depth_dynamic_1s,
            self.depth_norm_layer_c,
            self.depth_norm_layer_d,
            submode=submode,
            eps=1e-6
        )
        prev_depth_context_2s, prev_depth_dynamic_2s = batch_depth_processing(
            prev_depth_context_2s,
            prev_depth_dynamic_2s,
            self.depth_norm_layer_c,
            self.depth_norm_layer_d,
            submode=submode,
            eps=1e-6
        )
        next_depth_context, next_depth_dynamic = batch_depth_processing(
            next_depth_context,
            next_depth_dynamic,
            self.depth_norm_layer_c,
            self.depth_norm_layer_d,
            submode=submode,
            eps=1e-6
        )
        
        token_mode = "truth"
        c_latent_size = 0
        # start_time = time.time()
        
        # 5. 处理prev2s - 参考prepare_inputs_and_labels的调用方式
        if token_mode == "mock":
            # Mock input_ids_prev2s和labels_prev2s - 前三帧使用真实数据，后面mock
            # 先合并context和dynamic为完整特征
            prev2s_feat = torch.cat([prev_depth_context_2s, prev_depth_dynamic_2s], dim=1)
            input_ids_prev2s, labels_prev2s, c_latent_size = self._mock_tokenize_with_partial_real(
                input_ids_feat=prev2s_feat,
                uni_prompting=uni_prompting,
                real_frames=3,
            )
        else:
            # 使用与prepare_inputs_and_labels一致的参数方式
            input_ids_prev2s, labels_prev2s = self.vq_model.tokenize(
                prev_depth_dynamic_2s,
                context_pixel_values=prev_depth_context_2s,
                context_length=condition_len,
                special_token=uni_prompting.sptids_dict,
                # only_context_vq=True,
                is_img=False,
                return_label=False,
            )
        
        # 6. 处理next
        # start_time = time.time()
        # 使用与prepare_inputs_and_labels一致的参数方式
        input_ids_next, labels_ids_next = self.vq_model.tokenize(
            next_depth_dynamic,
            context_pixel_values=next_depth_context,
            context_length=condition_len,
            special_token=uni_prompting.sptids_dict,
            # only_context_vq=True,
            is_img=False,
            return_label=True,
        )
        
        # 7. 处理prev1s
        # start_time = time.time()
        if token_mode == "mock":
            # Mock input_ids_prev1s和labels_prev1s - 直接全mock
            batch_size = prev_depth_dynamic_1s.shape[0]
            # 计算序列长度：dynamic帧数
            seq_len_prev1s = prev_depth_dynamic_1s.shape[1]
            input_ids_prev1s, labels_prev1s = self._mock_tokenize_all(
                batch_size=batch_size,
                seq_len=seq_len_prev1s,
                c_latent_size=c_latent_size,
                device=prev_depth_dynamic_1s.device,
            )
        else:
            # 使用与prepare_inputs_and_labels一致的参数方式
            input_ids_prev1s, labels_prev1s = self.vq_model.tokenize(
                prev_depth_dynamic_1s,
                context_pixel_values=prev_depth_context_1s,
                context_length=condition_len,
                special_token=uni_prompting.sptids_dict,
                # only_context_vq=False,
                is_img=False,
                return_label=False,
            )
        
        return (
            input_ids_prev1s,
            labels_prev1s,
            input_ids_prev2s,
            labels_prev2s,
            input_ids_next,
            labels_ids_next,
        )

    @torch.no_grad()
    def prepare_depth_inputs_and_labels(
        self,
        prev_img_context_1s: Union[torch.FloatTensor, torch.LongTensor],
        prev_img_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor],
        prev_img_context_2s: Union[torch.FloatTensor, torch.LongTensor],
        prev_img_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor],
        next_img_context: Union[torch.FloatTensor, torch.LongTensor],
        next_img_dynamic: Union[torch.FloatTensor, torch.LongTensor],
        prev_depth_img_input: Union[torch.FloatTensor, torch.LongTensor],
        next_depth_img_input: Union[torch.FloatTensor, torch.LongTensor],
        cached_prev_depth=None,
        cached_next_depth=None,
        token_id: Union[list, None] = None,
        prev_img_depth_tokenid: Union[torch.Tensor, None] = None,
        next_img__depth_tokenid: Union[torch.Tensor, None] = None,
    ):
        """准备深度输入数据"""
        batch_size = prev_img_context_1s.shape[0] if len(prev_img_context_1s.shape) > 0 else 1
        prev_img_depth, next_depth_img = None, None
        # 处理b,t维度的tokenid
        prev_img_depth, next_depth_img = self.depth_loader.process(
            token_id,
            prev_img_depth_tokenid,
            next_img__depth_tokenid,
            prev_depth_img_input,
            next_depth_img_input,
            cached_prev_depth,
            cached_next_depth,
        )
        # with torch.autocast("cuda", enabled=False):  # 关键
        return self.depth_tokenizer(prev_img_depth, next_depth_img)

    def depth_encoder(
        self,
        prev_depth_img_input: torch.Tensor,
        next_depth_img_input: torch.Tensor,
        next_depth_context_img_input: torch.Tensor,
    ): 
        prev_img_depth = self.da3_main.da3_infer(prev_depth_img_input)
        next_depth_img = self.da3_main.da3_infer(next_depth_img_input)
        # for i in range(next_depth_context_img_input.shape[1]):
        b,context_time,context_len,c,w,h = next_depth_context_img_input.shape
        next_depth_context_img = self.da3_main.da3_infer(next_depth_context_img_input.reshape(-1,context_len,c,w,h))
            
        # 1s
        return prev_img_depth, next_depth_img, next_depth_context_img

    def _depth_encoder(
        self,
        prev_depth_img_input: torch.Tensor,
        
    ): 
        prev_img_depth = self.da3_main.da3_infer(prev_depth_img_input)
        # 1s
        return prev_img_depth


    def _mock_tokenize_with_partial_real(
        self,
        input_ids_feat,
        uni_prompting,
        start_idx=None,
        end_idx=None,
        real_frames=3,
    ):
        """
        对数据进行部分真实tokenize和部分mock tokenize
        
        Args:
            input_ids_feat: 输入特征张量
            uni_prompting: 提示词配置
            start_idx: 起始索引，如果为None则从中间开始
            end_idx: 结束索引
            real_frames: 使用真实数据的帧数
        
        Returns:
            tuple: (input_ids, labels)
        """
        # 提取需要真实tokenize的数据
        if start_idx is None:
            start_idx = input_ids_feat.shape[1] // 2
        if end_idx is None:
            end_idx = start_idx + real_frames
            
        real_data = input_ids_feat[:, start_idx:end_idx, ...]
        
        # 对前几帧进行真实的tokenize
        input_ids_real, labels_real = self.vq_model.tokenize(
            pixel_values=real_data,
            special_token=uni_prompting.sptids_dict,
            only_context_vq=True,
            is_img=False,
            return_label=False,
        )

        # 计算需要mock的部分长度
        batch_size = input_ids_feat.shape[0]
        if end_idx is not None and start_idx is not None:
            total_len = input_ids_feat.shape[1] - start_idx
        else:
            total_len = input_ids_feat.shape[1] // 2
            
        seq_len = end_idx - start_idx if end_idx is not None else total_len
        mock_len = max(0, total_len - real_frames)

        # 从真实数据中获取结构信息
        real_context = input_ids_real["context"]
        real_dynamic = input_ids_real["dynamic"]
        real_labels = labels_real["dynamic"]
        c_latent_size = real_context.shape[-1] // real_frames

        # 创建mock数据
        mock_context = torch.zeros(
            batch_size,
            mock_len * c_latent_size,
            dtype=torch.long,
            device=input_ids_feat.device,
        )
        mock_dynamic = torch.zeros(
            batch_size,
            mock_len * c_latent_size,
            dtype=torch.long,
            device=input_ids_feat.device,
        )
        mock_labels = torch.zeros(
            batch_size,
            mock_len * c_latent_size,
            dtype=torch.long,
            device=input_ids_feat.device,
        )

        # 拼接真实数据和mock数据
        input_ids = {
            "context": torch.cat([real_context, mock_context], dim=1),
            "dynamic": torch.cat([real_dynamic, mock_dynamic], dim=1),
        }
        labels = {"dynamic": torch.cat([real_labels, mock_labels], dim=1)}
        
        return input_ids, labels, c_latent_size

    def _mock_tokenize_all(self, batch_size, seq_len, c_latent_size, device):
        """
        创建全mock的tokenize结果
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            c_latent_size: latent大小
            device: 设备
        
        Returns:
            tuple: (input_ids, labels)
        """
        input_ids = {
            "context": torch.zeros(
                batch_size,
                seq_len * c_latent_size,
                dtype=torch.long,
                device=device,
            ),
            "dynamic": torch.zeros(
                batch_size,
                (seq_len - 1) * c_latent_size,
                dtype=torch.long,
                device=device,
            ),
        }
        labels = {
            "dynamic": torch.zeros(
                batch_size,
                (seq_len - 1) * c_latent_size,
                dtype=torch.long,
                device=device,
            )
        }
        return input_ids, labels

    def depth_tokenizer(
        self, prev_depth_all, next_depth_all, mode="depth", submode="norm"
    ):
        condition_len = self.condition_len
        uni_prompting = self.uni_prompting

        if mode == "depth":
            # 按需提取数据字段（默认取第一个可用字段）
            prev_depth_data = self._extract_depth_field(prev_depth_all, "prev")
            next_depth_data = self._extract_depth_field(next_depth_all, "next")
            prev_depth_data = prev_depth_data.to(self.device, non_blocking=True)
            next_depth_data = next_depth_data.to(self.device, non_blocking=True)

            input_ids_prev_feat = prev_depth_data.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            input_ids_next_feat = next_depth_data.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            
            # 使用批量处理函数应用深度数据处理
            # 支持的submode: "norm", "inverse", "inverse_norm", "norm_inverse"
            input_ids_prev_feat, input_ids_next_feat = batch_depth_processing(
                input_ids_prev_feat,
                input_ids_next_feat,
                self.depth_norm_layer_c,
                self.depth_norm_layer_d,
                submode=submode,
                eps=1e-6
            )
            # 扩充next 的数据，区分当前和现在
            input_ids_next_feat = torch.cat([input_ids_prev_feat[:,-2:,...],input_ids_next_feat],1)
            # 确保变量被正确定义
            if input_ids_prev_feat is None:
                raise ValueError(
                    "input_ids_prev_feat is None, check prev_depth_img_input"
                )
            if input_ids_next_feat is None:
                raise ValueError(
                    "input_ids_next_feat is None, check next_depth_img_input"
                )

            token_mode = "mock"
            c_latent_size = 0
            start_time = time.time()
            if token_mode == "mock":
                # Mock input_ids_prev2s和labels_prev2s - 前三帧使用真实数据，后面mock
                input_ids_prev2s, labels_prev2s, c_latent_size = self._mock_tokenize_with_partial_real(
                    input_ids_feat=input_ids_prev_feat,
                    uni_prompting=uni_prompting,
                    real_frames=3,
                )
            else:
                input_ids_prev2s, labels_prev2s = self.vq_model.tokenize(
                    pixel_values=input_ids_prev_feat,
                )
            
            return input_ids_prev2s, labels_prev2s
        
        # TODO: 实现其他模式的处理
        raise ValueError(f"Unsupported mode: {mode}")

    def set_trainable(
        self,
        trainable=True,
        modules=None,
        stage=None,
        verbose=True,
    ):
        """
        设置 depth_encoder 各模块的训练状态

        Args:
            trainable (bool): 当modules=None且stage=None时应用到所有模块
            modules (str | list | None): 指定要训练的模块
                - str: 单个模块名称
                - list: 模块名称列表，列表中的模块设为 trainable，其他设为 ~trainable
                - None: 所有模块设置为 trainable 参数的值

            stage (str | int | None): 训练阶段，自动设置模块训练状态
                - 1, 'stage1', '1': 训练所有模块（包括magvit_dual_encoder）
                - 2, 'stage2', '2': 训练所有模块（包括magvit_dual_encoder）
                - 3, 'stage3', '3': 训练所有模块（包括magvit_dual_encoder）
                - 4, 'stage4', '4': 训练除magvit_dual_encoder外的所有模块
                - None: 使用modules和trainable参数的配置

            verbose (bool): 是否打印统计信息

        Returns:
            dict: {'total_params', 'trainable_params', 'frozen_params', 'trainable_ratio'}

        Examples:
            # 训练阶段1-3：训练所有模块
            encoder.set_trainable(stage=1)
            encoder.set_trainable(stage='stage2')

            # 训练阶段4：冻结magvit_dual_encoder
            encoder.set_trainable(stage=4)
            encoder.set_trainable(stage='stage4')

            # 手动控制模块
            encoder.set_trainable(modules=['depth_norm_layer_c', 'depth_norm_layer_d'])

            # 所有模块可训练
            encoder.set_trainable(True)
        """
        if not trainable :
            return 
        # 处理stage参数，自动配置训练状态和use_dual_vit
        if stage is not None:
            # 标准化stage参数
            stage_str = str(stage).lower().replace('stage', '').strip()

            if stage_str in ['1', '2', '3']:
                # 阶段1-3：训练所有模块（包括magvit_dual_encoder）
                self.use_dual_vit = True
                modules = None
                trainable = True
                if verbose:
                    print(f"[训练阶段{stage_str}] 训练所有模块，包括magvit_dual_encoder")
            elif stage_str == '4':
                # 阶段4：训练除magvit_dual_encoder外的所有模块
                self.use_dual_vit = True
                modules = None
                # 获取除magvit_dual_encoder外的所有模块
                available_modules = ['depth_norm_layer_c', 'depth_norm_layer_d']
                if self.use_dual_vit:
                    available_modules.extend(['context_mlp', 'dynamic_mlp', 'depth_cross_attention_module','residual_net'])
                else:
                    available_modules.extend(['depth_feature_extractor', 'depth_cross_attention'])
                modules = available_modules
                trainable = True
                if verbose:
                    print(f"[训练阶段4] 训练除magvit_dual_encoder外的所有模块: {modules}")
            else:
                raise ValueError(f"未知的训练阶段: {stage}. 可用: 1, 2, 3, 4, stage1, stage2, stage3, stage4")

        # 根据use_dual_vit条件确定可用模块
        module_map = {
            'depth_norm_layer_c': self.depth_norm_layer_c,
            'depth_norm_layer_d': self.depth_norm_layer_d,
            # 'depth_loader': self.depth_loader,
            'magvit_dual_encoder': self.magvit_dual_encoder,
        }

        if not getattr(self, 'use_dual_vit', True):
            module_map.update({
                'depth_feature_extractor': self.depth_feature_extractor,
                'depth_cross_attention': self.depth_cross_attention,
            })
        else:
            module_map.update({
                'context_mlp': self.context_mlp,
                'dynamic_mlp': self.dynamic_mlp,
                'depth_cross_attention_module': self.depth_cross_attention_module,
                "residual_net":self.residual_net,
            })

        # 解析 modules 参数
        if modules is None:
            # 全部设置为 trainable
            modules_set = set(module_map.keys()) if trainable else set()
        elif isinstance(modules, str):
            # 单个模块
            if modules not in module_map:
                raise ValueError(f"未知的模块名称: {modules}。可用: {list(module_map.keys())}")
            modules_set = {modules} if trainable else set(module_map.keys()) - {modules}
        elif isinstance(modules, (list, tuple)):
            # 模块列表
            for m in modules:
                if m not in module_map:
                    raise ValueError(f"未知的模块名称: {m}。可用: {list(module_map.keys())}")
            modules_set = set(modules) if trainable else set(module_map.keys()) - set(modules)
        else:
            raise TypeError(f"modules参数类型错误: {type(modules)}。应为 str/list/tuple 或 None")

        # 应用训练状态
        total = trainable_count = frozen_count = 0
        for name, module in module_map.items():
            should_train = name in modules_set
            for param in module.parameters():
                total += param.numel()
                param.requires_grad = should_train
                if should_train:
                    trainable_count += param.numel()
                else:
                    frozen_count += param.numel()

        # 打印统计
        if verbose:
            ratio = (trainable_count / total * 100) if total > 0 else 0
            print(f"{'='*60}")
            print(f"Depth Encoder 模块训练状态")
            print(f"{'='*60}")
            for name, module in module_map.items():
                is_trainable = name in modules_set
                status = "✅ 训练" if is_trainable else "🔒 冻结"
                count = sum(p.numel() for p in module.parameters())
                print(f"{name:<30} {count/1e6:<12.3f}M {status}")
            print(f"{'-'*60}")
            print(f"总计: {total/1e6:.3f}M | 可训练: {trainable_count/1e6:.3f}M | 比例: {ratio:.1f}%")
            print(f"{'='*60}\n")

        return {
            'total_params': total,
            'trainable_params': trainable_count,
            'frozen_params': frozen_count,
            'trainable_ratio': (trainable_count / total * 100) if total > 0 else 0
        }
    
    def get_trainable_info(self, verbose=True):
        """
        获取当前训练状态统计
        
        Returns:
            dict: {'total_params', 'trainable_params', 'frozen_params', 'trainable_ratio'}
        """
        total = trainable_count = frozen_count = 0
        
        for param in self.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable_count += param.numel()
            else:
                frozen_count += param.numel()
        
        if verbose:
            ratio = (trainable_count / total * 100) if total > 0 else 0
            print(f"{'='*50}")
            print(f"Depth Encoder 状态:")
            print(f"总参数: {total/1e6:.2f}M | 可训练: {trainable_count/1e6:.2f}M | 冻结: {frozen_count/1e6:.2f}M ({ratio:.1f}%)")
            print(f"{'='*50}")
        
        return {
            'total_params': total / 1e6,
            'trainable_params': trainable_count / 1e6,
            'frozen_params': frozen_count / 1e6,
            'trainable_ratio': (trainable_count / total * 100) if total > 0 else 0
        }

    def _extract_depth_field(self, depth_input, position="unknown", field="depth"):
        """
        从输入中按需提取指定字段的数据
        
        Args:
            depth_input: 输入数据，可能是字典、对象、张量
            position: 位置标识，用于日志输出（"prev" 或 "next"）
            field: 指定要提取的字段名称，如果为None则使用第一个可用的张量字段
        
        Returns:
            torch.Tensor: 提取的数据张量
        """
        if isinstance(depth_input, dict):
            # 字典格式
            if field is not None:
                # 指定了特定字段，直接提取
                if field in depth_input:
                    # print(f"  {position}: 从字典中提取指定的 '{field}' 字段")
                    return depth_input[field]
                else:
                    raise ValueError(f"{position}: 字典中没有找到指定的字段 '{field}'，可用字段: {list(depth_input.keys())}")
            else:
                # 未指定字段，使用第一个可用的张量字段
                available_fields = [k for k, v in depth_input.items()
                                  if isinstance(v, torch.Tensor) and v is not None]
                if available_fields:
                    field_used = available_fields[0]
                    print(f"  {position}: 未指定字段，使用第一个可用字段 '{field_used}'")
                    return depth_input[field_used]
                else:
                    raise ValueError(f"{position}: 字典中没有可用的张量字段，可用字段: {list(depth_input.keys())}")

        elif hasattr(depth_input, field) if field is not None else hasattr(depth_input, 'depth'):
            # 对象格式，有指定字段属性或默认depth属性
            attr_name = field if field is not None else 'depth'
            print(f"  {position}: 从对象的 '{attr_name}' 属性提取数据")
            return getattr(depth_input, attr_name)

        elif isinstance(depth_input, torch.Tensor):
            # 直接是张量格式
            print(f"  {position}: 直接使用张量数据")
            return depth_input

        else:
            attr_name = field if field is not None else 'depth'
            raise TypeError(f"{position}: 不支持的输入类型 {type(depth_input)}，期望dict、有'{attr_name}'属性的对象或torch.Tensor")

    def get_norm_parameters(self):
        """
        获取LayerNorm的可训练参数
        
        Returns:
            list: LayerNorm的参数列表
        """
        return list(self.depth_norm_layer.parameters())

    def _process_single_depth_batch(
        self,
        depth_context: torch.Tensor,
        depth_dynamic: torch.Tensor,
        image_context_feat: Union[torch.Tensor, None],
        image_dynamic_feat: Union[torch.Tensor, None],
        submode: str,
        use_magvit: bool = False,
    ) -> dict:
        """
        处理单个批次的深度特征：归一化 -> 特征提取 -> 融合
        
        Args:
            depth_context: 深度context特征 [B, T_c, H, W]
            depth_dynamic: 深度dynamic特征 [B, T_d, H, W]
            image_context_feat: 图像context特征（可选）
            image_dynamic_feat: 图像dynamic特征（可选）
            submode: 归一化模式
            use_magvit: 是否使用MagVIT风格的encoder提取特征
            
        Returns:
            dict: 包含context和dynamic的深度特征及attended特征
        """
        from models.depth_utils import batch_depth_processing
        result = {}
        
        # 归一化
        norm_layer_c = self.depth_norm_layer_c
        norm_layer_d = self.depth_norm_layer_d
        
        context_norm, dynamic_norm = batch_depth_processing(
            depth_context,
            depth_dynamic,
            norm_layer_c,
            norm_layer_d,
            submode=submode,
            eps=1e-6
        )
        # dynamic_norm, _ = batch_depth_processing(
        #     depth_dynamic.unsqueeze(2),
        #     depth_dynamic.unsqueeze(2),
        #     norm_layer_d,
        #     norm_layer_d,
        #     submode=submode,
        #     eps=1e-6
        # )
        use_magvit = True
        if use_magvit:
            # 使用双encoder模式：同时处理context和dynamic
            context_depth_feat, dynamic_depth_feat, context_attended, dynamic_attended = \
                self._extract_depth_features_dual_magvit(
                    context_norm, dynamic_norm, image_context_feat, image_dynamic_feat
                )
            result['context'] = context_depth_feat
            result['dynamic'] = dynamic_depth_feat
            if context_attended is not None:
                result['context_attended'] = context_attended
            if dynamic_attended is not None:
                result['dynamic_attended'] = dynamic_attended
        else:
            # 使用CNN模式：分别处理context和dynamic
            context_depth_feat, context_attended = self._extract_depth_features_with_attention(
                context_norm, image_context_feat
            )
            result['context'] = context_depth_feat
            if context_attended is not None:
                result['context_attended'] = context_attended
            
            dynamic_depth_feat, dynamic_attended = self._extract_depth_features_with_attention(
                dynamic_norm, image_dynamic_feat
            )
            result['dynamic'] = dynamic_depth_feat
            if dynamic_attended is not None:
                result['dynamic_attended'] = dynamic_attended
        
        return result
    
    def _ensure_4d_format(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """确保深度张量是[B*T, 1, H, W]格式"""
        if len(depth_tensor.shape) != 4:  # [B, T, H, W]
            B, T, H, W = depth_tensor.shape
            return depth_tensor.view(B * T, 1, H, W)
        return depth_tensor
    
    def _apply_simple(
        self,
        depth_latent: torch.Tensor,
    ) -> Union[torch.Tensor, None]:
        """
        对深度latent应用cross-attention
        
        Args:
            depth_latent: 深度latent [Bt, C, H, W]
            image_feat: 图像特征（张量或列表）
            
        Returns:
            attended特征或None
        """
        if depth_latent is None:
            return None
        
        if isinstance(depth_latent, list):
            depth_latent = depth_latent[-1] if depth_latent[-1] is not None else depth_latent[0]
        Bt, C, H, W = depth_latent.shape
        depth_seq = depth_latent.permute(0, 2, 3, 1).reshape(Bt, H * W, C)
        return depth_seq
    def _apply_cross_attention(
        self,
        depth_latent: torch.Tensor,
        image_feat: Union[torch.Tensor, list, None],
    ) -> Union[torch.Tensor, None]:
        """
        对深度latent应用cross-attention
        
        Args:
            depth_latent: 深度latent [Bt, C, H, W]
            image_feat: 图像特征（张量或列表）
            
        Returns:
            attended特征或None
        """
        if image_feat is None or depth_latent is None:
            return None
        
        if isinstance(depth_latent, list):
            depth_latent = depth_latent[-1] if depth_latent[-1] is not None else depth_latent[0]
        else:
            depth_latent = depth_latent
        Bt, C, H, W = depth_latent.shape
        depth_seq = depth_latent.permute(0, 2, 3, 1).reshape(Bt, H * W, C)
        
        # 处理图像特征（可能是多层特征列表）
        if isinstance(image_feat, list):
            img_feat = image_feat[-1] if image_feat[-1] is not None else image_feat[0]
        else:
            img_feat = image_feat
        
        Bi, C, H, W = img_feat.shape
        img_seq = img_feat.permute(0, 2, 3, 1).reshape(Bi, H * W, C)
        # 应用cross-attention
        assert depth_seq.shape == img_seq.shape
        # if Bt == Bi:
        #     return self.depth_cross_attention(depth_seq, img_seq)
        # img_seq = img_seq.repeat(Bt, C, 1)
        return depth_seq
    
    def _extract_depth_features_dual_magvit(
        self,
        depth_context: torch.Tensor,
        depth_dynamic: torch.Tensor,
        image_context_feat: Union[torch.Tensor, None],
        image_dynamic_feat: Union[torch.Tensor, None],
    ) -> tuple:
        """
        同时提取context和dynamic深度特征（使用双encoder模式）
        
        类似图像tokenizer：
        - context encoder: 无条件编码
        - dynamic encoder: 有条件编码，使用context的多层特征进行cross-attention
        
        Args:
            depth_context: context深度数据 [B, T_c, 1, H, W] 或 [B*T_c, 1, H, W]
            depth_dynamic: dynamic深度数据 [B, T_d, 1, H, W] 或 [B*T_d, 1, H, W]
            image_context_feat: 图像context特征（可选）
            image_dynamic_feat: 图像dynamic特征（可选）
            
        Returns:
            tuple: (context_depth_feat, dynamic_depth_feat, context_attended, dynamic_attended)
                  其中 context_attended 和 dynamic_attended 的形状为 [B, T, ...]
        """
        # 记录原始的 batch_size 和 time_steps，用于后续还原形状
        import torch
        shape_len = len(depth_context.shape)
        if shape_len == 6:
            B_6 = depth_context.shape[0]
            # 获取所有秒的context数据 (保持batch_size, num_sec, ...不变)
            batch_next_img_context = depth_context
            # 获取所有秒的dynamic数据 (batch_size, num_sec, 3, ...)
            batch_next_img_dynamic = torch.stack([
                depth_dynamic[:, sec*2:sec*2+3] 
                for sec in range(depth_context.shape[1])
            ], dim=1)
            merged_context = batch_next_img_context.reshape(-1, *batch_next_img_context.shape[2:])
            # 合并后的dynamic数据 (batch*time, 3, ...) 
            merged_dynamic = batch_next_img_dynamic.reshape(-1, *batch_next_img_dynamic.shape[2:])
            depth_context = merged_context
            depth_dynamic = merged_dynamic
        # if len(depth_context.shape) == 5:  # [B, T, 1, H, W]
        B_context, T_context, c_context, H_context, W_context = depth_context.shape
        B_dynamic, T_dynamic, c_dynamic, H_dynamic, W_dynamic = depth_dynamic.shape
        original_B = B_context
        original_T_context = T_context
        original_T_dynamic = T_dynamic
        # else:
        #     # [B*T, 1, H, W] 格式，无法准确还原，直接使用
        #     original_B = None
        #     original_T_context = None
        #     original_T_dynamic = None
        
        # # 处理形状，确保是[B*T, 1, H, W]格式
        # depth_context = self._ensure_4d_format(depth_context)
        # depth_dynamic = self._ensure_4d_format(depth_dynamic)
        
        # 使用双encoder同时提取context和dynamic特征
        # 使用新的 tokenize_depth 方法
        result = self.magvit_dual_encoder.tokenize_depth(depth_context, depth_dynamic, return_encoder_features=True)
        
        # 处理返回值：可能是2个或3个值
        if len(result) == 3:
            indices, labels, encoder_features = result
        else:
            indices, labels = result
            encoder_features = None
        
        # 如果没有返回encoder_features，需要从context_vq_model和dynamic_vq_model重新获取
        # if encoder_features is None:
        #     context_latent, context_cond_features = self.magvit_dual_encoder.context_vq_model.encoder(
        #         depth_context, return_features=True
        #     )
        #     dynamic_latent = self.magvit_dual_encoder.dynamic_vq_model.encoder(
        #         depth_dynamic, context_cond_features,
        #         cond=self.magvit_dual_encoder.cond_enable, attn_mask=None, time_ids=None
        #     )
        #     context_latents = [context_latent]  # 包装成列表以保持一致性
        # else:
            # 从 encoder_features 中提取 latent
        context_latents = encoder_features['context']  # List of features
        dynamic_latent = encoder_features['dynamic']  # Dynamic latent
            
            # # context_latents 是一个列表，取最后一个作为 context_latent
            # context_latent = context_latents[-1] if context_latents else None
        
        # 应用cross-attention
        if context_latents is not None:
            context_attended = self._apply_simple(context_latents)  
        else:
            context_attended = None
        if dynamic_latent is not None:
            dynamic_attended = self._apply_simple(dynamic_latent.reshape(-1, dynamic_latent.shape[-3]*4, dynamic_latent.shape[-2]//2, dynamic_latent.shape[-1]//2) )                                                    
        else:
            dynamic_attended = None
        # 使用 MLP 投影到 2048 维
        if context_attended is not None:
            context_attended = self.context_mlp(context_attended)
            # context_attended = self.context_norm(context_attended)
        if dynamic_attended is not None:
            dynamic_attended = self.dynamic_mlp(dynamic_attended)
            # dynamic_attended = self.dynamic_norm(dynamic_attended)
        
        # 还原形状：从 [B*T, ...] -> [B, T, ...]
        # if original_B is not None and original_T_context is not None and original_T_dynamic is not None:
        # context_attended: [B*T_c, ...] -> [B, T_c, ...]
        if context_attended is not None:
            context_attended = context_attended.view(original_B, original_T_context, *context_attended.shape[1:])
            if shape_len==6:
                context_attended = context_attended.reshape(B_6, -1, *context_attended.shape[1:])[:,0]
        
        # dynamic_attended: [B*T_d, ...] -> [B, T_d, ...]
        if dynamic_attended is not None:
            dynamic_attended = dynamic_attended.view(original_B, original_T_dynamic, *dynamic_attended.shape[1:])
            if shape_len==6:
                dynamic_attended = dynamic_attended.reshape(B_6,-1,*dynamic_attended.shape[1:])
                dynamic_attended_context_before_05 = dynamic_attended[:,0,0:1].reshape(B_6, -1, *dynamic_attended.shape[3:])
                dynamic_attended_4s = dynamic_attended[:,:,1:].reshape(B_6, -1, *dynamic_attended.shape[3:])
                dynamic_attended = torch.cat([dynamic_attended_context_before_05, dynamic_attended_4s], dim=1)
        
        return context_latents, dynamic_latent, context_attended, dynamic_attended
    
    
    def _extract_depth_features_with_attention(
        self,
        depth_feat: torch.Tensor,
        image_feat: Union[torch.Tensor, None],
    ) -> tuple:
        """
        使用CNN提取深度特征并与图像特征进行cross-attention
        
        Args:
            depth_feat: 深度特征 [B, T, H, W] 或 [B*T, 1, H, W]
            image_feat: 图像特征（可选）
            
        Returns:
            depth_features: 提取的深度特征
            attended_features: cross-attention后的特征（如果没有图像特征则为None）
        """
        # 使用CNN特征提取器
        if image_feat is not None:
            depth_features, attended_features = self.depth_feature_extractor(
                depth_feat,
                image_features=image_feat
            )
        else:
            depth_features, attended_features = self.depth_feature_extractor(depth_feat)
        return depth_features, attended_features
    
    def _extract_image_features(self, feats_dict):
        """
        从图像编码器特征字典中提取context和dynamic特征
        
        Args:
            feats_dict: 图像编码器特征字典 {'context': [...], 'dynamic': tensor}
        
        Returns:
            context_feat: 处理后的context特征 [B, H*W, C] 或 None
            dynamic_feat: 处理后的dynamic特征 [B, H*W, C] 或 None
        """
        if feats_dict is None:
            return None, None
            
        context_feats = feats_dict.get('context')
        dynamic_feats = feats_dict.get('dynamic')
        
        # # 提取context特征（倒数第二层，通常是16x16的特征）
        # context_feat = None
        # if context_feats is not None and len(context_feats) >= 2:
        #     feat = context_feats[-2]
        #     if len(feat.shape) == 4:
        #         B, C, H, W = feat.shape
        #         context_feat = feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # # 提取dynamic特征（patchify之前的原始特征）
        # dynamic_feat = None
        # if dynamic_feats is not None:
        #     feat = dynamic_feats
        #     if len(feat.shape) == 4:
        #         B, C, H, W = feat.shape
        #         dynamic_feat = feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        return context_feats, dynamic_feats
    
    def _process_time_step_depth(
        self,
        depth_context: torch.Tensor,
        depth_dynamic: torch.Tensor,
        image_feats_dict: Union[dict, None],
        submode: str,
        use_magvit: bool = False,
    ) -> dict:
        """
        处理单个时间步的深度数据
        
        Args:
            depth_context: 深度context数据 [B, T_c, H, W]
            depth_dynamic: 深度dynamic数据 [B, T_d, H, W]
            image_feats_dict: 图像特征字典 {'context': [...], 'dynamic': tensor}
            submode: 归一化模式
            use_magvit: 是否使用MagVIT风格的encoder提取特征
        
        Returns:
            dict: 包含context和dynamic的深度特征
        """
        image_context, image_dynamic = self._extract_image_features(image_feats_dict)
        return self._process_single_depth_batch(
            depth_context, depth_dynamic, image_context, image_dynamic, submode, use_magvit
        )
    
    def prepare_depth_features(
        self,
        prev_depth_context_1s: Union[torch.FloatTensor, torch.LongTensor],
        prev_depth_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor],
        prev_depth_context_2s: Union[torch.FloatTensor, torch.LongTensor],
        prev_depth_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor],
        next_depth_context: Union[torch.FloatTensor, torch.LongTensor],
        next_depth_dynamic: Union[torch.FloatTensor, torch.LongTensor],
        image_encoder_features=None,
        submode="norm_transform",
        use_magvit: bool = False,
    ):
        """
        准备深度特征，支持两种模式（内存优化版）
        
        1. CNN模式（use_magvit=False）: 使用CNN提取特征
        2. MagVIT双encoder模式（use_magvit=True）: 使用类似图像tokenizer的双encoder架构
           - context encoder: 无条件编码器
           - dynamic encoder: 条件编码器，使用context特征作为条件
        
        Args:
            prev_depth_context_1s: 1秒前的context深度数据 [B, T_context, H, W]
            prev_depth_dynamic_1s: 1秒前的dynamic深度数据 [B, T_dynamic, H, W]
            prev_depth_context_2s: 2秒前的context深度数据 [B, T_context, H, W]
            prev_depth_dynamic_2s: 2秒前的dynamic深度数据 [B, T_dynamic, H, W]
            next_depth_context: 未来的context深度数据 [B, T_context, H, W]
            next_depth_dynamic: 未来的dynamic深度数据 [B, T_dynamic, H, W]
            image_encoder_features: 图像编码器特征字典 {'1s': ..., '2s': ..., 'next': ...}
            submode: 深度预处理模式
            use_magvit: 是否使用MagVIT双encoder模式（类似图像tokenizer）
        
        Returns:
            dict: 包含提取的深度特征
                - 'prev_depth_context_features': prev深度context特征
                - 'prev_depth_dynamic_features': prev深度dynamic特征
                - 'next_depth_context_features': next深度context特征
                - 'next_depth_dynamic_features': next深度dynamic特征
                - 以及attended特征等（如果有图像特征）
        """
        device = self.device
        
        # 将数据移到设备
        prev_depth_context_1s = prev_depth_context_1s.to(device, non_blocking=True)
        prev_depth_dynamic_1s = prev_depth_dynamic_1s.to(device, non_blocking=True)
        prev_depth_context_2s = prev_depth_context_2s.to(device, non_blocking=True)
        prev_depth_dynamic_2s = prev_depth_dynamic_2s.to(device, non_blocking=True)
        next_depth_context = next_depth_context.to(device, non_blocking=True)
        next_depth_dynamic = next_depth_dynamic.to(device, non_blocking=True)
        
        # 分别处理每个时间步，使用统一的处理方法
        # 处理prev 1s数据
        prev_1s_image = image_encoder_features.get('1s') if image_encoder_features else None
        prev_1s_result = self._process_time_step_depth(
            prev_depth_context_1s, prev_depth_dynamic_1s, prev_1s_image, submode, use_magvit
        )
        
        # 处理prev 2s数据
        prev_2s_image = image_encoder_features.get('2s') if image_encoder_features else None
        prev_2s_result = self._process_time_step_depth(
            prev_depth_context_2s, prev_depth_dynamic_2s, prev_2s_image, submode, use_magvit
        )
        
        # 处理next数据
        next_image = image_encoder_features.get('next') if image_encoder_features else None
        next_result = self._process_time_step_depth(
            next_depth_context, next_depth_dynamic, next_image, submode, use_magvit
        )
        
        # 合并1s和2s结果
        result = {
            'prev_depth_context_features': torch.cat([
                prev_1s_result['context'][-1], prev_2s_result['context'][-1]
            ], dim=0) if prev_1s_result['context'] is not None and prev_2s_result['context'] is not None else None,
            'prev_depth_dynamic_features': torch.cat([
                prev_1s_result['dynamic'], prev_2s_result['dynamic']
            ], dim=0) if prev_1s_result['dynamic'] is not None and  prev_2s_result['dynamic'] is not None else None,
            'next_depth_context_features': next_result['context'],
            'next_depth_dynamic_features': next_result['dynamic'],
            'prev_1s_context_features':prev_1s_result['context'],
            'prev_1s_dynamic_features':prev_1s_result['dynamic'],
            'prev_2s_context_features':prev_2s_result['context'],
            'prev_2s_dynamic_features':prev_2s_result['dynamic'],
        }
        
        # 添加attended特征
        if 'context_attended' in prev_1s_result:
            result['prev_1s_context_attended'] = prev_1s_result['context_attended']
        if 'dynamic_attended' in prev_1s_result:
            result['prev_1s_dynamic_attended'] = prev_1s_result['dynamic_attended']
        if 'context_attended' in prev_2s_result:
            result['prev_2s_context_attended'] = prev_2s_result['context_attended']
        if 'dynamic_attended' in prev_2s_result:
            result['prev_2s_dynamic_attended'] = prev_2s_result['dynamic_attended']
        if 'context_attended' in next_result:
            result['next_context_attended'] = next_result['context_attended']
        if 'dynamic_attended' in next_result:
            result['next_dynamic_attended'] = next_result['dynamic_attended']
        result["index"]=-1
        return result
 
    def set_depth_cross_attention_training_stage(self, enable_context: bool, enable_dynamic: bool, verbose: bool = False):
        """
        设置 depth 相关模块的启用状态
        同时设置 depth_cross_attention_module 和 magvit_dual_encoder 的训练状态
        
        Args:
            enable_context: 是否启用 context 模块（soi/eoi区间）
            enable_dynamic: 是否启用 dynamic 模块（sod/eod区间）
            verbose: 是否打印详细信息
        
        启用状态说明：
        - enable_context=True, enable_dynamic=False: 只执行 context_attended
        - enable_context=False, enable_dynamic=True: 只执行 dynamic_attended
        - enable_context=True, enable_dynamic=True: 同时执行两者
        - enable_context=False, enable_dynamic=False: 都不执行（不推荐）
        """
        stage_desc = "both" if enable_context and enable_dynamic else \
                     "context" if enable_context and not enable_dynamic else \
                     "dynamic" if not enable_context and enable_dynamic else "none"
        # if hasattr(self, 'depth_encoder') and hasattr(self.depth_encoder, 'depth_cross_attention_module'):
        #     self.depth_encoder.depth_cross_attention_module.set_context_dynamic_enabled(enable_context, enable_dynamic)
        # 设置 depth_cross_attention_module 的状态
        if hasattr(self,'depth_cross_attention_module'):
            self.depth_cross_attention_module.set_context_dynamic_enabled(enable_context, enable_dynamic)
            if verbose:
                print(f"✅ DepthCrossAttention 启用状态: context={enable_context}, dynamic={enable_dynamic} ({stage_desc})")
        else:
            if verbose:
                print(f"⚠️  未找到 depth_cross_attention_module，跳过训练阶段设置")
        
        # 设置 magvit_dual_encoder (depth_encoder中的) 的状态
        if hasattr(self, 'magvit_dual_encoder'):
            self.magvit_dual_encoder.set_context_dynamic_enabled(enable_context, enable_dynamic)
            if verbose:
                print(f"✅ MagvitDualEncoder 启用状态: context={enable_context}, dynamic={enable_dynamic} ({stage_desc})")
        else:
            if verbose:
                print(f"⚠️  未找到 depth_encoder.magvit_dual_encoder，跳过训练阶段设置")
    
        self.enable_context = enable_context
        self.enable_dynamic = enable_dynamic    