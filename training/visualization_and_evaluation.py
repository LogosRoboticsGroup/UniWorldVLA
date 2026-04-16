# coding=utf-8
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
# limitations under the License.

"""
可视化和评测相关函数模块

从 fine-tune_navsim.py 中提取的所有可视化和评测相关函数（不包括evaluate函数）
"""

import moviepy
from moviepy.editor import ImageSequenceClip
import imageio
import os
import time
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Union
import copy
from PIL import Image
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

from data_utils.prepare_input_ids import tokenize_next_img_dynamic
from models.video_metric import Evaluator, FeatureStats
from navsim.pdsm_test_utils import PDSM_eval, pdsm_score_process
from navsim.visualization.camera import visualize_pred_gt_camera_traj

logger = logging.getLogger(__name__)

# ==================== 可视化函数 ====================

@torch.no_grad()
def visualize_predictions(model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step,
                        input_ids, #all task tokenized GT
                        logits, #all task predicted
                        image_ori, #t2d tokenized context and dynamic
                        token_ori, #t2d GT pixel
                        planning, #t2d GT command
                        desc_a, #d2t GT text
                        token,
                        real_len,
                        accelerator,
                        num_per_frame=30,
                        pred_frames=11
                            ):
    """
    可视化预测结果，包括原始图像、重建图像和预测图像的对比
    
    Args:
        model: 模型对象
        vq_model: VQ模型
        uni_prompting: 统一提示对象
        config: 配置对象
        global_step: 全局步数
        input_ids: 输入token (所有任务token化的GT)
        logits: 预测logits (所有任务预测的)
        image_ori: 原始图像t2d token化的context和dynamic
        token_ori: t2d GT pixel
        planning: t2d GT command
        desc_a: d2t GT text
        token: token列表
        real_len: 真实长度
        accelerator: 加速器对象
        num_per_frame: 每帧的token数量
        pred_frames: 预测帧数
    """
    logger.info("Visualizing training set logits...")
    model.eval()
    batch_ids = torch.randint(low=0, high=input_ids.shape[0], size=(1,)).to(input_ids.device)

    if len(real_len[:, 1].unique()) != 1 or len(real_len[:, 0].unique()) != 1:  # drop sampls len less than 11
        complete_idx = torch.where((real_len[:, 1] == 12) & (real_len[:, 0] == 12))[0].to(input_ids.device)

    else:
        complete_idx = None
    if complete_idx is not None:  #
        while (batch_ids not in complete_idx):
            batch_ids = torch.randint(low=0, high=input_ids.shape[0], size=(1,)).to(input_ids.device)


    if accelerator.is_main_process:
        with torch.no_grad():
            prev_token, next_token = token_ori
            #pixel value images
            images_prev = torch.clamp((image_ori[1][batch_ids] + 1.0) / 2.0, min=0.0, max=1.0)[0]  # (T(c+d), 3, H, W)
            images_prev *= 255.0
            # images_prev = images_prev.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            images_next = torch.clamp((image_ori[3][batch_ids] + 1.0) / 2.0, min=0.0, max=1.0)[0]  # (T(c+d), 3, H, W)
            images_next *= 255.0
            images = torch.cat((images_prev[1:], images_next[1:]), dim=0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


            #prev images  GT
            context_len_prev = len(prev_token['context'][0])//450
            #Tokenizer recon
            recons_images_prev, recons_lens_prev = vq_model.detokenize(indices=prev_token,
                                                                       batch_ids=batch_ids,
                                                                       offset_tokenzier=len(uni_prompting.text_tokenizer),
                                                                       sptids_dict=uni_prompting.sptids_dict)#(B, T,C,H,W) tokenizer reconstruct
            recons_images_prev = torch.clamp((recons_images_prev + 1.0) / 2.0, min=0.0, max=1.0)
            recons_images_prev = 255.0*recons_images_prev
            recons_images_prev = recons_images_prev[0]#.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)# remove first 2-context frames
            #next images    GT
            recons_images_next, recons_lens_next = vq_model.detokenize(indices=next_token,
                                                                       batch_ids=batch_ids,
                                                                       offset_tokenzier=len(uni_prompting.text_tokenizer),
                                                                       sptids_dict=uni_prompting.sptids_dict)#(B, T,C,H,W) tokenizer reconstruct
            recons_images_next = torch.clamp((recons_images_next + 1.0) / 2.0, min=0.0, max=1.0)
            recons_images_next = 255.0*recons_images_next
            recons_images_next = recons_images_next[0]#.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            recons_images = torch.cat((recons_images_prev[2:], recons_images_next[2:]), dim=0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            # nusc Predicted recon
            pred_logits_token = dict(context=next_token["context"], dynamic=logits.argmax(-1))
            d_start_sod = torch.arange(0, pred_frames)*num_per_frame
            d_end_sod = torch.arange(1, pred_frames+1)*num_per_frame-1
            predicted_images, _ = vq_model.detokenize(pred_logits_token,
                                                   batch_ids=batch_ids,
                                                   offset_tokenzier=len(uni_prompting.text_tokenizer),
                                                   sptids_dict=uni_prompting.sptids_dict,
                                                   c_start_sod=None,
                                                   c_end_sod=None,
                                                   d_start_sod=d_start_sod.to(input_ids.device),
                                                   d_end_sod=d_end_sod.to(input_ids.device),
                                                   )#(T-1,C,H,W)
            predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
            predicted_images *= 255.0
            predicted_images = predicted_images[0][2:].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            # pred_images = torch.cat((recons_images_prev[2:], predicted_images[2:]), dim=0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            t2d_v = video_concate(images, recons_images, predicted_images)
            t2d_v = np.stack(t2d_v, 0)
            output_root = os.path.join(config.experiment.output_dir, "visual_gif", "train")
            os.makedirs(output_root, exist_ok=True)
            imageio.mimsave(os.path.join(output_root, f"{global_step}_{token[batch_ids]}.gif"), list(t2d_v), fps=10, loop=0)  # fps
            t2d_v = np.transpose(t2d_v, (0, 3, 1, 2))
            # video display
            wandb.log({"Original v.s. Reconstructed v.s. Predicted": wandb.Video(t2d_v, fps=10, format="webm", caption=desc_a[batch_ids])}, step=global_step)
        logger.info("Visualizing finished...")

def img_token2pixel(image_tokens_ori, uni_prompting, vq_model, gen_image_token_ids=None):
    """
    将图像token IDs解码为像素值
    
    Args:
        image_tokens_ori: 原始图像token数据
        uni_prompting: 统一提示对象
        vq_model: VQ模型
        gen_image_token_ids: 生成的图像token IDs (可选)
    
    Returns:
        torch.Tensor: 解码后的像素值 [T, C, H, W]
    """
    if gen_image_token_ids is not None:
        img_token = dict(context=image_tokens_ori["context"], dynamic=gen_image_token_ids)
    else:
        img_token = image_tokens_ori
    img_pixel, _ = vq_model.detokenize(img_token,
                                      offset_tokenzier=len(uni_prompting.text_tokenizer),
                                      sptids_dict=uni_prompting.sptids_dict,
                                      )  # (T-1,C,H,W)
    img_pixel = torch.clamp((img_pixel + 1.0) / 2.0, min=0.0, max=1.0)
    return img_pixel

def depth_token2pixel(depth_tokens_ori, uni_prompting, vq_model, gen_depth_token_ids=None):
    """
    将深度token IDs解码为深度像素值
    
    Args:
        depth_tokens_ori: 原始深度token数据
        uni_prompting: 统一提示对象
        vq_model: VQ模型
        gen_depth_token_ids: 生成的深度token IDs (可选)
    
    Returns:
        torch.Tensor: 解码后的深度值 [T, H, W]
    """
    if gen_depth_token_ids is not None:
        depth_token = dict(context=depth_tokens_ori["context"], dynamic=gen_depth_token_ids)
    else:
        depth_token = depth_tokens_ori
    
    # 使用VQ模型解码深度tokens
    depth_pixel, _ = vq_model.detokenize(depth_token,
                                      offset_tokenzier=len(uni_prompting.text_tokenizer),
                                      sptids_dict=uni_prompting.sptids_dict,
                                      )  # (T-1, C, H, W)
    
    # 对于深度数据，我们只取第一个通道（因为是灰度深度值）
    if depth_pixel.shape[1] > 1:
        depth_pixel = depth_pixel[:, 0:1, :, :]  # 取第一个通道
    
    # 归一化深度值到 [0, 1] 范围
    depth_pixel = torch.clamp(depth_pixel, min=-1.0, max=1.0)
    depth_pixel = (depth_pixel + 1.0) / 2.0
    
    return depth_pixel

def video_concate(o_images, r_images, p_images, context_length=None):
    """
    拼接原始、重建和预测图像为视频序列
    
    Args:
        o_images: 原始图像列表
        r_images: 重建图像列表
        p_images: 预测图像列表
        context_length: 上下文长度 (可选)
    
    Returns:
        list: 拼接后的图像序列
    """
    len_o = len(o_images)
    len_r = len(r_images)
    len_p = len(p_images)

    max_len = max(len_o, len_r, len_p)
    t2d_v = []
    for i in range(max_len):
        i_o = o_images[i % len_o]
        i_r = r_images[i % len_r]
        i_p = p_images[i % len_p]
        t2d_v.append(np.concatenate((i_o, i_r, i_p), axis=-2))
    return t2d_v

def save_as_webm(t2d_v, output_path, fps=10):
    """
    将视频序列保存为webm格式
    
    Args:
        t2d_v: 视频序列张量 [T, C, H, W]
        output_path: 输出路径
        fps: 帧率
    """
    T, C, H, W = t2d_v.shape
    assert C in (1, 3), "The channel dimension (C) must be 1 (grayscale) or 3 (RGB)."
    frames = [t2d_v[i].transpose(1, 2, 0) for i in range(T)]
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libvpx", verbose=False)

def video_display_and_visualization(accelerator, num_visual, pixel_values, recons_images, predicted_images, 
                                  context_length, output_root, global_step, token, batch_size, i):
    """
    执行视频显示和可视化功能
    
    Args:
        accelerator: 加速器对象
        num_visual: 可视化计数器
        pixel_values: 像素值
        recons_images: 重建图像
        predicted_images: 预测图像
        context_length: 上下文长度
        output_root: 输出根目录
        global_step: 全局步数
        token: token列表
        batch_size: 批次大小
        i: 当前迭代次数
    
    Returns:
        int: 更新后的 num_visual
    """
    # if accelerator.is_main_process and num_visual <= 1000000080:
    magic_number = np.random.rand()
    # if magic_number < 0.45:
    b_sample = np.random.randint(batch_size)
    # for cur in range(batch_size):
    for cur in range(1):
        b_sample=cur
        num_visual += 1
        o_images = (255.0*pixel_values[b_sample]).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        r_images = (255.0*recons_images[b_sample]).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        p_images = (255.0*predicted_images[b_sample]).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        t2d_v = video_concate(o_images, r_images, p_images, context_length)
        t2d_v = np.stack(t2d_v, 0)
        val_root = os.path.join(output_root, "visual_gif", "val")
        os.makedirs(val_root, exist_ok=True)
        imageio.mimsave(os.path.join(val_root,
                                        f"step_{global_step}_{token[b_sample]}.gif"), list(t2d_v), fps=10, loop=0)
                                        
        # 创建单独的文件夹保存每一帧的o_images和p_images
        frame_save_folder = os.path.join(val_root, f"step_{global_step}_{token[b_sample]}")
        os.makedirs(frame_save_folder, exist_ok=True)
        
        # 保存o_images的每一帧
        for frame_idx, o_img in enumerate(o_images):
            o_img_pil = Image.fromarray(o_img)
            o_img_pil.save(os.path.join(frame_save_folder, f"o_frame_{frame_idx:03d}.png"))
        
        # 保存p_images的每一帧
        for frame_idx, p_img in enumerate(p_images):
            p_img_pil = Image.fromarray(p_img)
            p_img_pil.save(os.path.join(frame_save_folder, f"p_frame_{frame_idx:03d}.png"))

        t2d_v = np.transpose(t2d_v, (0, 3, 1, 2))
        # final_step = global_step + i
        # if num_visual <= 10:
        #     wandb.log({"VAL-Original v.s. Reconstructed v.s. generated": wandb.Video(t2d_v, fps=10, format="webm",
        #                 caption=token[b_sample])}, step=final_step)
    
    return num_visual

def depth_visualization_draw(depth_data, original_image, output_dir, base_name='', device=None):
    """
    专门为深度数据设计的可视化函数，参考DA3的draw_one函数
    
    Args:
        depth_data: 深度数据张量 [H, W] 或 [C, H, W]
        original_image: 原始图像张量 [H, W, 3] 或 [3, H, W]
        output_dir: 输出目录
        base_name: 文件名前缀
        device: 设备
    
    Returns:
        tuple: (depth_color_array, combined_image_array)
    """
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import torch
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理深度数据
    if isinstance(depth_data, torch.Tensor):
        depth = depth_data.cpu().numpy()
    else:
        depth = depth_data
    
    # 如果深度数据有多个通道，取第一个通道
    if len(depth.shape) == 3:
        depth = depth[0]
    
    # 使用1%-99%百分位进行鲁棒归一化，再着色
    vmin, vmax = np.percentile(depth, [1, 99])
    depth_clip = np.clip(depth, vmin, vmax)
    depth_norm = (depth_clip - vmin) / (vmax - vmin + 1e-8)  # 归一到 [0, 1]
    depth_color = (plt.get_cmap('magma')(depth_norm)[..., :3] * 255).astype(np.uint8)  # RGB
    
    # 处理原始图像
    if isinstance(original_image, torch.Tensor):
        if original_image.dim() == 4:  # [1, 3, H, W]
            original_image = original_image.squeeze(0)
        
        if original_image.shape[0] == 3:  # [3, H, W]
            image_input = original_image.permute(1, 2, 0).cpu().numpy()
        else:  # [H, W, 3]
            image_input = original_image.cpu().numpy()
    else:
        image_input = original_image
    
    # 如果图像数据是归一化的，需要反归一化
    if image_input.max() <= 1.0:
        # 使用ImageNet的均值和标准差进行反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_imgs = image_input * std + mean
        processed_imgs = np.clip(processed_imgs, 0, 1)
        processed_imgs = (processed_imgs * 255).astype(np.uint8)
    else:
        processed_imgs = image_input.astype(np.uint8)
    
    # 保存单独的图像
    Image.fromarray(depth_color).save(os.path.join(output_dir, f"{base_name}_depth.png"))
    Image.fromarray(processed_imgs).save(os.path.join(output_dir, f"{base_name}_original.png"))
    
    # 创建组合图像：深度图 + 原始图
    # 确保两个图像高度一致
    h1, w1 = depth_color.shape[:2]
    h2, w2 = processed_imgs.shape[:2]
    
    if h1 != h2:
        # 调整高度以匹配
        min_h = min(h1, h2)
        depth_color = depth_color[:min_h]
        processed_imgs = processed_imgs[:min_h]
    
    all_data = np.concatenate([depth_color, processed_imgs], axis=1)
    Image.fromarray(all_data).save(os.path.join(output_dir, f"{base_name}_combined.png"))
    
    return depth_color, all_data

def depth_display_and_visualization(accelerator, num_visual, predicted_depths, recons_depths, depth_values,
                                  context_length, output_root, global_step, token, batch_size, i):
    """
    执行深度数据的显示和可视化功能
    
    Args:
        accelerator: 加速器对象
        num_visual: 可视化计数器
        predicted_depths: 预测深度数据 [B, T, H, W]
        recons_depths: 重建深度数据 [B, T, H, W]
        depth_values: 真实深度数据 [B, T, H, W]
        context_length: 上下文长度
        output_root: 输出根目录
        global_step: 全局步数
        token: token列表
        batch_size: 批次大小
        i: 当前迭代次数
    
    Returns:
        int: 更新后的 num_visual
    """
    if accelerator.is_main_process and num_visual <= 80:
        magic_number = np.random.rand()
        if magic_number < 0.45:
            b_sample = np.random.randint(batch_size)
            num_visual += 1
            
            # 处理深度数据
            pred_depth = predicted_depths[b_sample]  # [T, H, W]
            recon_depth = recons_depths[b_sample] if recons_depths is not None else None
            real_depth = depth_values[b_sample] if depth_values is not None else None
            
            # 创建输出目录
            depth_val_root = os.path.join(output_root, "depth_visual", "val")
            os.makedirs(depth_val_root, exist_ok=True)
            
            # 为每个时间步创建可视化
            for t in range(pred_depth.shape[0]):
                base_name = f"step_{global_step}_{token[b_sample]}_t{t}"
                
                # 预测深度可视化
                depth_visualization_draw(
                    pred_depth[t],
                    pred_depth[t],  # 使用深度数据本身作为"图像"
                    depth_val_root,
                    f"{base_name}_pred"
                )
                
                # 如果有重建深度，也进行可视化
                if recon_depth is not None:
                    depth_visualization_draw(
                        recon_depth[t],
                        recon_depth[t],
                        depth_val_root,
                        f"{base_name}_recon"
                    )
                
                # 如果有真实深度，也进行可视化
                if real_depth is not None:
                    depth_visualization_draw(
                        real_depth[t],
                        real_depth[t],
                        depth_val_root,
                        f"{base_name}_real"
                    )
            
            # 创建深度序列的gif（如果有wandb）
            if num_visual <= 10:
                try:
                    import wandb
                    
                    # 创建深度序列数据用于wandb
                    depth_sequence = []
                    for t in range(pred_depth.shape[0]):
                        # 将深度数据转换为3通道用于wandb
                        depth_frame = pred_depth[t]
                        if depth_frame.dim() == 2:
                            depth_frame = depth_frame.unsqueeze(0).repeat(3, 1, 1)  # [H, W] -> [3, H, W]
                        depth_sequence.append(depth_frame)
                    
                    depth_sequence = torch.stack(depth_sequence, 0)  # [T, 3, H, W]
                    final_step = global_step + i
                    
                    # 转换为numpy数组用于wandb
                    depth_sequence_np = depth_sequence.cpu().numpy()
                    
                    wandb.log({
                        "VAL-Depth-Prediction": wandb.Video(depth_sequence_np, fps=10, format="webm",
                        caption=token[b_sample])
                    }, step=final_step)
                    
                except ImportError:
                    print("wandb not available, skipping depth video logging")
                except Exception as e:
                    print(f"Error logging depth video to wandb: {e}")
    
    return num_visual

# ==================== 评测函数 ====================

def batch_forward(batch_size, input, forward, context_length=None, special_token=None, verbose=False):
    """
    批量前向传播处理
    
    Args:
        batch_size: 批次大小
        input: 输入数据
        forward: 前向传播函数
        context_length: 上下文长度 (可选)
        special_token: 特殊token (可选)
        verbose: 是否显示进度条
    
    Returns:
        torch.Tensor: 批量处理结果
    """
    from tqdm import trange
    if context_length is None and special_token is None:
        return torch.cat([forward(input[i: i + batch_size], ) for i in trange(0, input.shape[0], batch_size, disable=not verbose)], dim=0)
    else:
        return torch.cat(
            [forward(input[i: i + batch_size], context_length=context_length, special_token=special_token ) for i in trange(0, input.shape[0], batch_size, disable=not verbose)],
            dim=0)

def process_images(accelerator, images, evaluator, detector_kwargs, max_decode_batchsize=20):
    """
    处理图像用于评测
    
    Args:
        accelerator: 加速器对象
        images: 图像张量
        evaluator: 评估器对象
        detector_kwargs: 检测器参数
        max_decode_batchsize: 最大解码批次大小
    
    Returns:
        torch.Tensor: 处理后的特征
    """
    images = images.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, c, t, h, w]
    if max_decode_batchsize is not None and images.shape[0] > max_decode_batchsize:
        features = batch_forward(
            max_decode_batchsize,
            images * 255.,
            lambda x: accelerator.unwrap_model(evaluator).i3d_model(x, **detector_kwargs)
        )
    else:
        features = accelerator.unwrap_model(evaluator).i3d_model(images * 255., **detector_kwargs)
    gathered_features = accelerator.gather(features)
    return gathered_features

def video_metrics_process(config, mse_values, fvd, psnr_values, ssim_values, lpips_values):
    """
    处理视频评估指标，生成日志字典
    
    Args:
        config: 配置对象
        mse_values: MSE值列表
        fvd: FVD值
        psnr_values: PSNR值列表
        ssim_values: SSIM值列表
        lpips_values: LPIPS值列表
    
    Returns:
        dict: 包含视频指标的字典
    """
    eval_logs = {
        'eval/mse': torch.cat(mse_values, 0).mean().item(),
    }
    if config.experiment.eval.use_fvd:
        eval_logs.update({'eval/fvd': fvd})
    if config.experiment.eval.use_frame_metrics:
        eval_logs.update({
            'eval/psnr': torch.cat(psnr_values, 0).mean().item(),
            'eval/ssim': torch.cat(ssim_values, 0).mean().item(),
            'eval/lpips': torch.cat(lpips_values, 0).mean().item(),
        })
    return eval_logs

def depth_metrics_process(config, depth_mse_values, depth_abs_rel_values, depth_rmse_values):
    """
    处理深度评估指标，生成日志字典
    
    Args:
        config: 配置对象
        depth_mse_values: 深度MSE值列表
        depth_abs_rel_values: 深度绝对相对误差值列表
        depth_rmse_values: 深度RMSE值列表
    
    Returns:
        dict: 包含深度指标的字典
    """
    eval_logs = {}
    
    # 检查深度评估开关和数据是否可用
    if (getattr(config.experiment.eval, 'use_depth_evaluation', False) and
        getattr(config.experiment.eval, 'use_depth_metrics', False) and
        len(depth_mse_values) > 0):
        eval_logs.update({
            'eval/depth_mse': torch.cat(depth_mse_values, 0).mean().item(),
            'eval/depth_abs_rel': torch.cat(depth_abs_rel_values, 0).mean().item(),
            'eval/depth_rmse': torch.cat(depth_rmse_values, 0).mean().item(),
        })
    
    return eval_logs

def video_metrics_evaluation(config, accelerator, evaluator, uni_prompting, vq_model,
                           image_tokens_ori, next_img_dynamic, gen_image_token_ids,
                           batch_size, mse_values, psnr_values, ssim_values,
                           lpips_values, fvds, real_feats, gen_feats, time_0,next_img_context_per_sec,context_length):
    """
    执行视频指标评估，包括FVD、MSE、PSNR、SSIM、LPIPS等指标计算
    
    Args:
        config: 配置对象
        accelerator: 加速器对象
        evaluator: 评估器对象
        uni_prompting: 统一提示对象
        vq_model: VQ模型
        image_tokens_ori: 原始图像token
        next_img_dynamic: 动态图像
        gen_image_token_ids: 生成的图像token IDs
        batch_size: 批次大小
        mse_values: MSE值列表
        psnr_values: PSNR值列表
        ssim_values: SSIM值列表
        lpips_values: LPIPS值列表
        fvds: FVD值列表
        real_feats: 真实特征统计对象
        gen_feats: 生成特征统计对象
        time_0: 开始时间
    
    Returns:
        tuple: (predicted_images, recons_images, pixel_values, mse_values, psnr_values, ssim_values, lpips_values, fvds, real_feats, gen_feats)
    """
    # logger.info(f'infer_time:{infer_time}s')
    # predicted_images = img_token2pixel(image_tokens_ori[1], uni_prompting, vq_model, gen_image_token_ids)[:, 2:]
    # recons_images = img_token2pixel(image_tokens_ori[1], uni_prompting, vq_model)[:, 2:]
    # pixel_values = torch.clamp((next_img_dynamic + 1.0) / 2.0, min=0.0, max=1.0)[:, 1:]
    next_context_tokens, _, next_dynamic_tokens,_ = tokenize_next_img_dynamic(next_img_context_per_sec, 
                                                            next_img_dynamic, vq_model, uni_prompting, condition_len=context_length)
    predicted_images = img_token2pixel_customed(next_context_tokens, gen_image_token_ids, uni_prompting, vq_model, context_length)
    recons_images = img_token2pixel_customed(next_context_tokens, next_dynamic_tokens, uni_prompting, vq_model, context_length)
    # predicted_images = img_token2pixel(image_tokens_ori[1], uni_prompting, vq_model, gen_image_token_ids)[:, 2:]
    # recons_images = img_token2pixel(image_tokens_ori[1], uni_prompting, vq_model)[:, 2:]
    pixel_values = torch.clamp((next_img_dynamic + 1.0) / 2.0, min=0.0, max=1.0)[:, 1:] # [B, 10, 3, H, W]
    # pad 8 frames to 10 frames
    fvd = None
    padded_predicted_images = torch.cat([pixel_values[:, :1], predicted_images, pixel_values[:, -1:]], dim=1)
    padded_pixel_values = torch.cat([pixel_values[:, :1], pixel_values, pixel_values[:, -1:]], dim=1)
    if config.experiment.eval.use_fvd:
        with torch.autocast("cuda", dtype=torch.float32, enabled=False):
            detector_kwargs = dict(rescale=True, resize=True, return_features=True)
            real_feat = process_images(accelerator, padded_pixel_values, evaluator, detector_kwargs)
            gen_feat = process_images(accelerator, padded_predicted_images, evaluator, detector_kwargs)
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                real_feats.append_torch(real_feat)
                gen_feats.append_torch(gen_feat)
                fvd = accelerator.unwrap_model(evaluator).compute_fvd(real_feats, gen_feats)
                fvds.append(torch.tensor(fvd).repeat(batch_size))
                logger.info(f"current fvd estimate:{fvd}")
    
    # video metrics:mse,psnr,ssim,lpips
    if config.experiment.eval.use_frame_metrics:
        with torch.autocast("cuda", dtype=torch.float32, enabled=False):
            mse_value, psnr_value, ssim_value, lpips_value = accelerator.unwrap_model(evaluator)(pixel_values.clamp(0.0, 1.0), predicted_images)
            mse_values.append(accelerator.gather(mse_value.repeat(batch_size)))
            psnr_values.append(accelerator.gather(psnr_value.repeat(batch_size)))
            ssim_values.append(accelerator.gather(ssim_value.repeat(batch_size)))
            lpips_values.append(accelerator.gather(lpips_value.repeat(batch_size)))
    
    return predicted_images, recons_images, pixel_values, mse_values, psnr_values, ssim_values, lpips_values, fvds, real_feats, gen_feats,fvd

def depth_metrics_evaluation(config, accelerator, uni_prompting, vq_model, depth_tokens_ori, next_depth_dynamic, gen_depth_token_ids,
                           batch_size, depth_mse_values=None, depth_abs_rel_values=None, depth_rmse_values=None):
    """
    执行深度指标评估，包括深度相关的指标计算
    
    Args:
        config: 配置对象
        accelerator: 加速器对象
        uni_prompting: 统一提示对象
        vq_model: VQ模型
        depth_tokens_ori: 原始深度token
        next_depth_dynamic: 动态深度数据
        gen_depth_token_ids: 生成的深度token IDs
        batch_size: 批次大小
        depth_mse_values: 深度MSE值列表 (可选)
        depth_abs_rel_values: 深度绝对相对误差值列表 (可选)
        depth_rmse_values: 深度RMSE值列表 (可选)
    
    Returns:
        tuple: (predicted_depths, recons_depths, depth_values, depth_mse_values, depth_abs_rel_values, depth_rmse_values)
    """
    # 初始化列表
    if depth_mse_values is None:
        depth_mse_values = []
    if depth_abs_rel_values is None:
        depth_abs_rel_values = []
    if depth_rmse_values is None:
        depth_rmse_values = []
    
    # 解码生成的深度tokens
    predicted_depths = depth_token2pixel(depth_tokens_ori, uni_prompting, vq_model, gen_depth_token_ids)
    recons_depths = depth_token2pixel(depth_tokens_ori, uni_prompting, vq_model)
    
    # 跳过前两帧，与视频处理保持一致
    if predicted_depths.shape[1] > 2:
        predicted_depths = predicted_depths[:, 2:]
    if recons_depths.shape[1] > 2:
        recons_depths = recons_depths[:, 2:]
    
    # 处理真实深度数据
    if next_depth_dynamic is not None:
        depth_values = torch.clamp(next_depth_dynamic, min=0.0, max=1.0)[:, 1:]
    else:
        depth_values = None
    
    # 计算深度指标 - 检查深度评估开关
    if (getattr(config.experiment.eval, 'use_depth_evaluation', False) and
        getattr(config.experiment.eval, 'use_depth_metrics', False) and
        depth_values is not None):
        with torch.autocast("cuda", dtype=torch.float32, enabled=False):
            # 计算MSE
            depth_mse_value = torch.mean((predicted_depths - depth_values) ** 2, dim=[1, 2, 3])
            depth_mse_values.append(accelerator.gather(depth_mse_value.repeat(batch_size)))
            
            # 计算绝对相对误差 (Abs Rel)
            # 避免除零错误
            depth_abs_rel_value = torch.mean(torch.abs(predicted_depths - depth_values) / (depth_values + 1e-8), dim=[1, 2, 3])
            depth_abs_rel_values.append(accelerator.gather(depth_abs_rel_value.repeat(batch_size)))
            
            # 计算RMSE
            depth_rmse_value = torch.sqrt(torch.mean((predicted_depths - depth_values) ** 2, dim=[1, 2, 3]))
            depth_rmse_values.append(accelerator.gather(depth_rmse_value.repeat(batch_size)))
    
    return predicted_depths, recons_depths, depth_values, depth_mse_values, depth_abs_rel_values, depth_rmse_values

def action_metrics_evaluation(config, accelerator, gen_trj, token_encode, scoring_params, cache_path, score_rows, logger,data_out,data_loader,save_trj_vis=False):
    """
    执行动作指标评估，处理轨迹数据和PDSM评估
    
    Args:
        config: 配置对象
        accelerator: 加速器对象
        gen_trj: 生成的轨迹数据
        token_encode: 编码的token
        scoring_params: 评分参数
        cache_path: 缓存路径
        score_rows: 评分行列表
        logger: 日志记录器
    
    Returns:
        list: 更新后的 score_rows
    """
    if config.experiment.eval.use_trj_metrics:
        if save_trj_vis:
            # save_root = os.path.join("wandb", "validation_only")
            traj_save_dir = os.path.join(data_out, "traj")
            os.makedirs(traj_save_dir, exist_ok=True)

        samples_eval = []
        for trj_i in range(gen_trj.shape[0]):
            gathered_gen_trj = accelerator.gather(gen_trj[trj_i][None, ...].contiguous())
            gathered_token = accelerator.gather(token_encode[trj_i][None, ...].contiguous())
            assert gathered_gen_trj.shape[0] == gathered_token.shape[0], ( f"Shape mismatch: gathered_gen_trj has {gathered_gen_trj.shape[0]} samples, "
                                                                           f"but gathered_token has {gathered_token.shape[0]} samples." )
            for gather_idx in range(gathered_gen_trj.shape[0]):
                result_action = dict(cfg=scoring_params,
                                 cache_path=Path(cache_path),
                                 token=bytes(gathered_token[gather_idx].cpu().numpy().tolist()).decode("utf-8"),
                                 future_trajectory=gathered_gen_trj[gather_idx].detach().cpu().to(torch.float32))
                samples_eval.append(result_action)
                if save_trj_vis:
                    data_set = data_loader.dataset.dataset if hasattr(data_loader.dataset, "dataset") else data_loader.dataset
                    scene = data_set.scene_loader.get_scene_from_token(result_action["token"])
                    future_trajectory = scene.get_future_trajectory(num_trajectory_frames=8)
                    future_trajectory = future_trajectory.poses
                    visualize_pred_gt_camera_traj(
                        camera_=scene.frames[scene.scene_metadata.num_history_frames-1].cameras,
                        pred_point_lidar=result_action["future_trajectory"],
                        gt=future_trajectory,
                        save_path=traj_save_dir+f"/{result_action['token']}_traj_cam.png",
                    )
        
        if accelerator.is_main_process:
            score_rows.extend(PDSM_eval(config, samples_eval, logger))
    
    return score_rows

def img_token2pixel_customed(next_context_tokens, next_dynamic_tokens, uni_prompting, vq_model, condition_len=2):
    assert next_context_tokens.ndim == 3, 'incorrect next_context_tokens.dim()'
    sec_num = next_context_tokens.shape[1] # 4
    if next_dynamic_tokens.ndim == 2:
        B = next_dynamic_tokens.shape[0]
        next_dynamic_tokens = next_dynamic_tokens.reshape(B, sec_num,-1)
    img_pixel_list = []
    for sec in range(sec_num):
        img_token_dict = dict(context=next_context_tokens[:,sec,:], dynamic=next_dynamic_tokens[:,sec,:])
        img_pixel, _ = vq_model.detokenize(img_token_dict,
                                        offset_tokenzier=len(uni_prompting.text_tokenizer),
                                        sptids_dict=uni_prompting.sptids_dict,
                                        )  # (B,T,C,H,W)
        img_pixel = img_pixel[:,condition_len:]
        img_pixel = torch.clamp((img_pixel + 1.0) / 2.0, min=0.0, max=1.0)
        img_pixel_list.append(img_pixel)
    img_pixel = torch.cat(img_pixel_list, dim=1) #(B,8,3,H,W)
    return img_pixel