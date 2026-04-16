"""
Navsim token generation utilities
独立的token生成函数模块
"""
import torch

from data_utils.sequence_visualizer import plot_attention_mask


def frame_token_gen(special_token_s, special_token_e, cur_frame_num, logits, device, total_length, config, B):
    """
    生成frame token
    
    Args:
        special_token_s: 开始token
        special_token_e: 结束token
        cur_frame_num: 当前帧数量
        logits: 模型输出的logits
        device: 设备
        total_length: 当前总长度
        config: 配置对象
        B: batch size
        
    Returns:
        cur_input: 生成的token序列
        total_length: 更新后的总长度
    """
    offset_vcab = config.model.showo.vocab_size
    pred_logits = logits[:, -cur_frame_num:, offset_vcab:]
    total_length += pred_logits.shape[1]
    probs = pred_logits[:, 1:-1].softmax(dim=-1).argmax(-1).to(torch.int64)  # dynamic的codebook是8192
    cur_input = torch.cat(
        [special_token_s[None, ...].to(device).repeat(B, 1),
         probs + offset_vcab,
         special_token_e[None, ...].to(device).repeat(B, 1)], dim=-1)
    return cur_input, total_length


def depth_token_gen(special_token_s, special_token_e, cur_frame_num, logits, device, total_length, uni_prompting, B):
    """
    生成depth token
    
    Args:
        special_token_s: 开始token
        special_token_e: 结束token
        cur_frame_num: 当前帧数量
        logits: 模型输出的logits
        device: 设备
        total_length: 当前总长度
        uni_prompting: 统一提示对象
        B: batch size
        
    Returns:
        cur_input: 生成的token序列
        total_length: 更新后的总长度
    """
    offset_vcab = len(uni_prompting.text_tokenizer)
    pred_logits = logits[:, -cur_frame_num:, offset_vcab:offset_vcab + 8192]
    total_length += pred_logits.shape[1]
    probs = (
        pred_logits[:, 1:-1].softmax(dim=-1).argmax(-1)
    )  # dynamic的codebook是8192

    cur_input = torch.cat(
        [
            special_token_s[None, ...].to(device).repeat(B, 1),
            probs + offset_vcab,
            special_token_e[None, ...].to(device).repeat(B, 1),
        ],
        dim=-1,
    )
    return cur_input, total_length


def frame_gen(device, logits, attention_mask_step_input, total_length, gen_type, 
              frame_num, depth_num, uni_prompting, switch_type, attention_mask_one_frame,
              attention_mask_one_depth, gen_frames, genframe_token, gendepth_token,
              showo_model, config, B):
    """
    生成frame或depth token
    
    Args:
        cur_input: 当前输入
        logits: 模型输出的logits
        attention_mask_step_input: 注意力掩码
        total_length: 当前总长度
        gen_type: 生成类型 ('frame' 或 'depth')
        frame_num: 帧数量
        depth_num: depth数量
        uni_prompting: 统一提示对象
        switch_type: 是否切换类型
        attention_mask_one_frame: 单帧注意力掩码
        attention_mask_one_depth: 单depth注意力掩码
        gen_frames: 生成帧数
        genframe_token: 生成的frame token列表
        gendepth_token: 生成的depth token列表
        showo_model: showo模型
        config: 配置对象
        B: batch size
        
    Returns:
        cur_input: 更新后的输入
        attention_mask_step_input: 更新后的注意力掩码
        total_length: 更新后的总长度
        genframe_token: 更新后的frame token列表
    """
    cur_frame_num = 30
    back_gen_type = 'frame'
    if gen_type == 'frame':
        cur_frame_num = frame_num
        special_token_e = uni_prompting.sptids_dict['<|eod|>']
        special_token_s = uni_prompting.sptids_dict['<|sod|>']
        attention_mask_one_frame_cur = attention_mask_one_frame
        if switch_type:
            back_gen_type = 'depth'
        cur_input, total_length = frame_token_gen(
            special_token_s,
            special_token_e,
            cur_frame_num,
            logits,
            device,
            total_length,
            config,
            B,
        )
    elif gen_type == 'depth':
        cur_frame_num = depth_num
        special_token_e = uni_prompting.sptids_dict['<|eop|>']
        special_token_s = uni_prompting.sptids_dict['<|sop|>']
        attention_mask_one_frame_cur = attention_mask_one_depth
        if switch_type:
            back_gen_type = "frame"
        # cur_input, total_length = depth_token_gen(
        #     special_token_s,
        #     special_token_e,
        #     cur_frame_num,
        #     logits,
        #     device,
        #     total_length,
        #     uni_prompting,
        #     B,
        # )
        cur_input, total_length = frame_token_gen(
            special_token_s,
            special_token_e,
            cur_frame_num,
            logits,
            device,
            total_length,
            config,
            B,
        )
    else:
        pass

    if gen_type == 'frame':
        genframe_token.append(cur_input)
        cur_len = len(genframe_token)
    else:
        gendepth_token.append(cur_input)
        cur_len = len(gendepth_token)
    cur_input_emddings = showo_model.embed_tokens(cur_input.to(torch.long))
    if cur_len < gen_frames:
        attention_mask_step_input_last = attention_mask_step_input[
            :, :, -1:
        ].expand(-1, -1, cur_frame_num, -1)
        attention_mask_step_input = torch.cat((attention_mask_step_input_last, attention_mask_one_frame_cur), -1)  # True 为mask

    return cur_input_emddings, attention_mask_step_input, total_length, genframe_token, back_gen_type,cur_input


def group_gen(device, logits, attention_mask_step_input, total_length, attension_type,
              frame_num, depth_num, uni_prompting, switch_type, attention_mask_one_frame,
              attention_mask_one_depth, attention_unmask_one_depth_before_frame,
              gen_frames, genframe_token, gendepth_token, showo_model, config, B):
    """
    生成group token (同时生成depth和frame)
    
    Args:
        cur_input: 当前输入
        logits: 模型输出的logits
        attention_mask_step_input: 注意力掩码
        total_length: 当前总长度
        gen_type: 生成类型
        frame_num: 帧数量
        depth_num: depth数量
        uni_prompting: 统一提示对象
        switch_type: 是否切换类型
        attention_mask_one_frame: 单帧注意力掩码
        attention_mask_one_depth: 单depth注意力掩码
        attention_unmask_one_depth_before_frame: depth在前frame在后的注意力掩码
        gen_frames: 生成帧数
        genframe_token: 生成的frame token列表
        gendepth_token: 生成的depth token列表
        showo_model: showo模型
        config: 配置对象
        B: batch size
        
    Returns:
        cur_input: 更新后的输入
        attention_mask_step_input: 更新后的注意力掩码
        total_length: 更新后的总长度
        genframe_token: 更新后的frame token列表
    """
    (
        cur_input_depth,
        attention_mask_step_input_depth,
        total_length,
        genframe_token,
        gen_type,cur_input_ids_frame
    ) = frame_gen(
        device,
        logits[:, -frame_num - depth_num : -frame_num],
        attention_mask_step_input,
        total_length,
        "depth",
        frame_num,
        depth_num,
        uni_prompting,
        switch_type,
        attention_mask_one_frame,
        attention_mask_one_depth,
        gen_frames,
        genframe_token,
        gendepth_token,
        showo_model,
        config,
        B,
    )
    (
        cur_input_frame,
        attention_mask_step_input_frame,
        total_length,
        genframe_token,
        gen_type,cur_input_ids_depth
    ) = frame_gen(
        device,
        logits[:, -frame_num:],
        attention_mask_step_input_depth,
        total_length,
        "frame",
        frame_num,
        depth_num,
        uni_prompting,
        switch_type,
        attention_mask_one_frame,
        attention_mask_one_depth,
        gen_frames,
        genframe_token,
        gendepth_token,
        showo_model,
        config,
        B,
    )
    if len(genframe_token) < gen_frames:
        if attension_type == "group_attension":
            attention_mask_step_input_depth = torch.cat((attention_mask_step_input_depth, attention_unmask_one_depth_before_frame), -1)
            attention_mask_step_input = torch.cat((attention_mask_step_input_depth, attention_mask_step_input_frame), -2)
        else:
            # todo
            attention_mask_step_input = attention_mask_step_input_depth[
                :, :, -1:
            ].expand(-1, -1, frame_num + depth_num, -1)
    # plot_attention_mask(attention_mask_step_input[0,0], "attention_mask_step_input.png")
    cur_input_back = torch.cat((cur_input_depth, cur_input_frame), dim=1)
    #todo
    cur_input_ids_all = torch.cat((cur_input_ids_depth, cur_input_ids_frame), dim=1)
    return cur_input_back, attention_mask_step_input, total_length,cur_input_ids_all
