import os
from typing import Union
import torch
import copy
import time
import pickle

@torch.no_grad()
def tokenize_next_img_dynamic(next_img_context_per_sec_ori, next_img_dynamic, vq_model, uni_prompting, condition_len:None):
    # target_image_encoder_features_next context len 2 
    # target_image_encoder_features_next dynamic len 3
    target_context_next, target_inputs_next, target_labels_next,target_image_encoder_features_next_context,target_image_encoder_features_next_dynamic =  [], [], [], [], []
    next_img_context_per_sec=next_img_context_per_sec_ori
    for sec in range(next_img_context_per_sec.shape[1]): # 0,1,2,3
        cur_next_img_context = next_img_context_per_sec[:, sec]
        cur_next_img_dynamic = next_img_dynamic[:, range(sec*2,sec*2+3)]
        input_ids_next, labels_next,image_encoder_features_next = vq_model.tokenize(
                                            cur_next_img_dynamic,
                                            context_pixel_values=cur_next_img_context,
                                            context_length=condition_len,
                                            special_token=uni_prompting.sptids_dict,
                                            return_label=False,
                                            return_encoder_features=True)
        vocab_offset = len(uni_prompting.text_tokenizer)
        for k, v in input_ids_next.items():#context and dynamic
            mask = (v > 0) & (v < (vocab_offset - (len(uni_prompting.sptids_dict)-2)))
            input_ids_next[k][mask] += vocab_offset
            if k in labels_next:
                mask_label = (labels_next[k] > 0) & (labels_next[k] < (vocab_offset - (len(uni_prompting.sptids_dict)-2)))
                labels_next[k][mask_label] += vocab_offset
        target_context_next.append(input_ids_next['context'])
        target_inputs_next.append(input_ids_next['dynamic'])
        target_labels_next.append(labels_next['dynamic'])
        target_image_encoder_features_next_context.append(image_encoder_features_next['context'][0])
        target_image_encoder_features_next_dynamic.append(image_encoder_features_next['dynamic'])
    target_context_next = torch.stack(target_context_next, dim=1) #(B,4,900)
    target_inputs_next = torch.stack(target_inputs_next, dim=1) #(B,4,60) [:-30]
    target_labels_next = torch.stack(target_labels_next, dim=1) #(B,4,60) [30:]
    target_image_encoder_features_next_context = torch.stack(target_image_encoder_features_next_context, dim=1)
    target_image_encoder_features_next_dynamic = torch.stack(target_image_encoder_features_next_dynamic, dim=1)
    target_image_encoder_features_next = {}
    target_image_encoder_features_next["context"] = [target_image_encoder_features_next_context]
    target_image_encoder_features_next["dynamic"] = target_image_encoder_features_next_dynamic
    return target_context_next, target_inputs_next, target_labels_next,target_image_encoder_features_next
class InputIDsPreparer:
    """
    输入ID准备器类，负责处理图像和深度数据的tokenization和标签准备
    """

    def __init__(self, uni_prompting, vq_model=None, da3=None, accelerator=None, data_loader=None):
        """
        初始化输入ID准备器
        
        Args:
            uni_prompting: 统一提示处理器
            vq_model: 向量化量化模型
            da3: 深度编码器
            accelerator: 加速器
            data_loader: 数据加载器（用于验证prediction对象）
        """
        self.uni_prompting = uni_prompting
        self.vq_model = vq_model
        self.da3 = da3
        self.accelerator = accelerator
        self.data_loader = data_loader

    def _add_vocab_offset(self, input_ids_next, labels_next, device):
        """
        为输入ID和标签添加词汇表偏移量的公共方法
        
        Args:
            input_ids_next: 输入token IDs字典
            labels_next: 标签字典
            device: 设备张量
        """
        vocab_offset = len(self.uni_prompting.text_tokenizer)  # add offset
        id_sod = self.uni_prompting.sptids_dict["<|sop|>"].to(device)
        id_eod = self.uni_prompting.sptids_dict["<|eop|>"].to(device)

        for k, v in input_ids_next.items():  # context and dynamic
            copy_c = copy.deepcopy(v)
            mask = (v > 0) & (v < (vocab_offset - (len(self.uni_prompting.sptids_dict) - 2)))
            input_ids_next[k][mask] += vocab_offset
            if torch.where(input_ids_next[k] == id_sod)[0].shape[0] != torch.where(input_ids_next[k] == id_eod)[0].shape[0] \
                or torch.where(input_ids_next[k] == id_sod)[1].shape[0] != torch.where(input_ids_next[k] == id_eod)[1].shape[0]:
                assert False
                pass
            if k in labels_next:
                mask_label = (labels_next[k] > 0) & (labels_next[k] < (vocab_offset - (len(self.uni_prompting.sptids_dict) - 2)))
                labels_next[k][mask_label] += vocab_offset

    def _replace_image_tokens_with_depth_tokens(self, input_ids, device):
        """
        将图像相关特殊token替换为深度相关特殊token
        soi -> sop, eoi -> eop, sod -> sop, eod -> eop
        
        Args:
            input_ids: 输入token IDs字典
            device: 设备张量
        """
        # 获取特殊token ID
        id_soi = self.uni_prompting.sptids_dict["<|soi|>"].to(device)
        id_eoi = self.uni_prompting.sptids_dict["<|eoi|>"].to(device)
        id_sod = self.uni_prompting.sptids_dict["<|sod|>"].to(device)
        id_eod = self.uni_prompting.sptids_dict["<|eod|>"].to(device)
        id_sop = self.uni_prompting.sptids_dict["<|sop|>"].to(device)
        id_eop = self.uni_prompting.sptids_dict["<|eop|>"].to(device)
        
        # 替换：soi -> sop, eoi -> eop, sod -> sop, eod -> eop
        for k, v in input_ids.items():
            # 先复制原始数据，然后基于上一次的结果进行替换
            input_ids[k] = v.clone()
            input_ids[k] = torch.where(input_ids[k] == id_soi, id_sop, input_ids[k])
            input_ids[k] = torch.where(input_ids[k] == id_eoi, id_eop, input_ids[k])
            input_ids[k] = torch.where(input_ids[k] == id_sod, id_sop, input_ids[k])
            input_ids[k] = torch.where(input_ids[k] == id_eod, id_eop, input_ids[k])
    
    def prepare_inputs_and_labels(
            self,
            prev_img_context_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_context_2s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor],
            next_img_context: Union[torch.FloatTensor, torch.LongTensor],
            next_img_dynamic: Union[torch.FloatTensor, torch.LongTensor],
            # next_img_context_per_sec: Union[torch.FloatTensor, torch.LongTensor],
            ego_status: Union[torch.FloatTensor, torch.LongTensor],
            future_trajectories: Union[torch.FloatTensor, torch.LongTensor],
            condition_len: None,
            mode="navsim_alt",
            is_train: bool = True,
            # 深度相关参数改为可选和默认None
            input_depth_ids_prev=None,
            labels_depth_prev=None,
            input_depth_ids_next=None,
            labels_depth_ids_next=None,
            mock=False
    ):
        """
        纯图像处理函数，不包含深度处理逻辑
        深度数据通过参数传入，避免重复处理
        注意：移除了 @torch.no_grad() 装饰器，只对图像 tokenizer 应用 no_grad
        """
        # if mode in ["navsim"]:
            # 使用 torch.no_grad() 包裹图像 tokenizer 调用，因为图像不需要梯度
        with torch.no_grad():
            # 1s - 提取并返回图像编码器特征
            input_ids_prev1s, labels_prev1s, image_encoder_features_1s = self.vq_model.tokenize(
            prev_img_dynamic_1s,
            context_pixel_values=prev_img_context_1s,
            context_length=condition_len,
            special_token=self.uni_prompting.sptids_dict,
            return_label=False,
            return_encoder_features=True
        )  # (batch*T,3,H,W)
            # 2s - 提取并返回图像编码器特征
            input_ids_prev2s, labels_prev2s, image_encoder_features_2s = self.vq_model.tokenize(
                prev_img_dynamic_2s,
                context_pixel_values=prev_img_context_2s,
                context_length=condition_len,
                special_token=self.uni_prompting.sptids_dict,
                return_label=False,
                return_encoder_features=True
            )  # (batch*T,3,H,W)
            # next - 提取并返回图像编码器特征
            # input_ids_next, labels_next, image_encoder_features_next = self.vq_model.tokenize(
            #     next_img_dynamic,
            #     context_pixel_values=next_img_context,
            #     context_length=condition_len,
            #     special_token=self.uni_prompting.sptids_dict,
            #     return_encoder_features=True
            # )  # (batch*T,3,H,W)
            target_context_next, target_inputs_next, target_labels_next,image_encoder_features_next = tokenize_next_img_dynamic(next_img_context, 
                                                next_img_dynamic, 
                                                self.vq_model, self.uni_prompting, condition_len=condition_len)
        
            target_context_next = target_context_next[:,0,:]
            target_inputs_next = target_inputs_next.flatten(1)
            target_labels_next = target_labels_next.flatten(1)
            input_ids_next = dict(context=target_context_next, dynamic=target_inputs_next)
            labels_next = dict(dynamic=target_labels_next)
            
            labels_prev1s = {key_label: torch.ones_like(labels_prev1s[key_label]) * -100 for key_label in labels_prev1s}
            labels_prev2s = {key_label: torch.ones_like(labels_prev2s[key_label]) * -100 for key_label in labels_prev2s}
            # 使用公共的偏移量添加方法
            for input_ids_prev, labels_prev in [(input_ids_prev1s, labels_prev1s), (input_ids_prev2s, labels_prev2s), (input_ids_next, labels_next)]:
                self._add_vocab_offset(input_ids_prev, labels_prev, prev_img_context_1s.device)

            # caption part:
            action_num = future_trajectories.shape[1]
            texts = dict(input_caption=[""] * future_trajectories.shape[0])
            input_ids_prev = [input_ids_prev1s, input_ids_prev2s]
            labels_prev = [labels_prev1s, labels_prev2s]

            final_input_ids, labels = self.uni_prompting((texts, input_ids_prev, input_ids_next, labels_prev, labels_next,
                                                          action_num, ego_status, input_depth_ids_prev, labels_depth_prev,
                                                          input_depth_ids_next, labels_depth_ids_next), mode)
        # else:
        #     raise NotImplementedError
        
        # 返回image_encoder_features，包含分离的context和dynamic特征
        image_encoder_features = {
            '1s': image_encoder_features_1s,  # dict with 'context' and 'dynamic' keys
            '2s': image_encoder_features_2s,  # dict with 'context' and 'dynamic' keys
            'next': image_encoder_features_next  # dict with 'context' and 'dynamic' keys
        }
        return final_input_ids, labels, [input_ids_prev, input_ids_next], image_encoder_features

    def _prepare_depth_data_only(
            self,
            prev_img_context_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_context_2s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor],
            next_img_context: Union[torch.FloatTensor, torch.LongTensor],
            next_img_dynamic: Union[torch.FloatTensor, torch.LongTensor],
            # 旧格式参数
            prev_depth_img_input: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            next_depth_img_input: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            cached_prev_depth: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            cached_next_depth: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            token_id: Union[list, None] = None,
            prev_tokenids = None,
            next_tokenids = None,
            # 新格式参数
            prev_depth_context_1s: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            prev_depth_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            prev_depth_context_2s: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            prev_depth_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            next_depth_context: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            next_depth_dynamic: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            submode="norm",
    ):
        """
        独立的深度数据处理函数，支持新旧两种格式
        
        Returns:
            input_depth_ids_prev, labels_depth_prev, input_depth_ids_next, labels_depth_ids_next
        """
        # 检测使用哪种格式
        use_new_format = (prev_depth_context_1s is not None and
                         prev_depth_dynamic_1s is not None and
                         prev_depth_context_2s is not None and
                         prev_depth_dynamic_2s is not None and
                         next_depth_context is not None and
                         next_depth_dynamic is not None)
        
        if use_new_format:
            # 使用新的context/dynamic格式
            (
                input_ids_prev1s,
                labels_prev1s,
                input_ids_prev2s,
                labels_prev2s,
                input_depth_ids_next,
                labels_depth_ids_next,
            ) = self.da3.depth_tokenizer_context_dynamic(
                prev_depth_context_1s=prev_depth_context_1s,
                prev_depth_dynamic_1s=prev_depth_dynamic_1s,
                prev_depth_context_2s=prev_depth_context_2s,
                prev_depth_dynamic_2s=prev_depth_dynamic_2s,
                next_depth_context=next_depth_context,
                next_depth_dynamic=next_depth_dynamic,
                submode=submode,
            )
        else:
            # 使用旧格式
            (
                input_ids_prev1s,
                labels_prev1s,
                input_ids_prev2s,
                labels_prev2s,
                input_depth_ids_next,
                labels_depth_ids_next,
            ) = self.da3.prepare_depth_inputs_and_labels(
                prev_img_context_1s,
                prev_img_dynamic_1s,
                prev_img_context_2s,
                prev_img_dynamic_2s,
                next_img_context,
                next_img_dynamic,
                prev_depth_img_input,
                next_depth_img_input,
                token_id=token_id,
                cached_prev_depth=cached_prev_depth,
                cached_next_depth=cached_next_depth,
                prev_img_depth_tokenid=prev_tokenids,
                next_img__depth_tokenid=next_tokenids,
            )

        labels_prev1s = {key_label: torch.ones_like(labels_prev1s[key_label]) * -100 for key_label in labels_prev1s}
        labels_prev2s = {key_label: torch.ones_like(labels_prev2s[key_label]) * -100 for key_label in labels_prev2s}
        # 使用公共的偏移量添加方法
        for input_ids_prev, labels_prev in [
            (input_ids_prev1s, labels_prev1s),
            (input_ids_prev2s, labels_prev2s),
            (
                input_depth_ids_next,
                labels_depth_ids_next,
            ),
        ]:
            self._add_vocab_offset(input_ids_prev, labels_prev, input_ids_prev1s['context'].device)
        input_depth_ids_prev = [input_ids_prev1s, input_ids_prev2s]
        labels_prev = [labels_prev1s, labels_prev2s]
        return (
            input_depth_ids_prev,
            labels_prev,
            input_depth_ids_next,
            labels_depth_ids_next,
        )
    def prepare_inputs_and_labels_all(
            self,
            prev_img_context_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_context_2s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor],
            next_img_context: Union[torch.FloatTensor, torch.LongTensor],
            next_img_dynamic: Union[torch.FloatTensor, torch.LongTensor],
            # 旧格式参数
            prev_depth_img_input: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            next_depth_img_input: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            cached_prev_depth: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            cached_next_depth: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            # 新格式参数
            prev_depth_context_1s: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            prev_depth_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            prev_depth_context_2s: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            prev_depth_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            next_depth_context: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            next_depth_dynamic: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            # 其他参数
            ego_status: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            future_trajectories: Union[torch.FloatTensor, torch.LongTensor, None] = None,
            condition_len: None = None,
            mode="navsim",
            is_train: bool = True,
            token_id: Union[list, None] = None,
            prev_tokenids = None,
            next_tokenids = None,
            submode="norm",
    ):
        """
        重构后的统一处理函数：
        1. 先处理深度数据（只调用一次）
        2. 再处理原始图像数据
        3. 合并结果
        """
        # 步骤1: 独立处理深度数据（自动检测格式）
        input_depth_ids_prev, labels_depth_prev, input_depth_ids_next, labels_depth_ids_next = self._prepare_depth_data_only(
            prev_img_context_1s, prev_img_dynamic_1s, prev_img_context_2s, prev_img_dynamic_2s,
            next_img_context, next_img_dynamic,
            prev_depth_img_input, next_depth_img_input, cached_prev_depth, cached_next_depth,
            token_id=token_id, prev_tokenids=prev_tokenids, next_tokenids=next_tokenids,
            prev_depth_context_1s=prev_depth_context_1s,
            prev_depth_dynamic_1s=prev_depth_dynamic_1s,
            prev_depth_context_2s=prev_depth_context_2s,
            prev_depth_dynamic_2s=prev_depth_dynamic_2s,
            next_depth_context=next_depth_context,
            next_depth_dynamic=next_depth_dynamic,
            submode=submode,
        )

        # 步骤2: 处理原始图像数据（传入深度处理结果）
        final_input_ids, labels, image_tokens_ori, image_encoder_features = self.prepare_inputs_and_labels(
            prev_img_context_1s, prev_img_dynamic_1s, prev_img_context_2s, prev_img_dynamic_2s,
            next_img_context, next_img_dynamic, ego_status, future_trajectories, condition_len,
            mode, is_train,
            input_depth_ids_prev=input_depth_ids_prev,
            labels_depth_prev=labels_depth_prev,
            input_depth_ids_next=input_depth_ids_next,
            labels_depth_ids_next=labels_depth_ids_next
        )

        return final_input_ids, labels, image_tokens_ori, input_depth_ids_prev, labels_depth_prev, input_depth_ids_next, labels_depth_ids_next, image_encoder_features

    def _prepare_batch_data(self, batch, config, is_train=True, use_context_dynamic_depth=False):
        """
        准备批次数据的基础方法
        
        Args:
            batch: 输入批次数据
            config: 配置对象
            is_train: 是否为训练模式
            use_context_dynamic_depth: 是否使用新的context/dynamic深度数据格式
        
        Returns:
            解包后的所有数据
        """
        batch_size = batch["next_img_context"].shape[0]
        
        # 先处理基本的数据移动
        prev_img_context_1s = batch['prev_img_context_1s'].to(self.accelerator.device, non_blocking=True)
        prev_img_dynamic_1s = batch['prev_img_dynamic_1s'].to(self.accelerator.device, non_blocking=True)
        prev_img_context_2s = batch['prev_img_context_2s'].to(self.accelerator.device, non_blocking=True)
        prev_img_dynamic_2s = batch['prev_img_dynamic_2s'].to(self.accelerator.device, non_blocking=True)
        next_img_context = batch['next_img_context'].to(self.accelerator.device, non_blocking=True)
        next_img_dynamic = batch['next_img_dynamic'].to(self.accelerator.device, non_blocking=True)        
        
        # next_img_context_per_sec = batch['next_img_context_per_sec'].to(self.accelerator.device, non_blocking=True)
        # 处理深度数据 - 支持两种格式
        if use_context_dynamic_depth:
            # 新的context/dynamic格式
            prev_depth_context_1s = batch.get('prev_depth_context_1s')
            prev_depth_dynamic_1s = batch.get('prev_depth_dynamic_1s')
            prev_depth_context_2s = batch.get('prev_depth_context_2s')
            prev_depth_dynamic_2s = batch.get('prev_depth_dynamic_2s')
            next_depth_context = batch.get('next_depth_context')
            next_depth_dynamic = batch.get('next_depth_dynamic')
            
            # 如果存在则移动到设备
            if prev_depth_context_1s is not None and isinstance(prev_depth_context_1s, torch.Tensor):
                prev_depth_context_1s = prev_depth_context_1s.to(self.accelerator.device, non_blocking=True)
            if prev_depth_dynamic_1s is not None and isinstance(prev_depth_dynamic_1s, torch.Tensor):
                prev_depth_dynamic_1s = prev_depth_dynamic_1s.to(self.accelerator.device, non_blocking=True)
            if prev_depth_context_2s is not None and isinstance(prev_depth_context_2s, torch.Tensor):
                prev_depth_context_2s = prev_depth_context_2s.to(self.accelerator.device, non_blocking=True)
            if prev_depth_dynamic_2s is not None and isinstance(prev_depth_dynamic_2s, torch.Tensor):
                prev_depth_dynamic_2s = prev_depth_dynamic_2s.to(self.accelerator.device, non_blocking=True)
            if next_depth_context is not None and isinstance(next_depth_context, torch.Tensor):
                next_depth_context = next_depth_context.to(self.accelerator.device, non_blocking=True)
            if next_depth_dynamic is not None and isinstance(next_depth_dynamic, torch.Tensor):
                next_depth_dynamic = next_depth_dynamic.to(self.accelerator.device, non_blocking=True)
        else:
            # 新的字段设为None
            prev_depth_context_1s = None
            prev_depth_dynamic_1s = None
            prev_depth_context_2s = None
            prev_depth_dynamic_2s = None
            next_depth_context = None
            next_depth_dynamic = None
        # 旧的深度数据格式 "processed_prev","processed_next",
        if isinstance(batch.get('processed_prev'), torch.Tensor):
            prev_depth_img_input = batch['processed_prev'].to(self.accelerator.device, non_blocking=True)
        else:
            prev_depth_img_input = None
        if isinstance(batch.get('processed_next'), torch.Tensor):
            next_depth_img_input = batch['processed_next'].to(self.accelerator.device, non_blocking=True)
        else:
            next_depth_img_input = None
        if isinstance(batch.get('processed_next_context'), torch.Tensor):
            processed_next_context = batch['processed_next_context'].to(self.accelerator.device, non_blocking=True)
        else:
            processed_next_context = None
        # 确保 token 保持 list 格式，按 batch 维度展开
        token = batch['token']  # 这应该是一个 list，长度为 batch_size
        ego_status = batch['ego_status'].to(self.accelerator.device, non_blocking=True) if config.experiment.add_ego else None

        # 安全地处理缓存数据：先检查是否存在且有效，再进行设备移动
        cached_prev_depth = batch.get('cached_prev_depth', None)
        cached_next_depth = batch.get('cached_next_depth', None)
        cached_next_depth_context =  batch.get('cached_next_depth_context', None) # 没有经过reahspe 的展评状态
        prev_tokenids = batch.get('prev_tokenids', None)
        next_tokenids = batch.get('next_tokenids', None)
        next_context_tokenids = batch.get('next_context_tokenids',None) # 处理国的

        if is_train:
            # 训练模式：使用真实的future_trajectories
            future_trajectories = batch['future_trajectory'].to(self.accelerator.device, non_blocking=True)
        else:
            # 测试模式：创建零张量作为future_trajectories
            future_trajectories = torch.zeros_like(batch['future_trajectory'], device=self.accelerator.device, dtype=batch['future_trajectory'].dtype)

        # if use_context_dynamic_depth:
        return (batch_size, prev_img_context_1s, prev_img_dynamic_1s, prev_img_context_2s,
                prev_img_dynamic_2s, next_img_context, next_img_dynamic, token, ego_status,
                future_trajectories, prev_depth_context_1s, prev_depth_dynamic_1s,
                prev_depth_context_2s, prev_depth_dynamic_2s, next_depth_context, next_depth_dynamic, 
                prev_depth_img_input, next_depth_img_input,cached_prev_depth, cached_next_depth, 
                prev_tokenids, next_tokenids,next_context_tokenids,processed_next_context,cached_next_depth_context)
       

    def _create_attention_mask(self, input_ids, mask_dtype, create_attention_mask_for_nusc):
        """
        创建注意力掩码的通用方法
        
        Args:
            input_ids: 输入token IDs
            mask_dtype: 掩码数据类型
            create_attention_mask_for_nusc: 创建掩码的函数
            
        Returns:
            注意力掩码
        """
        attention_mask = create_attention_mask_for_nusc(input_ids,  # (B,1,L,L)
                                                         pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                         soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                         eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                         sod_id=int(self.uni_prompting.sptids_dict['<|sod|>']),
                                                         eod_id=int(self.uni_prompting.sptids_dict['<|eod|>']),
                                                         sop_id=int(self.uni_prompting.sptids_dict['<|sop|>']),
                                                         eop_id=int(self.uni_prompting.sptids_dict['<|eop|>']),
                                                         rm_pad_in_image=True,
                                                         return_inverse_mask=True,
                                                         mask_future_ratio=None)
        return attention_mask.to(mask_dtype)

    def prepare_one_batch(self, batch, config, mask_dtype, create_attention_mask_for_nusc,
                         is_train=True, data_time_m=None, end=None, model=None, submode="norm_transform",
                         return_depth_features=False):
        """
        统一的批次准备方法，支持训练和测试模式，自动检测深度数据格式
        
        Args:
            batch: 输入批次数据
            config: 配置对象
            mask_dtype: 掩码数据类型
            create_attention_mask_for_nusc: 创建注意力掩码的函数
            is_train: 是否为训练模式
            data_time_m: 训练时的时间统计器（仅训练模式需要）
            end: 训练时的结束时间（仅训练模式需要）
            model: 模型对象（仅训练模式需要）
            submode: 深度数据处理模式（norm等）
            return_depth_features: 是否返回深度特征（不使用VAE tokenizer，使用CNN提取）
            
        Returns:
            当return_depth_features=False时: input_ids, labels, image_tokens_ori, attention_mask
            当return_depth_features=True时: input_ids, labels, image_tokens_ori, attention_mask, depth_features_dict
        """
        # 检测使用哪种深度数据格式
        use_context_dynamic_depth = (
            'prev_depth_context_1s' in batch and
            'prev_depth_dynamic_1s' in batch and
            'prev_depth_context_2s' in batch and
            'prev_depth_dynamic_2s' in batch and
            'next_depth_context' in batch and
            'next_depth_dynamic' in batch
        )
        build_data = bool(int(os.getenv("RUN_FLASH_DATA_LOADER", 0)))
                
        # 准备批次数据（自动检测深度格式）
        (batch_size, prev_img_context_1s, prev_img_dynamic_1s, prev_img_context_2s,
                prev_img_dynamic_2s, next_img_context, next_img_dynamic, token, ego_status,
                future_trajectories, prev_depth_context_1s, prev_depth_dynamic_1s,
                prev_depth_context_2s, prev_depth_dynamic_2s, next_depth_context,
                next_depth_dynamic, prev_depth_img_input, next_depth_img_input,
                cached_prev_depth, cached_next_depth, prev_tokenids, next_tokenids,
                next_context_tokenids,processed_next_context,cached_next_depth_context) = self._prepare_batch_data(batch, config, is_train, use_context_dynamic_depth=use_context_dynamic_depth)
                
        if build_data or  isinstance(prev_depth_context_1s, list):
            return  self.da3.depth_loader.process_8frame(
                    None,
                    prev_tokenids,
                    next_tokenids,
                    next_context_tokenids,
                    prev_depth_img_input,
                    next_depth_img_input,
                    processed_next_context,
                    cached_prev_depth,
                    cached_next_depth,
                    cached_next_depth_context,
                )
            

        # 训练模式下的时间统计
        if is_train and data_time_m is not None and end is not None:
            data_time_m.update(time.time() - end)

        # with torch.no_grad():
        # 构建格式化序列
        context_length = config.dataset.ctd.context_length
        
        if return_depth_features:
            # 模式1: 返回深度特征（不使用VAE tokenizer，使用CNN提取）
            # 步骤1: 准备图像数据（不带深度token），同时获取image_encoder_features
            input_ids, labels, image_tokens_ori, image_encoder_features = self.prepare_inputs_and_labels(
                prev_img_context_1s=prev_img_context_1s,
                prev_img_dynamic_1s=prev_img_dynamic_1s,
                prev_img_context_2s=prev_img_context_2s,
                prev_img_dynamic_2s=prev_img_dynamic_2s,
                next_img_context=next_img_context,
                next_img_dynamic=next_img_dynamic,
                # next_img_context_per_sec=next_img_context_per_sec,
                ego_status=ego_status,
                future_trajectories=future_trajectories,
                condition_len=context_length,
                mode="navsim_alt",
                is_train=is_train,
                # 深度数据设为None，不使用VAE tokenizer
                input_depth_ids_prev=None,
                labels_depth_prev=None,
                input_depth_ids_next=None,
                labels_depth_ids_next=None,
            )
            
            # 步骤2: 将image_encoder_features转换为prepare_depth_features需要的格式
            image_encoder_features_dict = None
            if self.vq_model is not None:
                image_encoder_features_dict = {
                    '1s': image_encoder_features['1s'],
                    '2s': image_encoder_features['2s'],
                    'next': image_encoder_features['next'],
                }
            
            # 步骤3: 准备深度特征（不使用VAE，使用CNN提取）
            depth_features_dict = None
            if (prev_depth_context_1s is not None and prev_depth_dynamic_1s is not None and
                prev_depth_context_2s is not None and prev_depth_dynamic_2s is not None and
                next_depth_context is not None and next_depth_dynamic is not None):
                
                depth_features_dict = self.prepare_depth_features(
                    prev_depth_context_1s=prev_depth_context_1s,
                    prev_depth_dynamic_1s=prev_depth_dynamic_1s,
                    prev_depth_context_2s=prev_depth_context_2s,
                    prev_depth_dynamic_2s=prev_depth_dynamic_2s,
                    next_depth_context=next_depth_context,
                    next_depth_dynamic=next_depth_dynamic,
                    image_encoder_features=image_encoder_features_dict,
                    submode=submode,
                )
                # 步骤4: 创建注意力掩码
                attention_mask = self._create_attention_mask(input_ids, mask_dtype, create_attention_mask_for_nusc)
                
                # 返回时包含depth_features_dict
                return input_ids, labels, image_tokens_ori, attention_mask, depth_features_dict
        else:
            # 模式2: 标准模式（使用VAE tokenizer处理深度数据）#todo 没有适配间隔侦next_img_context_per_sec
            (input_ids, labels, image_tokens_ori, input_depth_ids_prev, labels_depth_prev,
                input_depth_ids_next, labels_depth_ids_next, image_encoder_features) = self.prepare_inputs_and_labels_all(
                prev_img_context_1s=prev_img_context_1s,
                prev_img_dynamic_1s=prev_img_dynamic_1s,
                prev_img_context_2s=prev_img_context_2s,
                prev_img_dynamic_2s=prev_img_dynamic_2s,
                next_img_context=next_img_context,
                next_img_dynamic=next_img_dynamic,
                # 旧格式参数
                prev_depth_img_input=prev_depth_img_input,
                next_depth_img_input=next_depth_img_input,
                cached_prev_depth=cached_prev_depth,
                cached_next_depth=cached_next_depth,
                # 新格式参数
                prev_depth_context_1s=prev_depth_context_1s,
                prev_depth_dynamic_1s=prev_depth_dynamic_1s,
                prev_depth_context_2s=prev_depth_context_2s,
                prev_depth_dynamic_2s=prev_depth_dynamic_2s,
                next_depth_context=next_depth_context,
                next_depth_dynamic=next_depth_dynamic,
                # 其他参数
                ego_status=ego_status,
                future_trajectories=future_trajectories,
                condition_len=context_length,
                token_id=token,  # 传递整个token列表，而不是第一个元素
                prev_tokenids=prev_tokenids,
                next_tokenids=next_tokenids,
                submode=submode,
            )

            # 创建注意力掩码
            attention_mask = self._create_attention_mask(input_ids, mask_dtype, create_attention_mask_for_nusc)

            return input_ids, labels, image_tokens_ori, attention_mask, None

    def prepare_one_batch_test(self, batch, config, mask_dtype, create_attention_mask_for_nusc, return_depth_features=True):
        """
        测试模式的批次准备方法（向后兼容）
        """
        return self.prepare_one_batch(batch, config, mask_dtype, create_attention_mask_for_nusc,
                                      is_train=False, return_depth_features=return_depth_features)
    
    def prepare_one_batch_train(self, batch, config, data_time_m, end, model, mask_dtype,
                               create_attention_mask_for_nusc, return_depth_features=True):
        """
        训练模式的批次准备方法（向后兼容）
        """
        return self.prepare_one_batch(batch, config, mask_dtype, create_attention_mask_for_nusc,
                                     is_train=True, data_time_m=data_time_m, end=end, model=model,
                                     return_depth_features=return_depth_features)
    
    def check_embedding_input(self, embedding_layer, input_ids, name="input"):
        """
        检查输入到embedding层的token ID是否在有效范围内
        """
        vocab_size = embedding_layer.num_embeddings

        if torch.any(input_ids < 0):
            print(f"❌ {name}: 发现负数token ID")
            print(f"   最小值: {input_ids.min().item()}")
            return False

        if torch.any(input_ids >= vocab_size):
            print(f"❌ {name}: token ID超出词汇表范围!")
            print(f"   最大值: {input_ids.max().item()}")
            print(f"   vocab_size: {vocab_size}")
            invalid_count = torch.sum(input_ids >= vocab_size).item()
            print(f"   超出范围的token数量: {invalid_count}")
            return False

        return True

    # @torch.no_grad()
    def prepare_depth_features(
            self,
            prev_depth_context_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_depth_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_depth_context_2s: Union[torch.FloatTensor, torch.LongTensor],
            prev_depth_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor],
            next_depth_context: Union[torch.FloatTensor, torch.LongTensor],
            next_depth_dynamic: Union[torch.FloatTensor, torch.LongTensor],
            image_encoder_features: dict = None,
            submode="norm_transform",
    ):
        """
        准备深度特征（不使用VAE tokenizer，使用CNN提取特征）
        
        Args:
            prev_depth_context_1s: 1秒前的context深度数据 [B, T_context, H, W]
            prev_depth_dynamic_1s: 1秒前的dynamic深度数据 [B, T_dynamic, H, W]
            prev_depth_context_2s: 2秒前的context深度数据 [B, T_context, H, W]
            prev_depth_dynamic_2s: 2秒前的dynamic深度数据 [B, T_dynamic, H, W]
            next_depth_context: 未来的context深度数据 [B, T_context, H, W]
            next_depth_dynamic: 未来的dynamic深度数据 [B, T_dynamic, H, W]
            image_encoder_features: 图像编码器特征字典（用于cross-attention）
            submode: 深度预处理模式
        
        Returns:
            dict: 深度特征字典
        """
        if self.da3 is None:
            raise ValueError("da3 (DepthEncoder) is required for depth feature extraction")
        
        # 使用 DepthEncoder 的 prepare_depth_features 方法
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            depth_features = self.da3.prepare_depth_features(
                prev_depth_context_1s=prev_depth_context_1s,
                prev_depth_dynamic_1s=prev_depth_dynamic_1s,
                prev_depth_context_2s=prev_depth_context_2s,
                prev_depth_dynamic_2s=prev_depth_dynamic_2s,
                next_depth_context=next_depth_context,
                next_depth_dynamic=next_depth_dynamic,
                image_encoder_features=image_encoder_features,
                submode=submode,
            )
            
            return depth_features
