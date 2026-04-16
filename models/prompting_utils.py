# coding=utf-8
# Copyright 2024 NUS Show Lab.
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
from data_utils.pwm_dataset import preprocess_multimodal, preprocess_v0
import torch
import copy
# from memory_profiler import profile
from llava.llava import conversation as conversation_lib
from data_utils.sequence_visualizer import *
DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_PROMPT_LEN = 28
# TODO - SHOULD BE FURTHER IMPROVED.
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 special_tokens=("<|soi|>", "<|eoi|>", "<|sod|>", "<|eod|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                 max_text_len=8000, max_seq_len=377, ignore_id=-100, cond_dropout_prob=0.1 ,skip_len = 0):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.text_tokenizer.add_special_tokens({'img_token': '<image>'})
        self.text_tokenizer.add_tokens(list(special_tokens))
        # 显式指定 dtype=torch.long，避免 bfloat16 精度丢失
        self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token]), dtype=torch.long) for token in
                            special_tokens}
        self.sptids_dict['<|sot|>'] = torch.tensor([self.text_tokenizer.bos_token_id], dtype=torch.long)
        self.sptids_dict['<|eot|>'] = torch.tensor([self.text_tokenizer.eos_token_id], dtype=torch.long)
        self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id], dtype=torch.long)
        # plus 1 because at this time we add a task token before
        self.max_text_len = max_text_len + 1
        self.pad_id = self.text_tokenizer.convert_tokens_to_ids('[PAD]')
        # self.img_id = self.text_tokenizer.convert_tokens_to_ids('<image>')
        self.ignore_id = ignore_id
        self.cond_dropout_prob = cond_dropout_prob
        self.add__depth_special_tokens(text=["<|sop|>", "<|eop|>"], skip_count=skip_len + 8192)

    def add__depth_special_tokens(self, text=["<|sop|>", "<|eop|>"], skip_count=8192):
        """
        在指定位置添加特殊词汇，可以跳过一定数量的ID
        由于词表被其他模块占用，此方法只在本地和Python层建立映射
        
        Args:
            text: 要添加的token列表
            skip_count: 要跳过的ID数量
        
        Returns:
            添加的token数量
        """
        # original_vocab_size = len(self.text_tokenizer)

        for idx, token in enumerate(text):
            target_id = skip_count + idx + 99999

            # # 只更新Python层的映射字典
            # self.text_tokenizer.added_tokens_encoder[token] = target_id
            # self.text_tokenizer.added_tokens_decoder[target_id] = token

            # 更新本地sptids_dict（这是最重要的，因为你的代码主要使用这个）
            # 显式指定 dtype=torch.long，避免 bfloat16 精度丢失
            self.sptids_dict[token] = torch.tensor([target_id], dtype=torch.long)

        return len(text)

    def _get_token_counts_from_special_tokens(self, image_tensor, dynamic_tensor, depth_c_tensor, depth_d_tensor):
        """
        通过输入张量中相邻特殊字符的位置计算各种token的数量
        
        Args:
            image_tensor: 包含图片特殊字符的张量
            dynamic_tensor: 包含dynamic特殊字符的张量
            depth_tensor: 包含depth特殊字符的张量
            
        Returns:
            tuple: (image_tokens, dynamic_tokens, depth_tokens)
        """
        def calculate_token_distance(sequence, start_token_id, end_token_id, default_value=30):
            """
            使用torch.where计算特殊字符之间的token数量
            
            Args:
                sequence: 输入序列张量
                start_token_id: 起始特殊字符ID
                end_token_id: 结束特殊字符ID
                default_value: 默认值，当找不到特殊字符时使用
                
            Returns:
                int: token数量
            """
            try:
                # 使用torch.where查找特殊字符位置
                start_positions = torch.where(sequence == start_token_id)[0]
                end_positions = torch.where(sequence == end_token_id)[0]

                if len(start_positions) == 0 or len(end_positions) == 0:
                    return default_value

                # 找到最小的有效距离
                min_distance = float('inf')
                for start_pos in start_positions:
                    for end_pos in end_positions:
                        if end_pos > start_pos:  # 确保结束位置在开始位置之后
                            distance = int(end_pos - start_pos - 1)  # 减1不包括特殊字符本身
                            min_distance = min(min_distance, distance)

                return min_distance if min_distance != float('inf') else default_value

            except Exception as e:
                print(f"计算token距离时出错: {e}")
                raise
                return default_value

        # 获取特殊字符的ID
        soi_id = int(self.sptids_dict['<|soi|>'])
        eoi_id = int(self.sptids_dict['<|eoi|>'])
        sod_id = int(self.sptids_dict['<|sod|>'])
        eod_id = int(self.sptids_dict['<|eod|>'])
        sop_id = int(self.sptids_dict['<|sop|>'])
        eop_id = int(self.sptids_dict['<|eop|>'])

        # 计算各种token的数量
        # 对于图片token: 使用soi和eoi
        image_tokens = calculate_token_distance(image_tensor[0], soi_id, eoi_id, 450-2)

        # 对于dynamic token: 使用sod和eod
        dynamic_tokens = calculate_token_distance(
            dynamic_tensor[0], sod_id, eod_id, 30-2
        )

        # 对于depth token: 使用sop和eop
        if depth_c_tensor is not None:
            depth_c_tokens = calculate_token_distance(depth_c_tensor[0], sop_id, eop_id, 450-2)
        else:
            depth_c_tokens = 0
        if depth_d_tensor is not None:
            depth__d_tokens = calculate_token_distance(depth_d_tensor[0], sop_id, eop_id, 30-2)
        else:
            depth__d_tokens = 0
            
        return int(image_tokens + 2), int(dynamic_tokens + 2), int(depth_c_tokens + 2), int(depth__d_tokens + 2)

    def conversation_tample(self,text, mode="t2d"):
        if mode in ["t2d", "d2t", "t2d_gen"]:
            return [
                {
                    'from': 'USER',
                    'value': DEFAULT_IMAGE_TOKEN + '\n' + text + '\n'
                },
                {
                    'from': 'ASSISTANT',
                    'value': text

                }
            ]
        elif mode == "d2t":
            return [
                {
                    'from': 'USER',
                    'value': DEFAULT_IMAGE_TOKEN + '\n' + "Briefly and precisely describe the current driving scene." + '\n'
                },
                {
                    'from': 'ASSISTANT',
                    'value': text

                }
            ]
        elif mode in ["nusc","nusc_wo_d","nusc_add_front"]:
            return [
                {
                    'from': 'USER',
                    'value': DEFAULT_IMAGE_TOKEN + '\n' + "What should be the next move?" + '\n'
                },
                {
                    'from': 'ASSISTANT',
                    'value': text
                    # + self.max_length*DEFAULT_IMAGE_TOKEN+'\n' # 设置 'value' 属性为 DEFAULT_VIDEO_TOKEN 和换行符
                }
            ]
        elif mode in ["navsim","navsim_1s","navsim_140o","navsim_no_future","navsim_alt"]:
            return [
                {
                    'from': 'USER',
                    'value': DEFAULT_IMAGE_TOKEN + '\n' + "Before planning, imagine what will happen in the immediate future." + '\n'
                },
                {
                    'from': 'ASSISTANT',
                    'value': ''

                }
            ]
        elif mode == "mmu":
            for role in text:
                if role['from'] == 'human':
                    role['from'] = 'USER'
                elif role['from'] == 'gpt':
                    role['from'] = 'ASSISTANT'
            return text
        else:
            return [
                {
                    'from': 'USER',
                    'value': DEFAULT_IMAGE_TOKEN + '\n' + text['Q'] + '\n'

                },
                {
                    'from': 'ASSISTANT',
                    'value': text['A']
                }
            ]

    def t2i_prompt(self, text_ids, image_ids, labels, T, clip_len):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        self.max_text_len = 3000
        conditoin_ids = [x.flatten(0) for x in clip_len]#
        for i in range(clip_len.shape[0]):#per batch
            # caption+perception
            if text_ids['input_p1_q_a']['input_ids'][i][0] != self.text_tokenizer.bos_token_id:
                # [soi][clip pad tokens][eoi]+[sot]+[caption tokens][eot]+[sot]+[p1 tokens][eot]
                input_ids = conditoin_ids[i].tolist()+ \
                            [self.text_tokenizer.bos_token_id] + text_ids['input_p1_q_a']['input_ids'][i].tolist()
                text_label = (self.ignore_id*torch.ones(len(conditoin_ids[i]),dtype=int)).tolist()+ \
                             [self.ignore_id] + text_ids['input_p1_q_a']['labels'][i].tolist()#[clip][text]
            ## [task token]+[system idx]+[soi][clip pad tokens] [eoi] [sot] [caption tokens]+ [eot]
            temp_ids = [int(self.sptids_dict['<|t2i|>'])] +text_ids['input_ids_system'][i].tolist() + input_ids## task ids任务
            temp_label = [self.ignore_id] +(self.ignore_id*torch.ones(text_ids['input_ids_system'][i].shape[-1],dtype=int)).tolist() + text_label
            # randomly dropout text condition

            # prompting -- [task token][system idx][soi][clip pad tokens] [eoi] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]*N
            temp_label_ids = torch.cat([  # label
                # should we predict text tokens when doing image reconstruction?yes
                torch.tensor(temp_label).to(device),  # system+vq+text,vq已经被ignore掉了
                # self.sptids_dict['<|soi|>'].to(device),#img_start,'<|soi|>
                labels[i * T:i * T + T].reshape(-1),
                # self.sptids_dict['<|eoi|>'].to(device)
                self.sptids_dict['<|pad|>'].to(device),
                text_ids['input_p3_q_a']['labels'][i].to(device),
                self.sptids_dict['<|pad|>'].to(device),
                text_ids['input_B_q_a']['labels'][i].to(device),
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)#pad位置用ignor给替换掉

            temp_ids = torch.cat([#ids
                torch.tensor(temp_ids).to(device),#[task token][soi][clip pad tokens] [eoi] [sot] [text tokens] [eot]
                # self.sptids_dict['<|soi|>'].to(device),
                image_ids[i*T:i*T+T].reshape(-1), #[soi][image tokens] [eoi][soi][image tokens] [eoi][soi][image tokens] [eoi]
                # self.sptids_dict['<|eoi|>'].to(device)
                self.sptids_dict['<|sot|>'].to(device),
                text_ids['input_p3_q_a']['input_ids'][i].to(device),
                self.sptids_dict['<|eot|>'].to(device),
                text_ids['input_B_q_a']['input_ids'][i].to(device),
            ], dim=0)

            if self.max_text_len >= len(temp_ids):
                # temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids#
                temp_masks = [1] * len(temp_ids)# + T*image_ids.shape[-1] + 1)#why add 1 token idx？
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1]
                temp_masks = [1] * (len(temp_ids) + T*image_ids.shape[-1] + 1)  # +2 for two special tokens

            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(torch.tensor(temp_masks).unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def t2i_gen_prompt(self, text_ids, image_ids):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * len(temp_ids)
            else:
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    # language modeling
    def lm_prompt(self, text_ids, max_seq_len, device):

        sequence_ids = []
        attention_masks = []
        label_ids = []
        text_ids = text_ids.tolist()
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.eos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                temp_labels_ids = temp_ids + [self.ignore_id] * (max_seq_len - len(temp_ids))
                temp_ids = temp_ids + [self.pad_id] * (max_seq_len - len(temp_ids))
                temp_masks = [1] * len(temp_ids) + [0] * (max_seq_len - len(temp_ids))
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.tensor(temp_ids)
            temp_masks = torch.tensor(temp_masks)
            temp_labels_ids = torch.tensor(temp_labels_ids)
            mask_labels = temp_labels_ids == self.pad_id
            temp_labels_ids[mask_labels] = self.ignore_id

            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    # mmu modeling
    def mmu_prompt(self, text_ids, image_ids,  device):

        sequence_ids = []
        B = image_ids['context'].shape[0]
        task_token = self.sptids_dict['<|mmu|>'][None, ...].repeat(B, 1).to(device)
        sot_token = self.sptids_dict['<|sot|>'][None, ...].repeat(B, 1).to(device)
        # prompting input-- [task token][system ids] +  [sod] [image tokens] [eod]*N  +  [sot] [text tokens] [eot]
        temp_ids = torch.cat((task_token, text_ids['input_ids_system'].to(device), image_ids['context'], sot_token,
                              text_ids['input_ids'].to(device)), dim=1)
        sequence_ids.append(temp_ids)
        pad_task_token = self.ignore_id * torch.ones_like(task_token)
        pad_sot_token = self.ignore_id * torch.ones_like(sot_token)

        temp_labels = torch.cat((pad_task_token,
                                 self.ignore_id * torch.ones_like(text_ids['input_ids_system'].to(device)),
                                 self.ignore_id * torch.ones_like(image_ids['context']), pad_sot_token,
                                 text_ids['labels'].to(device)), dim=1)
        sequence_ids.append(temp_labels)
        return sequence_ids

    def d2t_prompt(self, text_ids, input_ids_prev, input_ids_next, labels_prev, labels_next):
        """
        :return:
        """
        sequence_ids = []
        device = input_ids_prev['dynamic'].device
        B = input_ids_prev['dynamic'].shape[0]

        # 动态获取token数量参数
        image_tensor = input_ids_prev['context']  # 用于计算image tokens
        dynamic_tensor = input_ids_prev['dynamic']  # 用于计算dynamic tokens
        # 对于d2t，没有depth tensor，使用dynamic tensor作为替代
        depth_tensor = input_ids_prev['dynamic']

        image_tokens, dynamic_tokens, depth_tokens = self._get_token_counts_from_special_tokens(image_tensor, dynamic_tensor, depth_tensor)

        task_token = self.sptids_dict['<|t2d|>'][None, ...].repeat(B, 1).to(device)
        sot_token = self.sptids_dict['<|sot|>'][None, ...].repeat(B, 1).to(device)
        # prompting input-- [task token][system ids] +  [sod] [image tokens] [eod]*N  +  [sot] [text tokens] [eot]
        temp_ids = torch.cat((task_token, text_ids['input_ids_system'].to(device),
                              input_ids_prev['context'][:, :image_tokens], input_ids_prev['dynamic'],
                              input_ids_next['context'][:, :image_tokens], input_ids_next['dynamic'],
                              sot_token, text_ids['input_ids'].to(device)), dim=1)
        sequence_ids.append(temp_ids)
        pad_task_token = self.ignore_id * torch.ones_like(task_token)
        pad_sot_token = self.ignore_id * torch.ones_like(sot_token)
        end_text_token = self.ignore_id * torch.ones_like(sot_token)
        pad_prev_c_token = self.ignore_id * torch.ones_like(input_ids_prev['context'][:, :image_tokens])
        pad_next_c_token = self.ignore_id * torch.ones_like(input_ids_next['context'][:, :image_tokens])
        temp_labels = torch.cat((pad_task_token, self.ignore_id * torch.ones_like(text_ids['input_ids_system'].to(device)),
                                 pad_prev_c_token, labels_prev['dynamic'], pad_next_c_token, labels_next['dynamic'],
                                 pad_sot_token, text_ids['labels'][:,1:].to(device), end_text_token), dim=1)
        # temp_labels = torch.cat((pad_task_token, self.ignore_id * torch.ones_like(text_ids['input_ids_system'].to(device)), pad_sot_token, self.ignore_id * torch.ones_like(text_ids['input_ids'].to(device)), labels['dynamic']), dim=1)
        sequence_ids.append(temp_labels)
        return sequence_ids
    def t2d_prompt(self, text_ids, input_ids_prev, input_ids_next, labels_prev, labels_next):
        """
        :return:
        """
        sequence_ids = []
        device = input_ids_prev['dynamic'].device
        B = input_ids_prev['dynamic'].shape[0]

        # 动态获取token数量参数
        image_tensor = input_ids_prev['context']  # 用于计算image tokens
        dynamic_tensor = input_ids_prev['dynamic']  # 用于计算dynamic tokens
        # 对于t2d，没有depth tensor，使用dynamic tensor作为替代
        depth_tensor = input_ids_prev['dynamic']

        image_tokens, dynamic_tokens, depth_tokens = self._get_token_counts_from_special_tokens(image_tensor, dynamic_tensor, depth_tensor)

        task_token = self.sptids_dict['<|t2d|>'][None, ...].repeat(B, 1).to(device)
        sot_token = self.sptids_dict['<|sot|>'][None, ...].repeat(B, 1).to(device)
        soi_token = self.sptids_dict['<|soi|>'][None, ...].repeat(B, 1).to(device)
        eoi_token = self.sptids_dict['<|eoi|>'][None, ...].repeat(B, 1).to(device)
        # prompting input-- [task token][system ids]  +  [sot] [text tokens] [eot]  +  [sod] [image tokens] [eod]*N
        temp_ids = torch.cat((task_token, text_ids['input_ids_system'].to(device), sot_token, text_ids['input_ids'].to(device),
                              input_ids_prev['context'][:, :image_tokens], input_ids_prev['dynamic'], input_ids_next['context'][:, :image_tokens],
                              input_ids_next['dynamic']), dim=1)

        sequence_ids.append(temp_ids)
        pad_task_token = self.ignore_id * torch.ones_like(task_token)
        pad_sot_token = self.ignore_id * torch.ones_like(sot_token)
        pad_prev_c_token = self.ignore_id * torch.ones_like(input_ids_prev['context'][:, :image_tokens])
        pad_next_c_token = self.ignore_id * torch.ones_like(input_ids_next['context'][:, :image_tokens])
        temp_labels = torch.cat((pad_task_token, self.ignore_id * torch.ones_like(text_ids['input_ids_system'].to(device)), pad_sot_token,
                                 self.ignore_id * torch.ones_like(text_ids['input_ids'].to(device)),
                                 pad_prev_c_token, labels_prev['dynamic'], pad_next_c_token,
                                 labels_next['dynamic']), dim=1)
        sequence_ids.append(temp_labels)
        return sequence_ids
    def t2dgen_prompt(self, text_ids, image_ids, labels, pred_frame):
        """
        :return:
        """
        sequence_ids = []
        device = image_ids['dynamic'].device
        B = image_ids['dynamic'].shape[0]
        task_token = self.sptids_dict['<|t2d|>'][None, ...].repeat(B, 1).to(device)
        sot_token = self.sptids_dict['<|sot|>'][None, ...].repeat(B, 1).to(device)
        # prompting input-- [task token][system ids]  +  [sot] [text tokens] [eot]  +  [sod] [image tokens] [eod]*N
        temp_ids = torch.cat((task_token, text_ids['input_ids_system'].to(device), sot_token, text_ids['input_ids'].to(device), image_ids['dynamic']), dim=1)
        sequence_ids.append(temp_ids)
        pad_task_token = self.ignore_id * torch.ones_like(task_token)
        pad_sot_token = self.ignore_id * torch.ones_like(sot_token)

        temp_labels = torch.cat((pad_task_token, self.ignore_id * torch.ones_like(text_ids['input_ids_system'].to(device)), pad_sot_token, self.ignore_id * torch.ones_like(text_ids['input_ids'].to(device)), labels['dynamic']), dim=1)
        sequence_ids.append(temp_labels)
        return sequence_ids
    def i2v_prompt(self, image_ids, video_ids):
        """
        :param image_ids:
        :param video_ids:
        :return:
        """
        pass
    def nusc_prompt_front(self, text_ids,input_ids_prev, input_ids_next, labels_prev, labels_next, real_len, action_num, ego_status, H_cmd, add_prev_con=True,):
        sequence_ids = []
        device = input_ids_prev['dynamic'].device
        B = input_ids_prev['dynamic'].shape[0]
        ignor_d = 12-real_len
        prev_lengths = ignor_d[:, 0] * 30
        next_lengths = ignor_d[:, 1] * 30
        prev_mask = torch.arange(input_ids_prev['dynamic'].shape[1], device=device)[None, ...].repeat(B,1) < prev_lengths.unsqueeze(1).to(device)
        next_mask = torch.flip(torch.arange(input_ids_next['dynamic'].shape[1], device=device)[None, ...].repeat(B,1),dims=[1]) < next_lengths.unsqueeze(1).to(device)
        # assert real_len.unique() == 12 #debug for filtered dataset
        input_ids_prev['dynamic'][prev_mask] = self.sptids_dict['<|pad|>'].to(device)
        input_ids_next['dynamic'][next_mask] = self.sptids_dict['<|pad|>'].to(device)
        state_num = 1 if ego_status is not None else 0
        cmd_num = 1 if H_cmd is not None else 0
        task_token_mu = self.sptids_dict['<|mmu|>'][None, ...].repeat(B, 1).to(device)
        task_token_d = self.sptids_dict['<|t2d|>'][None, ...].repeat(B, 1).to(device)
        sot_token = self.sptids_dict['<|sot|>'][None, ...].repeat(B, 1).to(device)
        cmd_w_place_holder = self.sptids_dict['<|act|>'][None, ...].repeat(B, cmd_num).to(device)
        ego_status_w_place_holder = self.sptids_dict['<|act|>'][None, ...].repeat(B, state_num).to(device)
        act_w_place_holder = self.sptids_dict['<|act|>'][None, ...].repeat(B, action_num+1).to(device)
        pad_task_token_mu = self.ignore_id * torch.ones_like(task_token_mu)
        pad_task_token_d = self.ignore_id * torch.ones_like(task_token_d)
        # 动态获取token数量参数
        image_tensor = input_ids_prev['context']  # 用于计算image tokens
        dynamic_tensor = input_ids_prev['dynamic']  # 用于计算dynamic tokens
        # 对于nusc，没有depth tensor，使用dynamic tensor作为替代
        depth_tensor = input_ids_prev['dynamic']

        image_tokens, dynamic_tokens, depth_tokens = self._get_token_counts_from_special_tokens(image_tensor, dynamic_tensor, depth_tensor)

        pad_prev_token = self.ignore_id * torch.ones_like(input_ids_prev['dynamic'])
        pad_prevcontent_token = self.ignore_id * torch.ones_like(input_ids_prev['context'][:, :image_tokens])
        pad_content_token = self.ignore_id * torch.ones_like(input_ids_next['context'][:,:image_tokens])
        pad_sot_token = self.ignore_id * torch.ones_like(sot_token)
        pad_cmd_token = self.ignore_id * torch.ones_like(cmd_w_place_holder)
        pad_ego_status_token = self.ignore_id * torch.ones_like(ego_status_w_place_holder)
        pad_action_token = self.ignore_id * torch.ones_like(act_w_place_holder)
        # [task token][system ids] -> [sod] [prev d tokens] [eod]*N -> [sod] [next c tokens][0 frame] [eod] -> unsupervised
        # [sot] ([text QA tokens]) [eot] -> [sod] ([next d tokens]) [eod]*N -> [sot] ([6 queries]) [eot] ->queries(ignore here) ->supervised
        temp_ids = torch.cat((task_token_mu, task_token_d, text_ids['input_ids_system'].to(device), input_ids_prev['context'][:, :image_tokens],
                              input_ids_prev['dynamic'], input_ids_next['context'][:, :image_tokens],cmd_w_place_holder, ego_status_w_place_holder,
                              sot_token, text_ids['input_ids'].to(device), input_ids_next['dynamic'],
                              act_w_place_holder), dim=1)
        temp_labels = torch.cat((pad_task_token_mu, pad_task_token_d, self.ignore_id * torch.ones_like(text_ids['input_ids_system'].to(device)),
                             pad_prevcontent_token, pad_prev_token, pad_content_token,pad_cmd_token, pad_ego_status_token,
                             pad_sot_token, text_ids['labels'].to(device), labels_next['dynamic'], pad_action_token), dim=1)
        sequence_ids.append(temp_ids)
        sequence_ids.append(temp_labels)
        return sequence_ids

    def _extract_depth_segment_with_padding(self, depth_data, frame_idx, depth_tokens_per_frame):
        """
        辅助函数：从depth数据中提取指定帧的segment，包含智能填充
        
        Args:
            depth_data: depth数据tensor
            frame_idx: 当前帧索引
            depth_tokens_per_frame: 每帧depth token数量
            
        Returns:
            depth_segment: 提取并填充后的depth段
        """
        depth_frame_count = depth_data.shape[1] // depth_tokens_per_frame

        if frame_idx < depth_frame_count:
            # 如果depth数据足够，取当前帧的depth数据
            depth_start = frame_idx * depth_tokens_per_frame
            depth_end = (frame_idx + 1) * depth_tokens_per_frame
            if depth_end <= depth_data.shape[1]:
                depth_segment = depth_data[:, depth_start:depth_end, ...]
            else:
                print(f"frame_idx: {frame_idx} 超出depth数据范围")
                # 如果depth数据不足，取剩余的所有token
                depth_segment = depth_data[:, depth_start:, ...]
                # 如果取到的token不足，用最后一个token填充
                if depth_segment.shape[1] < depth_tokens_per_frame:
                    last_token = depth_data[:, -1:, ...]  # 取最后一个token
                    padding_needed = depth_tokens_per_frame - depth_segment.shape[1]
                    padding = last_token.repeat(1, padding_needed)
                    depth_segment = torch.cat([depth_segment, padding], dim=1)
        else:
            print(f"frame_idx: {frame_idx} 超出depth数据范围")
            # 如果depth数据不足，用最后一个depth帧填充
            last_token = depth_data[:, -1:, ...]
            depth_segment = last_token.repeat(1, depth_tokens_per_frame)

        return depth_segment

    def _create_alternating_sequence(self, input_data, depth_data, input_tokens_per_frame, depth_tokens_per_frame):
        """
        辅助函数：创建depth和input数据交替排列的序列
        
        Args:
            input_data: input数据（dynamic或context）
            depth_data: depth数据
            input_tokens_per_frame: input数据每帧的token数量
            depth_tokens_per_frame: depth数据每帧的token数量
            
        Returns:
            alternating_sequence: 交替排列后的序列
        """
        alternating_frames = []

        # 计算input数据的帧数
        input_frame_count = input_data.shape[1] // input_tokens_per_frame

        # 以input数据的帧数为基准进行遍历
        for frame_idx in range(input_frame_count):
            # 添加对应的depth段
            depth_segment = self._extract_depth_segment_with_padding(depth_data, frame_idx, depth_tokens_per_frame)
            alternating_frames.append(depth_segment)

            # 添加对应的input段
            start_idx = frame_idx * input_tokens_per_frame
            end_idx = (frame_idx + 1) * input_tokens_per_frame
            input_segment = input_data[:, start_idx:end_idx, ...]
            alternating_frames.append(input_segment)

        # 将交替帧拼接成新的序列
        alternating_sequence = torch.cat(alternating_frames, dim=1)
        return alternating_sequence

    def _process_single_input_with_depth(self, input_data, depth_data, dynamic_tokens_per_frame,
                                       context_tokens_per_frame, depth_c_tokens, depth_d_tokens):
        """
        辅助函数：处理单个输入数据（context和dynamic）与depth数据的匹配
        
        Args:
            input_data: 输入数据，包含'context'和'dynamic'字段
            depth_data: 对应的depth数据
            dynamic_tokens_per_frame: dynamic部分每帧的token数量
            context_tokens_per_frame: context部分每帧的token数量
            depth_tokens_per_frame: depth数据每帧的token数量
            
        Returns:
            processed_data: 包含原始数据和合并数据的字典
        """
        processed_data = {
            'context': input_data['context'] if 'context' in input_data else None,  # 保留原始context
            'dynamic': input_data['dynamic'] if 'dynamic' in input_data else None,  # 保留原始dynamic
        }

        # 处理dynamic部分的depth匹配
        if 'dynamic' in input_data:
            new_dynamic = self._create_alternating_sequence(
                input_data["dynamic"],
                depth_data["dynamic"],
                dynamic_tokens_per_frame,
                depth_d_tokens,
            )
            processed_data['dynamic_merged'] = new_dynamic

        # 处理context部分的depth匹配
        if 'context' in input_data:
            new_context = self._create_alternating_sequence(
                input_data["context"],
                depth_data["context"],
                context_tokens_per_frame,
                depth_c_tokens,
            )
            processed_data['context_merged'] = new_context

        return processed_data

    def meger_depth_input(self, depth_pre, depth_next, input_ids_prev, input_ids_next,
                         dynamic_tokens_per_frame=30, context_tokens_per_frame=450, depth_c_tokens=30,
                        depth_d_tokens=450):
        """
        将深度信息与现有的帧数据交替组合，形成新的input_ids_prev和input_ids_next
        分别对dynamic和context部分进行depth匹配
        
        Args:
            depth_pre: 深度输入，共20帧，shape: (b, 20, n, hidden) 或 dict containing 'context'
            depth_next: 后一个深度输入，与input_next对齐，shape: (b, 10, n, hidden) 或 dict containing 'context'
            input_ids_prev: 包含context和dynamic的前一个输入列表（2个元素）
            input_ids_next: 包含context和dynamic的后一个输入
            dynamic_tokens_per_frame: dynamic部分每帧的token数量
            context_tokens_per_frame: context部分每帧的token数量
            depth_tokens_per_frame: depth数据每帧的token数量
            
        Returns:
            new_input_ids_prev: 包含原始数据和合并数据的列表，每个元素包含：
                - context: 原始context数据
                - context_merged: 合并后的context数据（depth和context交替）
                - dynamic_original: 原始dynamic数据（未处理）
                - dynamic_merged: 合并后的dynamic数据（depth和dynamic交替）
            new_input_ids_next: 包含原始数据和合并数据的字典，包含：
                - context: 原始context数据
                - context_merged: 合并后的context数据（depth和context交替）
                - dynamic_original: 原始dynamic数据（未处理）
                - dynamic_merged: 合并后的dynamic数据（depth和dynamic交替）
        """
        # 创建新的数据结构，保留原始数据
        new_input_ids_prev = []
        new_input_ids_next = {}

        # 处理depth_pre：从20帧拆分成两个10帧部分，分别匹配input_ids_prev的两个元素
        if depth_pre is not None:
            # 获取深度帧数据（字段是context）
            # if isinstance(depth_pre, dict) and 'context' in depth_pre:
            #     depth_frames = depth_pre['context']
            # else:
            #     depth_frames = depth_pre

            # total_tokens = depth_frames.shape[1]
            # expected_tokens = 20 * depth_tokens_per_frame  # 期望的总token数

            # # 如果总token数不符合预期，但没有提供新的token参数，使用默认行为
            # if total_tokens == 20 and depth_tokens_per_frame == 1:
            #     # 原始的兼容模式
            #     pass
            # elif abs(total_tokens - expected_tokens) > expected_tokens * 0.1:  # 允许10%的误差
            #     print(f"警告：depth_pre总token数为 {total_tokens}，期望约为 {expected_tokens}")
            #     print(f"这可能是由于depth_tokens_per_frame={depth_tokens_per_frame}造成的")
            #     # 继续处理，但不严格检查

            # 处理input_ids_prev[0]
            if len(input_ids_prev) > 0:
                processed_data = self._process_single_input_with_depth(
                    input_ids_prev[0], depth_pre[0],
                    dynamic_tokens_per_frame, context_tokens_per_frame, depth_c_tokens, depth_d_tokens
                )
                new_input_ids_prev.append(processed_data)

            # 处理input_ids_prev[1]
            if len(input_ids_prev) > 1:
                processed_data = self._process_single_input_with_depth(
                    input_ids_prev[1], depth_pre[1],
                    dynamic_tokens_per_frame, context_tokens_per_frame,  depth_c_tokens, depth_d_tokens
                )
                new_input_ids_prev.append(processed_data)

        # 处理depth_next：分别处理context和dynamic部分
        if depth_next is not None:
            # 获取深度帧数据（字段是context）
            # if isinstance(depth_next, dict) and 'context' in depth_next:
            #     depth_next_frames = depth_next['context']
            # else:
            #     depth_next_frames = depth_next

            # 处理input_ids_next：使用统一的处理函数
            new_input_ids_next = self._process_single_input_with_depth(
                input_ids_next,
                depth_next,
                dynamic_tokens_per_frame,
                context_tokens_per_frame,
                depth_c_tokens, 
                depth_d_tokens
            )

        return new_input_ids_prev, new_input_ids_next

    def navsim_prompt(self, text_ids, input_ids_prev, input_ids_next, labels_prev, labels_next, action_num,
                      ego_status,input_depth_ids_prev, labels_depth_prev, input_depth_ids_next, labels_depth_ids_next):
        sequence_ids = []
        device = input_ids_prev[0]['dynamic'].device
        B = input_ids_prev[0]['dynamic'].shape[0]
        state_num = 1 if ego_status is not None else 0
        task_token_mu = self.sptids_dict['<|mmu|>'][None, ...].repeat(B, 1).to(device)
        task_token_d = self.sptids_dict['<|t2d|>'][None, ...].repeat(B, 1).to(device)
        sot_token = self.sptids_dict['<|sot|>'][None, ...].repeat(B, 1).to(device)
        ego_status_w_place_holder = self.sptids_dict['<|act|>'][None, ...].repeat(B, state_num).to(device)
        act_w_place_holder = self.sptids_dict['<|act|>'][None, ...].repeat(B, action_num+1).to(device)
        pad_task_token_mu = self.ignore_id * torch.ones_like(task_token_mu)
        pad_task_token_d = self.ignore_id * torch.ones_like(task_token_d)
        pad_sot_token = self.ignore_id * torch.ones_like(sot_token)
        pad_ego_status_token = self.ignore_id * torch.ones_like(ego_status_w_place_holder)
        pad_action_token = self.ignore_id * torch.ones_like(act_w_place_holder)
        # 动态获取token数量参数
        # 使用input_ids中的context作为image tensor，dynamic作为dynamic tensor，depth tensor作为depth tensor
        image_tensor = input_ids_prev[0]['context']  # 用于计算image tokens
        dynamic_tensor = input_ids_prev[0]['dynamic']  # 用于计算dynamic tokens
        if input_depth_ids_next is not None:
            depth_c_tensor = input_depth_ids_next["context"]
            depth_d_tensor = input_depth_ids_next["dynamic"]
        else:
            depth_c_tensor = None
            depth_d_tensor = None
        image_tokens, dynamic_tokens, depth_c_tokens, depth_d_tokens = self._get_token_counts_from_special_tokens(image_tensor, dynamic_tensor, depth_c_tensor,depth_d_tensor)
        mode = "base"    
        if mode == "base"    :
            merge_half_pre0 =0
            merge_half_pre1 = 0
            merge_half_next = 0
            merged_input_ids_next = {}
            merged_input_ids_prev = {}
            merged_prev_labels = {}
            merged_next_labels = {}
        else:
            # 获取包含深度信息的修改后的input_ids
            merged_input_ids_prev, merged_input_ids_next = self.meger_depth_input(
                input_depth_ids_prev,
                input_depth_ids_next,
                input_ids_prev,
                input_ids_next,
                dynamic_tokens,
                image_tokens,
                depth_c_tokens,
                depth_d_tokens,
            )
            merged_prev_labels, merged_next_labels = self.meger_depth_input(
                labels_depth_prev,
                labels_depth_ids_next,
                labels_prev,
                labels_next,
                dynamic_tokens,
                image_tokens,
                depth_c_tokens,
                depth_d_tokens,
            )
            merge_half_pre0 = merged_input_ids_prev[0]['context_merged'].shape[1]//2
            merge_half_pre1 = merged_input_ids_prev[1]['context_merged'].shape[1]//2
            merge_half_next = merged_input_ids_next['context_merged'].shape[1]//2

        
        if mode == "merged_all":    
            # 使用修改后的input_ids构建序列
            pad_prev1s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[0]['dynamic_merged'])
            pad_prev2s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[1]['dynamic_merged'])
            pad_prevcontent1s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[0]['context_merged'][:, merge_half_pre0:])
            pad_prevcontent2s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[1]['context_merged'][:, merge_half_pre1:])
            pad_content_token = self.ignore_id * torch.ones_like(merged_input_ids_next['context_merged'][:,merge_half_next:])
            lables_next = merged_next_labels['dynamic_merged']
            temp_ids = torch.cat(
                (
                    task_token_mu,
                    task_token_d,
                    text_ids['input_ids_system'].to(device),
                    merged_input_ids_prev[0]['dynamic_merged'],
                    merged_input_ids_prev[1]['context_merged'][:, merge_half_pre1:],
                    merged_input_ids_prev[1]['dynamic_merged'],
                    merged_input_ids_next['context_merged'][:, merge_half_next:],
                    ego_status_w_place_holder,
                    sot_token,
                    text_ids['input_ids'].to(device),
                    sot_token,
                    merged_input_ids_next['dynamic_merged'],
                    act_w_place_holder
                ),
                dim=1
            )
        elif mode == "merged_context_only":
            pad_prev1s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[0]['dynamic'])
            pad_prev2s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[1]['dynamic'])
            pad_prevcontent1s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[0]['context_merged'][:, merge_half_pre0:])
            pad_prevcontent2s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[1]['context_merged'][:, merge_half_pre1:])
            pad_content_token = self.ignore_id * torch.ones_like(merged_input_ids_next['context_merged'][:,merge_half_next:])
            lables_next = merged_next_labels['dynamic_merged']
            # 使用修改后的input_ids构建序列
            temp_ids = torch.cat(
                (
                    task_token_mu,
                    task_token_d,
                    text_ids['input_ids_system'].to(device),
                    merged_input_ids_prev[0]['dynamic'],
                    merged_input_ids_prev[1]['context_merged'][:, merge_half_pre1:],
                    merged_input_ids_prev[1]['dynamic'],
                    merged_input_ids_next['context_merged'][:, merge_half_next:],
                    ego_status_w_place_holder,
                    sot_token,
                    text_ids['input_ids'].to(device),
                    sot_token,
                    merged_input_ids_next['dynamic_merged'],
                    act_w_place_holder
                ),
                dim=1
            )
        elif mode == "merged_dynamic_only":
            pad_prev1s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[0]['dynamic'])
            pad_prev2s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[1]['dynamic'])
            pad_prevcontent1s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[0]['context'][:, image_tokens:])
            pad_prevcontent2s_token = self.ignore_id * torch.ones_like(merged_input_ids_prev[1]['context'][:, image_tokens:])
            pad_content_token = self.ignore_id * torch.ones_like(merged_input_ids_next['context'][:,image_tokens:])
            lables_next = merged_next_labels['dynamic_merged']
            # 使用修改后的input_ids构建序列
            temp_ids = torch.cat(
                (
                    task_token_mu,
                    task_token_d,
                    text_ids['input_ids_system'].to(device),
                    merged_input_ids_prev[0]['dynamic'],
                    merged_input_ids_prev[1]['context'][:, image_tokens:],
                    merged_input_ids_prev[1]['dynamic'],
                    merged_input_ids_next['context'][:, image_tokens:],
                    ego_status_w_place_holder,
                    sot_token,
                    text_ids['input_ids'].to(device),
                    sot_token,
                    merged_input_ids_next['dynamic_merged'],
                    act_w_place_holder
                ),
                dim=1
            )
        else:
            pad_prev1s_token = self.ignore_id * torch.ones_like(input_ids_prev[0]['dynamic'])
            pad_prev2s_token = self.ignore_id * torch.ones_like(input_ids_prev[1]['dynamic'])
            pad_prevcontent1s_token = self.ignore_id * torch.ones_like(input_ids_prev[0]['context'][:, image_tokens:])
            pad_prevcontent2s_token = self.ignore_id * torch.ones_like(input_ids_prev[1]['context'][:, image_tokens:])
            pad_content_token = self.ignore_id * torch.ones_like(input_ids_next['context'][:,image_tokens:])
            lables_next = labels_next['dynamic']
            temp_ids = torch.cat(
                (
                    task_token_mu,
                    task_token_d,
                    text_ids['input_ids_system'].to(device),
                    input_ids_prev[0]['dynamic'],
                    input_ids_prev[1]['context'][:, image_tokens:],
                    input_ids_prev[1]['dynamic'],
                    input_ids_next['context'][:, image_tokens:],
                    ego_status_w_place_holder,
                    sot_token,
                    text_ids['input_ids'].to(device),
                    sot_token,
                    input_ids_next['dynamic'],
                    act_w_place_holder
                ),
                dim=1
            )
        # 处理lables_next中的特殊标记
        lables_next = (
            self.ignore_id * torch.ones_like(pad_action_token)
            if lables_next is None
            else torch.where(
                (lables_next == int(self.sptids_dict["<|sod|>"].item()))
                | (lables_next == int(self.sptids_dict["<|eod|>"].item()))
                | (lables_next == int(self.sptids_dict["<|sop|>"].item()))
                | (lables_next == int(self.sptids_dict["<|eop|>"].item())),
                self.ignore_id,
                lables_next,
            )
        )
        temp_labels = torch.cat(
            (
                pad_task_token_mu,
                pad_task_token_d,
                self.ignore_id
                * torch.ones_like(
                    text_ids["input_ids_system"].to(device)
                ),  # pad_prevcontent1s_token,
                pad_prev1s_token,
                pad_prevcontent2s_token,
                pad_prev2s_token,
                pad_content_token,
                pad_ego_status_token,
                pad_sot_token,
                self.ignore_id * torch.ones_like(text_ids["labels"].to(device)),
                pad_sot_token,
                lables_next,
                pad_action_token,
            ),
            dim=1,
        )
        sequence_ids.append(temp_ids)
        sequence_ids.append(temp_labels)
        return sequence_ids
    
    def navsim_alt_prompt(self, text_ids, input_ids_prev, input_ids_next, labels_prev, labels_next, action_num, ego_status):
        sequence_ids=[]
        device = input_ids_prev[0]['dynamic'].device
        B = input_ids_prev[0]['dynamic'].shape[0]
        frame_len = 30
        act_len = 3
        state_num = 1 if ego_status is not None else 0
        task_token_mu = self.sptids_dict['<|mmu|>'][None, ...].repeat(B, 1).to(device)
        task_token_d = self.sptids_dict['<|t2d|>'][None, ...].repeat(B, 1).to(device)
        sot_token = self.sptids_dict['<|sot|>'][None, ...].repeat(B, 1).to(device)
        ego_status_w_place_holder = self.sptids_dict['<|act|>'][None, ...].repeat(B, state_num).to(device)
        act_w_place_holder = self.sptids_dict['<|act|>'][None, ...].repeat(B, act_len).to(device)

        pad_task_token_mu = self.ignore_id * torch.ones_like(task_token_mu)
        pad_task_token_d = self.ignore_id * torch.ones_like(task_token_d)
        pad_prev1s_token = self.ignore_id * torch.ones_like(input_ids_prev[0]['dynamic'])
        pad_prev2s_token = self.ignore_id * torch.ones_like(input_ids_prev[1]['dynamic'])
        pad_prevcontent1s_token = self.ignore_id * torch.ones_like(input_ids_prev[0]['context'][:, 450:])
        pad_prevcontent2s_token = self.ignore_id * torch.ones_like(input_ids_prev[1]['context'][:, 450:])
        pad_content_token = self.ignore_id * torch.ones_like(input_ids_next['context'][:,450:])
        pad_sot_token = self.ignore_id * torch.ones_like(sot_token)
        pad_ego_status_token = self.ignore_id * torch.ones_like(ego_status_w_place_holder)
        pad_action_token = self.ignore_id * torch.ones_like(act_w_place_holder)
        # construct alternative prediction sequence & labels
        tuple_num = frame_len + act_len # 33
        alt_gen_tokens = torch.ones([B, tuple_num * action_num],dtype=torch.int64).to(device)
        alt_labels = torch.ones([B, tuple_num * action_num],dtype=torch.int64).to(device)
        for alt_i in range(action_num): # 0~7
            alt_gen_tokens[:, alt_i*tuple_num:(alt_i+1)*tuple_num] = torch.cat(
                        [input_ids_next["dynamic"][:, alt_i*frame_len:(alt_i+1)*frame_len],
                        act_w_place_holder], dim=1)
            alt_labels[:, alt_i*tuple_num:(alt_i+1)*tuple_num] = torch.cat(
                        [labels_next["dynamic"][:, alt_i*frame_len:(alt_i+1)*frame_len],
                        pad_action_token], dim=1)

        temp_ids = torch.cat([task_token_mu, task_token_d, text_ids['input_ids_system'].to(device), #input_ids_prev[0]['context'][:, 450:],
                              input_ids_prev[0]['dynamic'],input_ids_prev[1]['context'][:, 450:],
                              input_ids_prev[1]['dynamic'], input_ids_next['context'][:, 450:], ego_status_w_place_holder,
                              sot_token, text_ids['input_ids'].to(device), sot_token, alt_gen_tokens
                              ], dim=1)
        temp_labels = torch.cat((pad_task_token_mu, pad_task_token_d, self.ignore_id * torch.ones_like(text_ids['input_ids_system'].to(device)), #pad_prevcontent1s_token,
                             pad_prev1s_token, pad_prevcontent2s_token, pad_prev2s_token, pad_content_token, pad_ego_status_token,
                             pad_sot_token, self.ignore_id * torch.ones_like(text_ids['labels'].to(device)), pad_sot_token, 
                             alt_labels), dim=1)
        sequence_ids.append(temp_ids)
        sequence_ids.append(temp_labels)

        return sequence_ids
    
    def __call__(self, input, task, padding=True, config=None):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).input【0】：text,input[1]:img
        task (str) : a flag indicates the current task.
        """
        data_dict = {}
        if task == "t2d":# labels require context padding
            assert len(input) == 5, "no enough values for input"
            texts, input_ids_prev, input_ids_next, labels_prev, labels_next = input
            caption = texts['input_caption']
            caption_input = []
            cmd = self.text_tokenizer(caption,
                                           return_tensors="pt",
                                           padding="longest",
                                           max_length=30,
                                           truncation=True).input_ids
            input_ids_system = self.text_tokenizer(SYSTEM_PROMPT,
                                           return_tensors="pt",
                                           padding="longest",
                                           max_length=30,
                                           truncation=True).input_ids
            text_ids = dict(
                input_ids=cmd.to(torch.int64),
                labels=self.ignore_id*torch.ones_like(cmd).to(torch.int64),
                input_ids_system=input_ids_system.repeat(cmd.shape[0],1)
            )
            sequence_ids = self.t2d_prompt(text_ids, input_ids_prev, input_ids_next, labels_prev, labels_next)
        elif task == "t2d_gen":
            assert len(input) == 4, "no enough values for input"
            texts, input_ids, labels, T = input
            caption = texts['input_caption']
            caption_input = []
            for i in caption:
                caption_input.append(self.conversation_tample(i, task))
            caption_sources = preprocess_multimodal(copy.deepcopy([e for e in caption_input])) #remove the </image>
            text_ids = preprocess_v0(caption_sources, self.text_tokenizer, return_system=True)
            sequence_ids = self.t2dgen_prompt(text_ids, input_ids, labels, T)

        elif task == "d2t":
            assert len(input) == 5, "no enough values for input"
            texts, input_ids_prev, input_ids_next, labels_prev, labels_next = input
            caption_sources = [i[0] for i in texts['input_caption']]
            # caption_input = []
            # for i in caption:
            #     caption_input.append(self.conversation_tample(i, task))
            # caption_sources = preprocess_multimodal(copy.deepcopy([e for e in caption_input])) #remove the </image>
            text_ids = preprocess_v0(caption_sources, self.text_tokenizer, return_system=True)
            sequence_ids = self.d2t_prompt(text_ids, input_ids_prev, input_ids_next, labels_prev, labels_next)

        elif task == "mmu":
            texts, input_ids, max_len_lm, device = input
            caption = texts['input_caption']
            L_input_ids = input_ids['context'].shape[-1]
            caption_input = []
            for i in caption:
                caption_input.append(self.conversation_tample(i, task))
            caption_sources = preprocess_multimodal(copy.deepcopy([e for e in caption_input]))
            text_ids = preprocess_v0(caption_sources, self.text_tokenizer, return_system=True, max_len=max_len_lm-L_input_ids-30)# Todo: note that in order to keep same length of all task ,mmu task may not adding eot token for some long samples
            if text_ids['input_ids'].size(-1) < (max_len_lm-L_input_ids-30):
                B = text_ids['input_ids'].size(0)
                pad_len = (max_len_lm-L_input_ids-30)-text_ids['input_ids'].size(-1)
                pad_input = self.sptids_dict['<|pad|>']*torch.ones(B, pad_len, dtype=text_ids['input_ids'].dtype)
                pad_label = self.ignore_id**torch.ones(B, pad_len, dtype=text_ids['labels'].dtype)
                text_ids['input_ids'] = torch.cat((pad_input, text_ids['input_ids']), dim=1)
                text_ids['labels'] = torch.cat((pad_label, text_ids['labels']), dim=1)

            sequence_ids = self.mmu_prompt(text_ids, input_ids, device)

        elif task == "lm":
            text, max_len, device = input
            text_ids = self.text_tokenizer(text,
                                            return_tensors="pt",
                                            padding="longest",
                                            max_length=max_len,
                                            truncation=True).input_ids
            sequence_ids = self.lm_prompt(text_ids, max_len, device)

        elif task == "nusc_add_front":
            assert len(input) == 9, "no enough values for input"
            texts, input_ids_prev, input_ids_next, labels_prev, labels_next, real_len, action_num, ego_status, H_cmd = input
            caption = texts['input_caption']
            if caption != [""] * real_len.shape[0]:
                caption_input = []
                for i in caption:
                    caption_input.append(self.conversation_tample(i, task))
                caption_sources = preprocess_multimodal(
                    copy.deepcopy([e for e in caption_input]))  # remove the </image>
                text_ids = preprocess_v0(caption_sources, self.text_tokenizer, return_system=True)
            else:
                caption_input = []
                for i in range(real_len.shape[0]):  # batch
                    caption_input.append(self.conversation_tample("", task))
                caption_sources = preprocess_multimodal(
                    copy.deepcopy([e for e in caption_input]))  # remove the </image>
                text_ids = preprocess_v0(caption_sources, self.text_tokenizer, return_system=True)
            sequence_ids = self.nusc_prompt_front(text_ids, input_ids_prev, input_ids_next, labels_prev, labels_next,
                                            real_len, action_num, ego_status, H_cmd)

        elif task == "navsim":
            assert len(input) == 11, "no enough values for input"
            texts, input_ids_prev, input_ids_next, labels_prev, labels_next, action_num, ego_status,input_depth_ids_prev, labels_depth_prev, input_depth_ids_next, labels_depth_ids_next = input
            B = ego_status.shape[0]
            caption = texts['input_caption']
            if caption != [""] * B:
                caption_input = []
                for i in caption:
                    caption_input.append(self.conversation_tample(i, task))
                caption_sources = preprocess_multimodal(
                    copy.deepcopy([e for e in caption_input]))  # remove the </image>
                text_ids = preprocess_v0(caption_sources, self.text_tokenizer, return_system=True)
            else:
                caption_input = []
                for i in range(B):  # batch
                    caption_input.append(self.conversation_tample("", task))
                caption_sources = preprocess_multimodal(
                    copy.deepcopy([e for e in caption_input]))  # remove the </image>
                text_ids = preprocess_v0(caption_sources, self.text_tokenizer, return_system=True)
            sequence_ids = self.navsim_prompt(text_ids, input_ids_prev, input_ids_next, labels_prev, labels_next,
                                             action_num, ego_status,input_depth_ids_prev, labels_depth_prev, input_depth_ids_next, labels_depth_ids_next)

        elif task == "navsim_alt":
            assert len(input) == 11, "no enough values for input"
            texts, input_ids_prev, input_ids_next, labels_prev, labels_next, action_num, ego_status,input_depth_ids_prev, labels_depth_prev, input_depth_ids_next, labels_depth_ids_next = input
            B = ego_status.shape[0]
            caption = texts['input_caption']
            if caption != [""] * B:
                caption_input = []
                for i in caption:
                    caption_input.append(self.conversation_tample(i, task))
                caption_sources = preprocess_multimodal(
                    copy.deepcopy([e for e in caption_input]))  # remove the </image>
                text_ids = preprocess_v0(caption_sources, self.text_tokenizer, return_system=True)
            else:
                caption_input = []
                for i in range(B):  # batch
                    caption_input.append(self.conversation_tample("", task))
                caption_sources = preprocess_multimodal(
                    copy.deepcopy([e for e in caption_input]))  # remove the </image>
                text_ids = preprocess_v0(caption_sources, self.text_tokenizer, return_system=True)
            sequence_ids = self.navsim_alt_prompt(text_ids, input_ids_prev, input_ids_next, labels_prev, labels_next,
                                             action_num, ego_status)

        else:
            raise NotImplementedError

        return sequence_ids
# @profile

def create_attention_mask_for_t2d(sequence,
                                  T, clip_len,
                                  pad_id=128256,
                                  soi_id=128257,
                                  eoi_id=128258,
                                  sot_id=128259,
                                  sod_id=128260,
                                  eod_id=128261,
                                  rm_pad_in_image=False,
                                  return_inverse_mask=False):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape
    is_start_image = sequence == sod_id
    is_end_image = sequence == eod_id

    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)#
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image #True if

    is_text = ~(in_image_segment)#action also included

    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)

    mask_text = is_text[:, :, None] * causal_mask[None, :, :]#

    B = mask_text.shape[0]
    if rm_pad_in_image:# remove pad token
        for i in range(B):#per_batch
            sod_img = torch.where(sequence[i] == sod_id)[0]#
            eod_img = torch.where(sequence[i] == eod_id)[0]
            pad_end_idx = torch.where(sequence[i] == pad_id)[0]#
            for sod_mask, eod_mask in zip(sod_img, eod_img):#img part
                assert sod_mask.item() < eod_mask.item()
                mask_text[i][sod_mask:eod_mask+1, :eod_mask+1] = 1 #img part
            if len(pad_end_idx) != 0:    # mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
                mask_text[i][:, pad_end_idx] = 0#
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1) #,clip_idx, action_idx, Img_idx, text_idx
    else:
        return mask_text.unsqueeze(1) #, clip_idx, action_idx, Img_idx
def create_attention_mask_for_lm(sequence,
                                  pad_id=128256,
                                  sot_id=128259,
                                  sod_id=128260,
                                  eod_id=128261,
                                  rm_pad_in_image=False,
                                  return_inverse_mask=False):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    is_start_image = sequence == sod_id

    is_end_image = sequence == eod_id
    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)#
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image #True if

    is_text = ~(in_image_segment)#action also included
    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)
    mask_text = is_text[:, :, None] * causal_mask[None, :, :]#
    # is_text_image = (is_text | in_image_segment ) & ~is_padding
    B = mask_text.shape[0]
    if rm_pad_in_image:#移除pad token
        for i in range(B):#per_batch
            pad_end_idx = torch.where(sequence[i] == pad_id)[0]
            if len(pad_end_idx) != 0:
                mask_text[i][:, pad_end_idx] = 0
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)#
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1) #,clip_idx, action_idx, Img_idx, text_idx
    else:
        return mask_text.unsqueeze(1) #, clip_idx, action_idx, Img_idx
def create_attention_mask_predict_next(sequence,T,clip_len, pad_id=128256, soi_id=128257, eoi_id=128258, sot_id=128259, rm_pad_in_image=False,
                                       return_inverse_mask=False):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # Masks to identify different types of tokens
    is_padding = sequence == pad_id

    is_start_image = sequence == soi_id

    is_end_image = sequence == eoi_id
    is_start_text = sequence == sot_id
    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)#
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image #True if

    is_text = ~(in_image_segment)#action also included

    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)

    mask_text = is_text[:, :, None] * causal_mask[None, :, :]#

    is_text_image = is_text | in_image_segment
    clip_idx = []
    action_idx = []
    Img_idx = []
    text_idx = []
    mask_text_image_bi = is_text_image[:, :, None] * is_text_image[:, None, :]
    if rm_pad_in_image:#移除pad token
        for i in range(mask_text.shape[0]):#per_batch
            sid_img = torch.where(sequence[i] == soi_id)[0]
            end_image = torch.where(sequence[i] == eoi_id)[0]
            text_ids = torch.where(sequence[i] == sot_id)[0]
            clip_idx.append([sid_img[0].item(),end_image[0].item()])#clip img position
            action_idx.append([end_image[1:], sid_img[2:]])
            Img_idx.append([sid_img,end_image])
            text_idx.append([text_ids])
            pad_end_idx = torch.where(sequence[i] == pad_id)

            for soi_mask, eoi_mask in zip(sid_img, end_image):#img part
                assert soi_mask.item()<eoi_mask.item()
                mask_text[i][soi_mask:eoi_mask+1, :eoi_mask+1] = 1 #img part
            if len(pad_end_idx) != 0:    # mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
                mask_text[i][:, pad_end_idx[0]] = 0#
            # id_padding = torch.where(is_padding[i] == True)
            # mask_text_image_bi[i][sid_img[i]:, id_padding[0]] = 0
    # mask_text[in_image_segment] = mask_text_image_bi[in_image_segment]
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1),clip_idx, action_idx, Img_idx, text_idx
    else:
        return mask_text.unsqueeze(1), clip_idx, action_idx, Img_idx
def create_attention_mask_for_t2d(sequence,
                                  T, clip_len,
                                  pad_id=128256,
                                  soi_id=128257,
                                  eoi_id=128258,
                                  sot_id=128259,
                                  sod_id=128260,
                                  eod_id=128261,
                                  rm_pad_in_image=False,
                                  return_inverse_mask=False):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape
    is_start_image = sequence == sod_id
    is_end_image = sequence == eod_id

    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)#
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image #True if

    is_text = ~(in_image_segment)#action also included

    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)

    mask_text = is_text[:, :, None] * causal_mask[None, :, :]#

    B = mask_text.shape[0]
    if rm_pad_in_image:#移除pad token
        for i in range(B):#per_batch
            sod_img_c = torch.where(sequence[i] == soi_id)[0]#
            eod_img_c = torch.where(sequence[i] == eoi_id)[0]
            sod_img = torch.where(sequence[i] == sod_id)[0]#
            eod_img = torch.where(sequence[i] == eod_id)[0]
            pad_end_idx = torch.where(sequence[i] == pad_id)[0]#
            for sod_mask, eod_mask in zip(sod_img, eod_img):#img part
                assert sod_mask.item() < eod_mask.item()
                mask_text[i][sod_mask:eod_mask+1, :eod_mask+1] = 1 #img part

            for sod_mask, eod_mask in zip(sod_img_c, eod_img_c):# context img part
                assert sod_mask.item() < eod_mask.item()
                mask_text[i][sod_mask:eod_mask+1, :eod_mask+1] = 1

            if len(pad_end_idx) != 0:    # mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
                mask_text[i][:, pad_end_idx] = 0# 先mask pad token
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)#
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1) #,clip_idx, action_idx, Img_idx, text_idx
    else:
        return mask_text.unsqueeze(1) #, clip_idx, action_idx, Img_idx
def create_attention_mask_for_nusc(sequence,
                                  pad_id=128256,
                                  soi_id=128257,
                                  eoi_id=128258,
                                  sod_id=128260,
                                  eod_id=128261,
                                  sop_id=128259,
                                  eop_id=128262,
                                  rm_pad_in_image=False,
                                  return_inverse_mask=False,
                                  mask_future_ratio=None):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape
    is_start_image_c = sequence == soi_id
    is_end_image_c = sequence == eoi_id
    is_start_image_d = sequence == sod_id
    is_end_image_d = sequence == eod_id
    is_sop = sequence == sop_id  # 新增 sop 标记识别
    is_eop = sequence == eop_id  # 新增 eop 标记识别
    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start_c = torch.cumsum(is_start_image_c, dim=1)#
    cumulative_end_c = torch.cumsum(is_end_image_c, dim=1)
    cumulative_start_d = torch.cumsum(is_start_image_d, dim=1)#
    cumulative_end_d = torch.cumsum(is_end_image_d, dim=1)

    in_image_segment = (cumulative_start_d > cumulative_end_d)|(cumulative_start_c > cumulative_end_c) | is_start_image_c | is_end_image_c | is_start_image_d | is_end_image_d | is_sop | is_eop #True if

    is_text = ~(in_image_segment)#action also included

    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)

    mask_text = is_text[:, :, None] * causal_mask[None, :, :]#
    # is_mask_future = mask_future_ratio>0.5

    B = mask_text.shape[0]
    if rm_pad_in_image:
        for i in range(B):#per_batch
            sod_img_c = torch.where(sequence[i] == soi_id)[0]
            eod_img_c = torch.where(sequence[i] == eoi_id)[0]
            sod_img_d = torch.where(sequence[i] == sod_id)[0]
            eod_img_d = torch.where(sequence[i] == eod_id)[0]
            sop_positions = torch.where(sequence[i] == sop_id)[0]  # 新增 sop 位置
            eop_positions = torch.where(sequence[i] == eop_id)[0]  # 新增 eop 位置
            pad_end_idx = torch.where(sequence[i] == pad_id)[0]#
            for sod_mask, eod_mask in zip(sod_img_d, eod_img_d):# dynamic img part
                assert sod_mask.item() < eod_mask.item()
                mask_text[i][sod_mask:eod_mask+1, :eod_mask+1] = 1
            for sod_mask, eod_mask in zip(sod_img_c, eod_img_c):# context img part
                assert sod_mask.item() < eod_mask.item()
                mask_text[i][sod_mask:eod_mask+1, :eod_mask+1] = 1
                
            for  sop_pos, eop_pos in zip(sop_positions, eop_positions):
                assert sop_pos.item() < eop_pos.item()
                mask_text[i][sop_pos:eop_pos+1, :eop_pos+1] = 1
                
            if mask_future_ratio is not None:
                start_pad_future = sod_img_d[sod_img_d > eod_img_c[-1]][0]
                end_pad_future = eod_img_d[-1]
                if mask_future_ratio[0]>0.5:
                    mask_text[i][-7:, start_pad_future:end_pad_future] = 0
            if len(pad_end_idx) != 0:# mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
                mask_text[i][:, pad_end_idx] = 0# 先mask pad token
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        if is_int_dtype(sequence.dtype):
            inverted_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
            )
        else:
            inverted_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(sequence.dtype).min
            )
        return inverted_mask.unsqueeze(1)
    else:
        return mask_text.unsqueeze(1)

def create_attention_mask_for_mmu(sequence,
                                  pad_id=128256,
                                  soi_id=128257,
                                  eoi_id=128258,
                                  sot_id=128259,
                                  sod_id=128260,
                                  eod_id=128261,
                                  rm_pad_in_image=False,
                                  return_inverse_mask=False):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # Masks to identify different types of tokens
    is_start_image = sequence == soi_id

    is_end_image = sequence == eoi_id
    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image #True if

    is_text = ~(in_image_segment)#action also included

    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)

    mask_text = is_text[:, :, None] * causal_mask[None, :, :]
    B = mask_text.shape[0]
    if rm_pad_in_image:
        for i in range(B):#per_batch
            soi_img = torch.where(sequence[i] == soi_id)[0]#
            eoi_img = torch.where(sequence[i] == eoi_id)[0]
            text_ids = torch.where(sequence[i] == sot_id)[0]
            pad_end_idx = torch.where(sequence[i] == pad_id)[0]#
            for soi_mask, eoi_mask in zip(soi_img, eoi_img):#img part
                assert soi_mask.item() < eoi_mask.item()
                mask_text[i][soi_mask:eoi_mask+1, :eoi_mask+1] = 1 #img part
            if len(pad_end_idx) != 0:    # mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
                mask_text[i][:, pad_end_idx] = 0# mask pad token
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1) #,clip_idx, action_idx, Img_idx, text_idx
    else:
        return mask_text.unsqueeze(1) #, clip_idx, action_idx, Img_idx

def create_attention_mask_for_mmu_vit(
        sequence,
        return_inverse_mask=True,
        system_prompt_len=0
):
    N, L, H = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(sequence.device)
    index = 1 + system_prompt_len + 1 + 576
    # TODO: PART OF SYSTEM PROMPT SHOULD BE CAUSAL ALSO
    causal_mask[:, :, :, :index] = 1
    if return_inverse_mask:
        inverted_mask = 1.0 - causal_mask.type(torch.int64)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(torch.int64).min
        )
        return inverted_mask
    else:
        return causal_mask

def is_int_dtype(dtype: torch.dtype) -> bool:
    return dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )

if __name__ == '__main__':
    pass
