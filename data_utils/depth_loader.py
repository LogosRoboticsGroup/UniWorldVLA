from typing import Union, Optional, TYPE_CHECKING
import os
import json
import pickle
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from matplotlib import cm
from DA3.Basic_usage import DA3Main
from data_utils.cache_utils import get_depth_cache_path, get_token_hash

if TYPE_CHECKING:
    from DA3.src.depth_anything_3.specs import Prediction


# 动态导入以避免循环依赖
def get_prediction_class():
    try:
        from DA3.src.depth_anything_3.specs import Prediction

        return Prediction
    except ImportError:
        return None


class Depthloader():
    def __init__(
        self,
        device=None,
        args=None,
        data_loader=None,
    ):
        super().__init__()
        self.args = args
        self.device = device
        # 深度数据存储配置
        self.depth_cache_dir = getattr(args, "cache_dir", "./depth_cache")
        self.use_cached_depth = getattr(args, "use_cached_depth", False)
        self.save_depth_cache = getattr(args, "use_cached_depth", False)

        # DataLoader引用，用于复用加载数据逻辑
        self.data_loader = data_loader

        # 创建缓存目录
        os.makedirs(self.depth_cache_dir, exist_ok=True)

    def _load_depth_data(
        self,
        token_id: str,
        is_prev: bool = True,
        validation_fields: list | None = None,
    ):
        """
        从文件加载深度数据 - 返回统一的字典格式

        Args:
            token_id: token标识符
            is_prev: 是否为prev数据
            validation_fields: 用于验证的字段列表

        Returns:
            统一的字段字典格式 {'depth': ..., 'extrinsics': ..., 'conf': ..., 'sky': ...}
        """
        # 强制使用DataLoader的加载逻辑
        if not self.data_loader:
            raise ValueError(
                "Depth Encoder必须通过DataLoader进行数据加载，请确保传入data_loader参数"
            )

        if not hasattr(self.data_loader, "_load_depth_data_from_cache"):
            raise AttributeError(
                "DataLoader缺少_load_depth_data_from_cache方法，无法复用加载逻辑"
            )

        # 直接使用DataLoader加载缓存数据，返回统一字典格式
        result_dict = self.data_loader._load_depth_data_from_cache(
            token_id, is_prev=is_prev, validation_fields=validation_fields
        )

        # 递归将字典中的所有tensor转移到self.device
        def _move_to_device(data):
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, dict):
                return {k: _move_to_device(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return type(data)(_move_to_device(v) for v in data)
            return data

        return _move_to_device(result_dict)

    def _struct_to_dict(self, struct_data):
        """
        将结构体转换为字典格式

        Args:
            struct_data: 结构体对象（如Prediction）

        Returns:
            dict: 转换后的字典
        """
        if not hasattr(struct_data, "__dict__"):
            return struct_data

        result_dict = {}
        for key, value in struct_data.__dict__.items():
            if value is None:
                result_dict[key] = None
            elif isinstance(value, torch.Tensor):
                result_dict[key] = value
            elif isinstance(value, (list, tuple)):
                result_dict[key] = list(value)
            elif hasattr(value, "__dict__"):
                # 嵌套结构体，递归转换
                result_dict[key] = self._struct_to_dict(value)
            else:
                result_dict[key] = value

        return result_dict

    def _dict_to_struct(self, dict_data, target_struct_class=None):
        """
        将字典转换为结构体格式

        Args:
            dict_data: 字典数据
            target_struct_class: 目标结构体类（可选）

        Returns:
            字典数据（暂时不尝试构造结构体，避免复杂的构造函数问题）
        """
        # 暂时直接返回字典，避免复杂的构造函数问题
        # 在需要时可以通过其他方式转换为结构体
        return dict_data

    def _normalize_to_dict(self, data):
        """
        将任意格式（结构体/字典/张量）统一转换为字典格式

        Args:
            data: 输入数据（结构体/字典/张量）

        Returns:
            字典格式数据
        """
        if hasattr(data, "__dict__") and hasattr(data, "depth"):
            # 结构体转字典
            return self._struct_to_dict(data)
        elif isinstance(data, dict):
            # 已经是字典
            return data
        elif isinstance(data, torch.Tensor):
            # 张量转换为字典
            return {"depth": data}
        else:
            # 其他类型，包装成字典
            return {"data": data}

    def _prepare_data_to_save(self, token_id: str, depth_data):
        """准备要保存的数据结构"""
        # data_dict = self._normalize_to_dict(depth_data)
        data_to_save = {
            "token_id": token_id,
            "prediction_data": {},
            "metadata": {"object_type": "dict", "has_all_fields": True},
        }

        for key, value in depth_data.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    data_to_save["prediction_data"][key] = value.cpu()
                elif isinstance(value, np.ndarray):
                    data_to_save["prediction_data"][key] = value
                else:
                    data_to_save["prediction_data"][key] = value
        return data_to_save

    def _serialize_prediction(self, prediction_obj):
        """序列化Prediction对象为可保存的字典格式"""
        serialized = {
            "token_id": None,  # 将在调用处设置
            "prediction_data": {},
            "metadata": {"object_type": "Prediction", "has_all_fields": True},
        }

        # 保存Prediction对象的所有字段
        # fields_to_save = [
        #     'depth', 'is_metric', 'sky', 'conf', 'extrinsics',
        #     'intrinsics', 'processed_images', 'scale_factor',
        #     'ray', 'ray_conf', 'aux'
        # ]
        fields_to_save = ["depth", "extrinsics", "intrinsics"]

        for field in fields_to_save:
            if hasattr(prediction_obj, field):
                value = getattr(prediction_obj, field)
                if value is not None:
                    # 处理不同类型的值
                    if isinstance(value, torch.Tensor):
                        serialized["prediction_data"][field] = value.cpu()
                    elif isinstance(value, np.ndarray):
                        serialized["prediction_data"][field] = value
                    elif hasattr(value, "__dict__"):
                        # 对于复杂对象如Gaussians，尝试序列化其属性
                        try:
                            # 如果是torch.Tensor类型，移动到CPU
                            if "tensor" in str(type(value)).lower():
                                value_cpu = value.cpu()
                                serialized["prediction_data"][field] = value_cpu
                            else:
                                # 对于其他复杂对象，尝试直接保存
                                serialized["prediction_data"][field] = value
                        except Exception as e:
                            print(f"警告: 无法序列化字段 {field}, 跳过. 错误: {e}")
                            serialized["prediction_data"][field] = None
                    else:
                        serialized["prediction_data"][field] = value
                else:
                    serialized["prediction_data"][field] = None

        return serialized

    def _update_index_file(self, token_id: str, token_hash: str):
        """更新深度数据索引文件"""
        index_file = os.path.join(self.depth_cache_dir, "depth_index.json")

        # 读取现有索引
        if os.path.exists(index_file):
            try:
                with open(index_file, "r") as f:
                    try:
                        index_data = json.load(f)
                    except json.JSONDecodeError:
                        print(
                            f"Warning: Invalid JSON in {index_file}, recreating index"
                        )
                        index_data = {}
            except Exception as e:
                print(f"Error loading {index_file}: {str(e)}")
                index_data = {}
        else:
            index_data = {}

        # 更新索引
        index_data[token_id] = {
            "hash": token_hash,
            "timestamp": str(torch.tensor(0).item()),  # 简单的时间戳
        }

        # 保存索引，确保目录存在
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        try:
            with open(index_file, "w") as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            print(f"Error saving index file {index_file}: {str(e)}")

    @torch.no_grad()
    def _is_valid_cache_data(self, cache_data):
        """检查缓存数据是否有效"""
        return (
            cache_data is not None
            and cache_data != {}
            and not isinstance(cache_data, str)
        )

    def _merge_dictlist_2_dictone(self, depth_list):
        """
        合并字典列表为单个字典，要求所有字典的value都是tensor
        
        Args:
            depth_list: 字典列表，每个字典结构相同且value都是tensor
            
        Returns:
            合并后的字典，每个value是沿batch维度(stack)合并的tensor
        """
        if not depth_list or not isinstance(depth_list, list):
            return None

        first_item = depth_list[0]
        if not isinstance(first_item, dict):
            return None

        # 验证所有字典结构相同且value都是tensor
        keys = set(first_item.keys())
        for item in depth_list:
            if not isinstance(item, dict) or set(item.keys()) != keys:
                return None
            for v in item.values():
                if not isinstance(v, torch.Tensor):
                    return None

        # 按字段合并tensor
        merged_dict = {}
        for key in keys:
            tensors = [item[key] for item in depth_list]
            merged_dict[key] = torch.stack(tensors, dim=0)

        return merged_dict
    def _get_single_result_from_batch_list(self, batch_list, batch_idx: int):
        """
        从列表中按索引返回单个样本

        Args:
            batch_list: 已拆解的批量数据列表
            batch_idx: 要提取的样本索引

        Returns:
            单个样本的数据
        """
        if isinstance(batch_list, list) and batch_idx < len(batch_list):
            return batch_list[batch_idx]
        else:
            raise IndexError(
                f"batch_idx {batch_idx} 超出列表范围 {len(batch_list) if hasattr(batch_list, '__len__') else 'unknown'}"
            )

    def _extract_batch_to_list(self, batch_data):
        """
        将结构体中的多batch数据拆解为单batch数据，并组成list

        Args:
            batch_data: 结构体数据，包含N个样本

        Returns:
            list: 包含N个单个样本字典的列表
        """
        # if not hasattr(batch_data, "__dict__"):
        #     # 非结构体格式，直接返回
        #     return [batch_data]

        # 从结构体的tensor字段获取batch size
        batch_size = batch_data.values[0].shape[0]

        # 生成单个样本的列表
        single_results = []
        for i in range(batch_size):
            # 结构体到字典的转化：提取第i个样本的所有字段
            single_result = {}
            for key, value in batch_data.__dict__.items():
                if value is None:
                    single_result[key] = None
                elif isinstance(value, torch.Tensor):
                    single_result[key] = value[i]  # 按batch维度提取
                else:
                    single_result[key] = None  # 非tensor字段直接复制

            single_results.append(single_result)

        return single_results

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
                    raise ValueError(
                        f"{position}: 字典中没有找到指定的字段 '{field}'，可用字段: {list(depth_input.keys())}"
                    )
            else:
                # 未指定字段，使用第一个可用的张量字段
                available_fields = [
                    k
                    for k, v in depth_input.items()
                    if isinstance(v, torch.Tensor) and v is not None
                ]
                if available_fields:
                    field_used = available_fields[0]
                    print(
                        f"  {position}: 未指定字段，使用第一个可用字段 '{field_used}'"
                    )
                    return depth_input[field_used]
                else:
                    raise ValueError(
                        f"{position}: 字典中没有可用的张量字段，可用字段: {list(depth_input.keys())}"
                    )

        elif (
            hasattr(depth_input, field)
            if field is not None
            else hasattr(depth_input, "depth")
        ):
            # 对象格式，有指定字段属性或默认depth属性
            attr_name = field if field is not None else "depth"
            print(f"  {position}: 从对象的 '{attr_name}' 属性提取数据")
            return getattr(depth_input, attr_name)

        elif isinstance(depth_input, torch.Tensor):
            # 直接是张量格式
            print(f"  {position}: 直接使用张量数据")
            return depth_input

        else:
            attr_name = field if field is not None else "depth"
            raise TypeError(
                f"{position}: 不支持的输入类型 {type(depth_input)}，期望dict、有'{attr_name}'属性的对象或torch.Tensor"
            )

    def save_depth_todisk(self, token_id, prev_depth_img):
        # 保存前帧数据
        prev_data = self._prepare_data_to_save(token_id, prev_depth_img)
        prev_path = get_depth_cache_path(self.depth_cache_dir, token_id,use_subdirs=False, is_prev=True)
        if not os.path.exists(prev_path):
            # print(f"存在这个id ===》不保存  {token_id}")
            torch.save(prev_data, prev_path)
            return True
        return False

    def get_cache_stats(self):
        """获取缓存统计信息"""
        index_file = os.path.join(self.depth_cache_dir, "depth_index.json")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index_data = json.load(f)
            return {
                "cached_tokens": len(index_data),
                "cache_dir": self.depth_cache_dir,
                "use_cached_depth": self.use_cached_depth,
                "save_depth_cache": self.save_depth_cache,
            }
        else:
            return {
                "cached_tokens": 0,
                "cache_dir": self.depth_cache_dir,
                "use_cached_depth": self.use_cached_depth,
                "save_depth_cache": self.save_depth_cache,
            }

    def clear_cache(self):
        """清除所有缓存数据"""
        import shutil

        if os.path.exists(self.depth_cache_dir):
            shutil.rmtree(self.depth_cache_dir)
            os.makedirs(self.depth_cache_dir, exist_ok=True)
            print(f"已清除深度数据缓存: {self.depth_cache_dir}")
        else:
            print("缓存目录不存在，无需清除")

    def _check_one_cache(self, token):
        """检查next方向缓存"""
        cached_data = self._load_depth_data(token, is_prev=False)
        if self._is_valid_cache_data(cached_data):
            print(f"✅ next缓存命中: token='{token}'")
            return cached_data
        return None

    def _batch_process_tokens(
        self, token_pre, token_next, prev_depth_input, next_depth_input
    ):
        """
        批量处理token列表，优化推理次数

        核心优化策略：
        1. 分离缓存命中和需要推理的数据
        2. 批量推理所有需要推理的数据（只调用一次深度推理）
        3. 按原始顺序重组结果，保持数据一致性

        Args:
            token_list: token列表 (例如: ['token_0', 'token_1', 'token_2'])
            prev_depth_input: 前帧深度输入张量 (形状: [batch_size, ...])
            next_depth_input: 后帧深度输入张量 (形状: [batch_size, ...])

        Returns:
            prev_depth_list, next_depth_list: 按原始顺序排列的深度数据列表

        Example:
            假设有8个token，其中3个缓存命中，5个需要推理：
            输入: ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7']
            缓存: [t0, t2, t5] (indices: [0, 2, 5])
            推理: [t1, t3, t4, t6, t7] (indices: [1, 3, 4, 6, 7])
            结果: 5个样本一次批量推理，而不是5次单独推理
        """
        return_list = []
        for token_list, depth_input in [
            (token_pre, prev_depth_input),
            (token_next, next_depth_input),
        ]:

            cached_result = {"indices": [], "load_data": []}
            infer_result = {"indices": [], "tokens": []}
            cached_result, infer_result = self._separate_cached_tokens(
                token_list, cached_result, infer_result
            )
            inferred_data = self._batch_infer_tokens(infer_result, depth_input)
            # 步骤3: 重组结果并返回
            prev_depth_list = self._recombine_single_direction(
                token_list,
                cached_result["indices"],
                cached_result["load_data"],
                infer_result["indices"],
                inferred_data,
                is_prev=True,
            )
            return_list.append(prev_depth_list)
            print(
                f"✨ 批量处理完成: {len(cached_result['indices'])}缓存, {len(infer_result['tokens'])}推理"
            )
        return return_list[0], return_list[1]

    def _separate_cached_tokens(self, token_list, cached_result, infer_result):
        """
        分离缓存命中和需要推理的token

        Args:
            token_list: 输入token列表

        Returns:
            tuple: (cached_result, infer_result)
            cached_result: 包含缓存数据的字典
            infer_result: 包含需要推理的数据的字典
        """
        for i, token in enumerate(token_list):
            if isinstance(token, list):
                for token_ in token:
                    cached_prev = self._check_one_cache(token)
                    if cached_prev:
                        cached_result["indices"].append(i)
                        cached_result["load_data"].append(cached_prev)
                    else:
                        infer_result["indices"].append(i)
                        infer_result["tokens"].append(token)
            else:
                cached_prev = self._check_one_cache(token)
                if cached_prev:
                    cached_result["indices"].append(i)
                    cached_result["load_data"].append(cached_prev)
                else:
                    infer_result["indices"].append(i)
                    infer_result["tokens"].append(token)

        return cached_result, infer_result

    def _batch_infer_tokens(self, infer_result, prev_depth_input):
        """
        批量推理需要计算的token
        Args:
            infer_result: 需要推理的数据字典
            prev_depth_input: 前帧深度输入张量
            next_depth_input: 后帧深度输入张量
        Returns:
            tuple: (prev_data, next_data) 推理结果
        """
        if not infer_result["tokens"]:
            return None, None

        infer_indices = infer_result["indices"]
        input_data_pre = self.de._depth_encoder(prev_depth_input[infer_indices])
        prev_data = self._extract_batch_to_list(input_data_pre)

        print("💾 保存推理结果...")
        for i, token in enumerate(infer_result["tokens"]):
            self.save_depth_todisk(
                token, self._get_single_result_from_batch_list(prev_data, i)
            )

        return prev_data

    def _recombine_single_direction(
        self,
        token_list,
        cached_indices,
        cached_data,
        infer_indices,
        inferred_data,
        is_prev,
    ):
        """
        重组单方向结果数据
        Args:
            token_list: 原始token列表
            cached_indices: 缓存索引
            cached_data: 缓存数据
            infer_indices: 推理索引
            inferred_data: 推理数据
            is_prev: 是否前帧数据
        Returns:
            按原始顺序排列的结果列表
        """
        index_map = {}
        direction = "prev" if is_prev else "next"

        # 填充缓存数据
        for idx, data in zip(cached_indices, cached_data):
            index_map[idx] = data
        print(f"  填充{direction}缓存数据: {len(cached_indices)}个样本")

        # 填充推理数据
        if inferred_data:
            items = [
                self._get_single_result_from_batch_list(inferred_data, i)
                for i in range(len(infer_indices))
            ]
            for idx, data in zip(infer_indices, items):
                index_map[idx] = data
            print(f"  填充{direction}推理数据: {len(infer_indices)}个样本")

        # 构建结果列表
        try:
            return [index_map[i] for i in range(len(token_list))]
        except KeyError as e:
            raise ValueError(f"数据完整性错误：{direction}方向索引 {e} 缺少数据")

    def _save_batch_results(self, token_list, prev_depth,key_list=["depth"]):
        """保存batch结果的辅助方法"""
        if not token_list:
            return
        # 张量格式 - 按维度分割保存
        depth_data = self._struct_to_dict(prev_depth)
        try:
            # token_hash = get_token_hash(token_id)
            for i in range(len(token_list)):
                for j in range(len(token_list[i])):
                    cur = {}
                    for key in depth_data.keys():
                        if key in key_list:
                            cur[key] = depth_data[key][i : i + 1, j : j + 1]
                    if cur != {}:
                        if self.save_depth_todisk(token_list[i][j], cur):
                            print(f"Depth Encoder: 成功保存深度数据, token_id: {token_list[i][j]}")
                    else:
                        print(f"Depth Encoder: 保存深度数据失败,nodata token_id: {token_list[i][j]}")
        except Exception as e:
            print(f"Depth Encoder: 保存深度数据失败, token_id: {token_list}, error: {e}")

    def process(
        self,
        token_id,
        prev_img_depth_tokenid,
        next_img__depth_tokenid,
        prev_depth_img_input,
        next_depth_img_input,
        cached_prev_depth,
        cached_next_depth,
    ):
        """处理单帧输入数据"""
        if cached_prev_depth is not None and cached_next_depth is not None:
            return cached_prev_depth, cached_next_depth

        elif  self.use_cached_depth:
            # if isinstance(token_id, list) and len(token_id) == batch_size:
            prev_depth_list, next_depth_list = self._batch_process_tokens(
                token_id, token_id, prev_depth_img_input, next_depth_img_input
            )
            return self._merge_dictlist_2_dictone(
                    prev_depth_list
                ), self._merge_dictlist_2_dictone(next_depth_list)

        else:
            print("Depth Encoder: 动态推理模式")
            prev_depth, next_depth = self.de.depth_encoder(
                prev_depth_img_input.to(torch.float32),
                next_depth_img_input.to(torch.float32),
            )
            if token_id is not None:
                token_list = token_id if isinstance(token_id, list) else [token_id]
                self._save_batch_results(token_list, prev_depth)
                self._save_batch_results(token_list, next_depth)
            return prev_depth, next_depth


class DepthloaderBatch(Depthloader):
    def __init__(self, device=None, args=None, data_loader=None,depth_encoder=None):
        super().__init__(device, args, data_loader)
        self.data_loader = data_loader
        self.args = args
        self.device = device
        self.de = depth_encoder


class DepthloaderOneimage(Depthloader):
    def __init__(self, device=None, args=None, data_loader=None,depth_encoder=None):
        super().__init__(device, args, data_loader)
        self.data_loader = data_loader
        self.args = args
        self.device = device
        self.de = depth_encoder   

    def process(
        self,
        token_id,
        prev_img_depth_tokenid,
        next_img__depth_tokenid,
        prev_depth_img_input,
        next_depth_img_input,
        cached_prev_depth,
        cached_next_depth,
    ):
        """处理单帧输入数据"""
        if cached_prev_depth is not None and cached_next_depth is not None:
            return cached_prev_depth, cached_next_depth
        # elif  self.use_cached_depth:
        #     # if isinstance(token_id, list) and len(token_id) == batch_size:
        #     prev_depth_list, next_depth_list = self._batch_process_tokens(
        #         prev_img_depth_tokenid,
        #         next_img__depth_tokenid,
        #         prev_depth_img_input,
        #         next_depth_img_input,
        #     )
        #     return self._merge_dictlist_2_dictone(
        #             prev_depth_list
        #         ), self._merge_dictlist_2_dictone(next_depth_list)
        # else:
        print("Depth Encoder: 动态推理模式")
        prev_depth, next_depth = self.de.depth_encoder(
            prev_depth_img_input.to(torch.float32),
            next_depth_img_input.to(torch.float32),
        )
        self._save_batch_results(prev_img_depth_tokenid, prev_depth)
        self._save_batch_results(next_img__depth_tokenid, next_depth)
        return prev_depth, next_depth

    def process_8frame(
        self,
        token_id,
        prev_img_depth_tokenid,
        next_img__depth_tokenid,
        next_img_context_tokenid,
        prev_depth_img_input,
        next_depth_img_input,
        next_context_img_input,
        cached_prev_depth,
        cached_next_depth,
        cached_next_context,
    ):
        """处理单帧输入数据"""
        if cached_prev_depth is not None and cached_next_depth is not None and cached_next_context is not None:
            return cached_prev_depth, cached_next_depth,cached_next_context
        # elif  self.use_cached_depth:
        #     # if isinstance(token_id, list) and len(token_id) == batch_size:
        #     prev_depth_list, next_depth_list = self._batch_process_tokens(
        #         prev_img_depth_tokenid,
        #         next_img__depth_tokenid,
        #         prev_depth_img_input,
        #         next_depth_img_input,
        #     )
        #     return self._merge_dictlist_2_dictone(
        #             prev_depth_list
        #         ), self._merge_dictlist_2_dictone(next_depth_list)
        # else:
        print("Depth Encoder: 动态推理模式")
        prev_depth, next_depth, next_context = self.de.depth_encoder(
            prev_depth_img_input.to(torch.float32),
            next_depth_img_input.to(torch.float32),
            next_context_img_input.to(torch.float32),
        )
        self._save_batch_results(prev_img_depth_tokenid, prev_depth)
        self._save_batch_results(next_img__depth_tokenid, next_depth)
        reshaped_next_img_context_tokenid = np.array(next_img_context_tokenid).reshape(-1, len(next_img_context_tokenid[0][0]))
        self._save_batch_results(reshaped_next_img_context_tokenid.tolist(), next_context)
        return prev_depth, next_depth, next_context
