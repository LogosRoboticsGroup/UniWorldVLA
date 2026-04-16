from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import seaborn as sns
from typing import List
import os

# 全局根目录配置
GLOBAL_OUTPUT_DIR = './seq_output/local'

# 导入 PyTorch（可选，用于处理 tensor 输入）
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # 避免 torch 未绑定的类型错误

def _to_numpy(input_data):
    """
    将输入数据转换为 numpy 数组
    支持 tensor、numpy 数组、列表等多种格式
    
    :param input_data: 输入数据，可以是 tensor、numpy 数组或列表
    :return: numpy 数组
    """
    if HAS_TORCH and torch is not None and isinstance(input_data, torch.Tensor):
        # 如果是 tensor，先移到 CPU 并转换为 numpy 数组
        return input_data.detach().cpu().numpy()
    elif isinstance(input_data, np.ndarray):
        # 如果已经是 numpy 数组，直接返回
        return input_data
    else:
        # 其他情况（列表等）转换为 numpy 数组
        return np.array(input_data)

def _parse_token_pairs_from_dict(targets_dict):
    """
    从字典中解析token对，重建成[start_id, end_id]数组，并提取无法成对的token
    
    规则：
    - 字典key的格式为 <|xxx|>，例如 <|soi|>、<|eoi|>
    - 如果第二个字母相同，则为一组（例如 'soi' 和 'eoi' 的第二个字母都是 'o'）
    - 第三个字母 's' 表示 start，'e' 表示 end
    
    :param targets_dict: 字典，key为token名称，value为ID值（支持 tensor）
                         例如：{"<|soi|>": 3, "<|eoi|>": 5, "<|sot|>": 7, "<|eot|>": 9}
    :return: 元组 (targets_list, single_tokens)
             - targets_list: 列表，每行包含[start_id, end_id]，例如：[[3, 5], [7, 9]]
             - single_tokens: 字典，包含无法成对的token信息 {token_name: token_id}
    """
    # 按第二个字母分组
    groups = {}
    single_tokens = {}

    for token, token_id in targets_dict.items():
        # 将 tensor 转换为标量值
        token_id = _to_numpy(token_id)
        # 如果是数组形式的tensor，取第一个元素
        if hasattr(token_id, 'size') and token_id.size > 1:
            token_id = token_id.item() if hasattr(token_id, 'item') else int(token_id[0])
        elif hasattr(token_id, 'item'):
            token_id = token_id.item()
        else:
            token_id = int(token_id)
        
        # 提取 <|> 内的内容
        if token.startswith('<|') and token.endswith('|>'):
            content = token[2:-2]  # 去掉 <| 和 |>

            if len(content) >= 2:
                first_char = content[0]
                second_char = content[1]  # 第二个字母
                third_char = content[2] if len(content) >= 3 else ''  # 第三个字母
                key = second_char + third_char
                if key not in groups:
                    groups[key] = {}

                # 第三个字母 's' 表示start，'e' 表示end
                if first_char == "s":
                    groups[key]["start"] = token_id
                    groups[key]["start_token"] = token
                elif first_char == "e":
                    groups[key]["end"] = token_id
                    groups[key]["end_token"] = token

    # 重建 [start_id, end_id] 数组，并收集无法成对的token
    targets_list = []
    for group_id, group in groups.items():
        if 'start' in group and 'end' in group:
            targets_list.append([group['start'], group['end']])
        else:
            # 无法成对，添加到single_tokens
            if 'start' in group:
                single_tokens[group['start_token']] = group['start']
            if 'end' in group:
                single_tokens[group['end_token']] = group['end']

    return targets_list, single_tokens


def _find_diff_ranges(diff_mask):
    """
    从差异掩码中找出连续的不一致区间
    
    :param diff_mask: 布尔数组，True表示该位置不一致
    :return: 区间列表，每个区间为 [start_idx, end_idx]
    """
    ranges = []
    in_range = False
    start_idx = 0
    
    for i, is_diff in enumerate(diff_mask):
        if is_diff and not in_range:
            # 开始一个新的不一致区间
            start_idx = i
            in_range = True
        elif not is_diff and in_range:
            # 结束当前不一致区间
            ranges.append([start_idx, i - 1])
            in_range = False
    
    # 处理最后一个区间
    if in_range:
        ranges.append([start_idx, len(diff_mask) - 1])
    
    return ranges

def _display_diff_ranges_on_plot(diff_ranges, seq1, seq2, seq_len):
    """
    Display inconsistent range information on the right side of the plot

    :param diff_ranges: List of inconsistent ranges
    :param seq1: First sequence
    :param seq2: Second sequence
    :param seq_len: Sequence length
    """
    if not diff_ranges:
        return

    # Sort by the first index of each range (ascending order)
    sorted_ranges = sorted(diff_ranges, key=lambda r: r[0])

    # Build range information text - display all ranges
    range_text = f"Inconsistent Ranges (Total: {len(diff_ranges)}):\n"
    range_text += "-" * 35 + "\n"

    for i, (start, end) in enumerate(sorted_ranges):  # Display ALL ranges
        length = end - start + 1
        range_text += f"Range {i+1:3d}: [{start:4d}, {end:4d}] (Len:{length:3d})\n"

    range_text += "-" * 35 + "\n"

    # 在图的右侧显示区间信息
    ax = plt.gca()

    # 获取数据范围
    y_min = min(float(seq1.min()), float(seq2.min()))
    y_max = max(float(seq1.max()), float(seq2.max()))
    y_range = y_max - y_min

    # 在x轴上标注区间
    for start, end in diff_ranges:
        # 绘制区间底部的标记
        plt.axvspan(start, end, alpha=0.1, color='red', ymin=0, ymax=0.02)
        # 在x轴上方标注区间范围
        plt.text((start + end) / 2, y_min - y_range * 0.08, 
                f'[{start},{end}]', 
                ha='center', va='top', fontsize=8, color='red',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', pad=1))

    # Display detailed range list - split into multiple columns if needed
    lines = range_text.split('\n')
    max_lines_per_column = 50  # Maximum lines per column
    
    if len(lines) <= max_lines_per_column:
        # Single column is enough
        plt.text(1.02, 1.0, range_text,
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.95, edgecolor='red', pad=1.5))
    else:
        # Split into multiple columns
        num_columns = (len(lines) + max_lines_per_column - 1) // max_lines_per_column
        
        for col in range(num_columns):
            start_line = col * max_lines_per_column
            end_line = min(start_line + max_lines_per_column, len(lines))
            column_text = '\n'.join(lines[start_line:end_line])
            
            # Calculate x position for this column with very tight spacing
            x_pos = 1.02 + col * 0.2  # 0.4 units spacing between columns (almost touching)
            
            plt.text(x_pos, 1.0, column_text,
                    transform=ax.transAxes,
                    fontsize=6,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.95, edgecolor='red', pad=0.4))


def visualize_sequence_intervals(sequence, targets, range_limits=None, save_path=None):
    """
    增强版序列可视化工具(支持区间筛选)
    :param sequence: 一维数组，包含ID序列（支持 tensor、numpy 数组或列表）
    :param targets: 可以是两种格式：
                    1. 字典格式：{"<|soi|>": 3, "<|eoi|>": 5, ...}
                       规则：||内第二个字母相同为一组，第三个字母s=start, e=end（支持 tensor 值）
                    2. 列表格式：n*2数组，每行包含[开始ID, 结束ID]（支持 tensor）
    :param range_limits: [start_index, end_index]或None，指定渲染区间
    :param save_path: 图片保存路径（如果为None，则使用全局根目录下的默认路径）
    :return: 显示间隔可视化图像
    """
    # 如果targets是字典，先解析成列表
    single_tokens = {}
    if isinstance(targets, dict):
        targets, single_tokens = _parse_token_pairs_from_dict(targets)
    elif HAS_TORCH and torch is not None and isinstance(targets, torch.Tensor):
        # 如果是 tensor，转换为 numpy 数组
        targets = _to_numpy(targets)
        if targets.ndim == 1 and len(targets) > 0:
            # 如果是一维 tensor，可能是单个 [start_id, end_id] 对
            targets = [list(targets)]
        elif targets.ndim == 2:
            # 如果是二维 tensor，转换为列表
            targets = targets.tolist()

    # 将 sequence 转换为 numpy 数组
    sequence = _to_numpy(sequence)
    
    # 设置保存路径
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(GLOBAL_OUTPUT_DIR, 'sequence_intervals.png')
    # else:
    #     save_path = os.path.join(GLOBAL_OUTPUT_DIR, save_path)

    plt.figure(figsize=(24, 12))  # 增加图形宽度以容纳右侧表格
    plt.subplots_adjust(hspace=0.5, right=0.75)  # 增加子图之间的垂直间距，为右侧表格预留空间

    # 应用区间筛选
    if range_limits:
        start_idx, end_idx = range_limits
        sequence = sequence[start_idx:end_idx+1]
        display_offset = start_idx  # 用于显示原始索引
    else:
        display_offset = 0

    # 查找所有目标ID的索引位置(考虑显示偏移)
    id_indices = {}
    for idx, val in enumerate(sequence):
        if val not in id_indices:
            id_indices[val] = []
        id_indices[val].append(idx + display_offset)  # 存储原始索引

    # 处理每个target对
    last_ax = None  # 保存最后一个子图用于后续放置表格
    for i, (start_id, end_id) in enumerate(targets):
        ax = plt.subplot(len(targets), 1, i+1)
        last_ax = ax  # 保存当前子图引用
        title = f"Target Range: {start_id}→{end_id}"
        if range_limits:
            title += f" (Display Range: {range_limits[0]}-{range_limits[1]})"
        plt.title(title)

        # 获取开始和结束ID的所有索引(考虑显示范围)
        start_indices = sorted([x for x in id_indices.get(start_id, [])
                              if not range_limits or (range_limits[0] <= x <= range_limits[1])])
        end_indices = sorted([x for x in id_indices.get(end_id, [])
                            if not range_limits or (range_limits[0] <= x <= range_limits[1])])

        if not start_indices or not end_indices:
            plt.text(0.5, 0.5, 'Target ID not found in display range', ha='center', va='center')
            continue
        
        # 创建自适应的横轴映射：将原始索引映射到非等间距的显示坐标
        def create_adaptive_x_mapping(all_indices, min_gap=5):
            """
            创建自适应的横轴映射，确保关键点之间有足够的最小间距
            
            :param all_indices: 所有关键索引的排序列表
            :param min_gap: 关键点之间的最小显示间距
            :return: 映射函数，将原始索引转换为显示坐标
            """
            if not all_indices:
                return lambda x: x
            
            # 收集所有需要放大的关键点及其相邻点
            key_points = set()
            for idx in all_indices:
                # 关键点本身
                key_points.add(idx)
                # 关键点前后的点也保留
                key_points.add(idx - 1)
                key_points.add(idx + 1)
            
            # 排序关键点
            sorted_keys = sorted(key_points)
            
            # 为区间分配显示坐标
            x_mapping = {}
            display_x = 0
            
            # 第一个区间：起点到第一个关键点
            if sorted_keys[0] > 0:
                x_mapping[0] = 0
                x_mapping[sorted_keys[0]] = display_x
                display_x += min_gap
            else:
                x_mapping[sorted_keys[0]] = display_x
                display_x += min_gap
            
            # 中间区间：关键点之间
            for i in range(1, len(sorted_keys)):
                prev_key = sorted_keys[i-1]
                curr_key = sorted_keys[i]
                
                # 计算原始距离
                original_dist = curr_key - prev_key
                
                # 如果距离很小（小于阈值），压缩显示
                # 如果距离较大，按比例适当放大
                if original_dist < min_gap:
                    # 压缩：间距与原始距离成正比
                    display_x += max(1, original_dist * 0.5)
                elif original_dist < 2 * min_gap:
                    # 轻微放大
                    display_x += min_gap * 1.5
                else:
                    # 较大距离：适当放大
                    display_x += original_dist * 0.8
                
                x_mapping[curr_key] = display_x
            
            # 最后一个区间：最后一个关键点到终点
            if sorted_keys[-1] < len(sequence) - 1:
                x_mapping[len(sequence) - 1] = display_x + (len(sequence) - 1 - sorted_keys[-1]) * 0.3
            
            # 创建插值函数
            def get_display_x(original_idx):
                # 找到当前索引所在的区间
                keys = sorted_keys.copy()
                
                # 如果是关键点，直接返回
                if original_idx in x_mapping:
                    return x_mapping[original_idx]
                
                # 否则在最近的关键点之间插值
                for i in range(len(keys)):
                    if keys[i] > original_idx:
                        # 在 keys[i-1] 和 keys[i] 之间
                        prev_key = keys[i-1]
                        next_key = keys[i]
                        prev_x = x_mapping[prev_key]
                        next_x = x_mapping[next_key]
                        
                        # 线性插值
                        ratio = (original_idx - prev_key) / (next_key - prev_key)
                        return prev_x + ratio * (next_x - prev_x)
                
                # 如果超出范围，返回最后一个关键点的坐标
                return x_mapping[keys[-1]]
            
            return get_display_x
        
        # 获取所有关键点
        all_key_indices = sorted(start_indices + end_indices)
        
        # 创建自适应映射
        adaptive_x = create_adaptive_x_mapping(all_key_indices, min_gap=8)
        
        # 使用自适应坐标绘制所有点
        display_x_coords = [adaptive_x(i) for i in range(len(sequence))]

        # 绘制无法成对的token（用紫色星号标记，使用自适应坐标）
        for token_name, token_id in single_tokens.items():
            if token_id in id_indices:
                for idx in id_indices[token_id]:
                    disp_idx_idx = idx if not range_limits else idx - display_offset
                    disp_idx = adaptive_x(disp_idx_idx)
                    plt.plot(disp_idx, sequence[disp_idx_idx], 'm*', markersize=10)
                    plt.text(disp_idx, float(sequence[disp_idx_idx])-0.8, f'{token_name}\nidx:{idx}',
                            ha='center', va='top', color='purple', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple'))

        # 配对最近的start和end
        pairs = []
        s_idx = e_idx = 0
        while s_idx < len(start_indices) and e_idx < len(end_indices):
            if start_indices[s_idx] < end_indices[e_idx]:
                pairs.append((start_indices[s_idx], end_indices[e_idx]))
                s_idx += 1
                e_idx += 1
            else:
                e_idx += 1

        # 收集所有的delta值并去重
        delta_info = {}  # 存储delta值及其对应的位置信息 {delta: [(pair1_start, pair1_end), ...]}
        for start, end in pairs:
            delta = end - start
            if delta not in delta_info:
                delta_info[delta] = []
            delta_info[delta].append((start, end))
        
        # 获取去重后的delta值（按值排序）
        unique_deltas = sorted(delta_info.keys())
        
        # 对每个唯一的delta值，选择一个代表性位置进行标注
        delta_annotation_positions = {}
        for delta in unique_deltas:
            # 选择该delta值对应的第一个位置作为标注位置
            first_pair = delta_info[delta][0]
            delta_annotation_positions[delta] = first_pair

        # 收集所有索引对信息用于显示
        index_pairs_list = [(start, end) for start, end in pairs]
        
        # 标记和连接每对start-end（使用自适应坐标）
        for start, end in pairs:
            delta = end - start
            # 转换为自适应显示坐标
            start_display = start if not range_limits else start - display_offset
            end_display = end if not range_limits else end - display_offset
            disp_start = adaptive_x(start_display)
            disp_end = adaptive_x(end_display)

            # 获取当前子图的 y 范围
            ylim = ax.get_ylim()
            y_min, y_max = ylim[0], ylim[1]
            y_pad = (y_max - y_min) * 0.15  # 使用子图高度的 15% 作为垂直间距

            # 标记start点（使用自适应坐标）
            plt.plot(disp_start, sequence[start_display], 'go', markersize=8)
            # 确保 index 标注在子图内部
            start_label_y = float(sequence[start_display]) + y_pad
            start_label_y = min(start_label_y, y_max - y_pad * 0.5)  # 确保不超过上边界
            plt.text(disp_start, start_label_y, f'idx:{start}',
                    ha='center', va='bottom', color='green')

            # 标记end点（使用自适应坐标）
            plt.plot(disp_end, sequence[end_display], 'ro', markersize=8)
            # 确保 index 标注在子图内部
            end_label_y = float(sequence[end_display]) + y_pad
            end_label_y = min(end_label_y, y_max - y_pad * 0.5)  # 确保不超过上边界
            plt.text(disp_end, end_label_y, f'idx:{end}',
                    ha='center', va='bottom', color='red')

            # 绘制箭头（使用自适应坐标）
            arrow = FancyArrowPatch(
                (disp_start, float(sequence[start_display])),
                (disp_end, float(sequence[end_display])),
                arrowstyle='<->', mutation_scale=15,
                color='blue', alpha=0.7)
            ax.add_patch(arrow)

        # 在指定位置标注去重后的delta值（避免重叠）
        for delta, (start, end) in delta_annotation_positions.items():
            start_display = start if not range_limits else start - display_offset
            end_display = end if not range_limits else end - display_offset
            disp_start = adaptive_x(start_display)
            disp_end = adaptive_x(end_display)
            
            # 在箭头中间显示delta（使用自适应坐标）
            mid_x = (disp_start + disp_end) / 2
            mid_y = (sequence[start_display] + sequence[end_display]) / 2
            
            # 获取当前子图的 y 范围
            ylim = ax.get_ylim()
            y_min, y_max = ylim[0], ylim[1]
            y_pad = (y_max - y_min) * 0.10  # 使用子图高度的 10% 作为垂直间距
            
            # 如果有多个相同的delta，显示统计信息
            count = len(delta_info[delta])
            label_text = f'Δ={delta}' if count == 1 else f'Δ={delta}(x{count})'
            
            # 确保 delta 标注在子图内部
            delta_label_y = float(mid_y) + y_pad
            delta_label_y = min(delta_label_y, y_max - y_pad * 0.5)  # 确保不超过上边界
            
            plt.text(mid_x, delta_label_y, label_text,
                    ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='blue', linewidth=1.5),
                    fontsize=10, fontweight='bold')
        
        # 在图的右侧显示delta统计信息
        if unique_deltas:
            delta_stats_text = "Delta统计:\n"
            delta_stats_text += "-" * 20 + "\n"
            for delta in unique_deltas:
                count = len(delta_info[delta])
                delta_stats_text += f"Δ={delta:4d}: {count:3d}次\n"
            delta_stats_text += "-" * 20
            
            # 显示统计信息
            plt.text(0.02, 0.98, delta_stats_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue'))

        # 设置x轴范围
        if range_limits:
            plt.xlim(-0.5, range_limits[1]-range_limits[0]+0.5)

        # # 美化图形
        # plt.grid(True, linestyle='--', alpha=0.6)
        # legend_elements = ['Start ID', 'End ID']
        # if single_tokens:
        #     legend_elements.append('Single Token')

        # plt.legend(legend_elements, loc='upper right')

    # 在整个图片的右侧空白区域显示索引对数组
    if len(targets) > 0:
        # 收集所有 subgraph 的索引对
        all_pairs = []
        for j, (start_id, end_id) in enumerate(targets):
            # 重新获取该对的索引
            start_indices = sorted([x for x in id_indices.get(start_id, [])
                                  if not range_limits or (range_limits[0] <= x <= range_limits[1])])
            end_indices = sorted([x for x in id_indices.get(end_id, [])
                                if not range_limits or (range_limits[0] <= x <= range_limits[1])])
            
            # 配对
            pairs = []
            s_idx = e_idx = 0
            while s_idx < len(start_indices) and e_idx < len(end_indices):
                if start_indices[s_idx] < end_indices[e_idx]:
                    pairs.append((start_indices[s_idx], end_indices[e_idx]))
                    s_idx += 1
                    e_idx += 1
                else:
                    e_idx += 1
            all_pairs.extend(pairs)
        
        # 对索引对进行排序（按 start 索引从小到大）
        all_pairs.sort(key=lambda x: x[0])
        
        if all_pairs:
            pairs_text = "索引对 (start, end):\n"
            pairs_text += "=" * 35 + "\n"
            
            # 显示所有索引对（已排序）
            for i, (start, end) in enumerate(all_pairs):
                delta = end - start
                pairs_text += f"{i+1:2d}: Δ={delta:4d} [{start:6d}, {end:6d}]\n"
            
            pairs_text += "=" * 35
            pairs_text += f"\n共 {len(all_pairs)} 对"
            
            # 使用第一个子图的坐标系，使表格上边界与第一个子图对齐
            # x=1.05 表示在子图右侧稍外一点
            # y=1.0 表示从子图顶部开始
            if last_ax is not None:
                # 获取第一个子图的引用
                first_ax = plt.gcf().axes[0] if plt.gcf().axes else last_ax
                plt.text(1.02, 1.0, pairs_text,
                        transform=first_ax.transAxes,
                        fontsize=7,
                        verticalalignment='top',
                        horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='orange', pad=1.5))
    
    # 调整布局，为右侧表格预留空间但不使用 rect 参数
    plt.tight_layout()
    plt.savefig(save_path, dpi=80)
    print(f"图片已保存至: {save_path}")
    plt.show()


def compare_sequences(seq1, seq2, seq1_label="Sequence 1", seq2_label="Sequence 2",
                     title="Sequence Comparison", show_markers=True, highlight_diff=True, save_path=None,
                     max_annotations=50, batch_lines=True, show_diff_ranges=True, csv_save_path=None):
    """
    Compare two numerical sequences on a single plot (optimized version)

    :param seq1: First sequence, 1D array or list (supports tensor, numpy array or list)
    :param seq2: Second sequence, 1D array or list (supports tensor, numpy array or list)
    :param seq1_label: Label name for the first sequence
    :param seq2_label: Label name for the second sequence
    :param title: Image title
    :param show_markers: Whether to show data point markers
    :param highlight_diff: Whether to highlight difference points
    :param save_path: Save path for image, if None then use global directory with default name
    :param max_annotations: Maximum number of annotations, only annotate some difference points if exceeded (default 50)
    :param batch_lines: Whether to batch draw vertical lines for performance improvement (default True)
    :param show_diff_ranges: Whether to display inconsistent interval indices on the plot (default True)
    :param csv_save_path: Save path for CSV file, if None then use global directory with default name
    :return: Display comparison image
    """
    # 设置保存路径
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(GLOBAL_OUTPUT_DIR, 'sequence_comparison.png')
    
    if csv_save_path is None:
        csv_save_path = os.path.join(GLOBAL_OUTPUT_DIR, 'sequence_comparison.csv')
    
    if seq1.shape != seq2.shape:
        print("=>>>>>>>>>>>>>>>>>>> sql shape不一致")
    
    plt.figure(figsize=(18, 8))  # 增加宽度以容纳右侧表格

    # 使用统一的转换函数处理输入数据
    seq1 = _to_numpy(seq1)
    seq2 = _to_numpy(seq2)

    # 确保两个序列长度相同
    min_len = min(len(seq1), len(seq2))
    seq1 = seq1[:min_len]
    seq2 = seq2[:min_len]

    x = np.arange(min_len)

    # 绘制两个序列
    if show_markers:
        plt.plot(x, seq1, 'b-o', label=seq1_label, alpha=0.7, markersize=4, linewidth=1.5)
        plt.plot(x, seq2, 'r-s', label=seq2_label, alpha=0.7, markersize=4, linewidth=1.5)
    else:
        plt.plot(x, seq1, 'b-', label=seq1_label, alpha=0.7, linewidth=2)
        plt.plot(x, seq2, 'r-', label=seq2_label, alpha=0.7, linewidth=2)

    # Highlight difference points (optimized version)
    if highlight_diff:
        diff_mask = seq1 != seq2
        if np.any(diff_mask):
            diff_indices = x[diff_mask]
            diff_values1 = seq1[diff_mask]
            diff_values2 = seq2[diff_mask]
            total_diffs = len(diff_indices)

            # 批量绘制垂直线（性能优化）
            if batch_lines and total_diffs > 0:
                # 使用 xlim 批量设置垂直线位置
                for x_pos in diff_indices:
                    plt.axvline(x=x_pos, color='orange', linestyle='--', alpha=0.3, linewidth=0.5)
            elif not batch_lines and total_diffs > 0:
                # 单独绘制（旧方式，保留兼容性）
                for i in diff_indices:
                    plt.axvline(x=i, color='orange', linestyle='--', alpha=0.3)

            # 智能标注：限制标注数量以提高性能
            if total_diffs > 0:
                # 提前计算范围偏移（避免重复计算）
                v_range = max(seq1.max(), seq2.max()) - min(seq1.min(), seq2.min())
                offset = v_range * 0.05

                # If too many difference points, only annotate some
                if total_diffs > max_annotations:
                    # 均匀采样标注点
                    step = total_diffs // max_annotations
                    if step > 1:
                        diff_indices = diff_indices[::step]
                        diff_values1 = diff_values1[::step]
                        diff_values2 = diff_values2[::step]
                        print(f"⚠️ Too many difference points ({total_diffs}), only annotating {len(diff_indices)} representative points")

                # 批量标注差异值
                for idx, v1, v2 in zip(diff_indices, diff_values1, diff_values2):
                    # 使用简化的标注样式以提高性能
                    plt.annotate(f'{v1}→{v2}',
                               xy=(idx, v1),
                               xytext=(idx, max(v1, v2) + offset),
                               fontsize=7, color='darkorange',
                               ha='center', va='bottom',
                               arrowprops=dict(arrowstyle='->', color='darkorange', alpha=0.5,
                                             connectionstyle='arc3,rad=0'))

                # If many unannotated difference points, show their positions with scatter plot
                if total_diffs > max_annotations:
                    all_diff_mask = seq1 != seq2
                    all_all_diff_indices = x[all_diff_mask]
                    # 标注的位置已经单独绘制，这里只标其他位置
                    annotated_set = set(diff_indices)
                    unannotated = [i for i in all_all_diff_indices if i not in annotated_set]
                    if unannotated:
                        unannotated_y = [max(float(seq1[i]), float(seq2[i])) for i in unannotated]
                        plt.scatter(unannotated, unannotated_y, c='orange', s=20,
                                  marker='x', alpha=0.5, label=f'Other {len(unannotated)} difference points')

                # 查找并显示不一致的连续区间
                if show_diff_ranges and total_diffs > 0:
                    diff_ranges = _find_diff_ranges(diff_mask)
                    if diff_ranges:
                        _display_diff_ranges_on_plot(diff_ranges, seq1, seq2, min_len)

    # 添加参考线
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)

    # 美化图形
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Index Position', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    plt.legend(loc='best', fontsize=11)

    # 计算并显示统计信息
    diff_count = np.sum(seq1 != seq2)
    diff_percent = (diff_count / min_len * 100) if min_len > 0 else 0

    # Build comprehensive statistics table
    diff_indices = np.where(seq1 != seq2)[0]
    
    stats_text = f'=== STATISTICS ===\n'
    stats_text += f'Sequence Length: {min_len}\n'
    stats_text += f'Difference Points: {diff_count}\n'
    stats_text += f'Difference Ratio: {diff_percent:.2f}%\n'
    stats_text += f'{seq1_label}\n'
    stats_text += f'  Mean:   {seq1.mean():.4f}\n'
    stats_text += f'  Std:    {seq1.std():.4f}\n'
    stats_text += f'  Min:    {seq1.min():.4f}\n'
    stats_text += f'  Max:    {seq1.max():.4f}\n'
    stats_text += f'{seq2_label}\n'
    stats_text += f'  Mean:   {seq2.mean():.4f}\n'
    stats_text += f'  Std:    {seq2.std():.4f}\n'
    stats_text += f'  Min:    {seq2.min():.4f}\n'
    stats_text += f'  Max:    {seq2.max():.4f}\n'
    
    # Add detailed difference table
    if len(diff_indices) > 0:
        stats_text += f'\n=== DIFFERENCES ===\n'
        stats_text += f'{"Idx":>4} | {seq1_label:>10} | {seq2_label:>10} | {"Diff":>8}\n'
        stats_text += f'-'*40 + '\n'
        
        sorted_diff_indices = sorted(diff_indices)
        for i, idx in enumerate(sorted_diff_indices):
            if i < 150:
                v1 = seq1[idx]
                v2 = seq2[idx]
                diff = v2 - v1
                stats_text += f'{idx:4d} | {v1:10.4f} | {v2:10.4f} | {diff:8.4f}\n'
        
        if len(sorted_diff_indices) > 150:
            stats_text += f'... and {len(sorted_diff_indices) - 150} more\n'
        
        stats_text += f'\nTotal: {len(sorted_diff_indices)} differences'

    # Adjust layout to accommodate inconsistent ranges table on the right
    plt.subplots_adjust(right=0.75)
    
    # Export statistics and differences to CSV (statistics table removed from plot, saved to CSV only)
    if csv_save_path:
        import csv
        
        with open(csv_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write summary statistics
            writer.writerow(['=== SUMMARY STATISTICS ==='])
            writer.writerow([])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Sequence Length', min_len])
            writer.writerow(['Difference Points', diff_count])
            writer.writerow(['Difference Ratio (%)', f'{diff_percent:.2f}'])
            writer.writerow([])
            
            # Write sequence 1 statistics
            writer.writerow([f'=== {seq1_label} ==='])
            writer.writerow([])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Mean', f'{seq1.mean():.4f}'])
            writer.writerow(['Std', f'{seq1.std():.4f}'])
            writer.writerow(['Min', f'{seq1.min():.4f}'])
            writer.writerow(['Max', f'{seq1.max():.4f}'])
            writer.writerow([])
            
            # Write sequence 2 statistics
            writer.writerow([f'=== {seq2_label} ==='])
            writer.writerow([])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Mean', f'{seq2.mean():.4f}'])
            writer.writerow(['Std', f'{seq2.std():.4f}'])
            writer.writerow(['Min', f'{seq2.min():.4f}'])
            writer.writerow(['Max', f'{seq2.max():.4f}'])
            writer.writerow([])
            
            # Write differences
            if len(diff_indices) > 0:
                writer.writerow(['=== DIFFERENCES ==='])
                writer.writerow([])
                writer.writerow(['Index', seq1_label, seq2_label, 'Difference'])
                
                sorted_diff_indices = sorted(diff_indices)
                for idx in sorted_diff_indices:
                    v1 = seq1[idx]
                    v2 = seq2[idx]
                    diff = v2 - v1
                    writer.writerow([idx, f'{v1:.4f}', f'{v2:.4f}', f'{diff:.4f}'])
                
                writer.writerow([])
                writer.writerow(['Total Differences', len(sorted_diff_indices)])
        
        print(f"Statistics and differences saved to: {csv_save_path}")

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")

    plt.show()


def visualize_sequence_all(labels, inputs, sptids_dict, save_path=None):
    """
    可视化序列的所有信息
    
    :param labels: 标签序列
    :param inputs: 输入序列
    :param sptids_dict: 特殊token ID字典
    :param save_path: 保存路径（如果为None，则使用全局根目录下的默认路径）
    """
    # 设置默认保存路径
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)
    
    # 为所有子函数调用使用全局根目录
    mmu_index = torch.where(inputs == sptids_dict['<|sot|>'].to(inputs.device))[1].unique()
    eod_img_d = torch.where(inputs == sptids_dict['<|eod|>'].to(inputs.device))[1].unique()
    compare_sequences(labels[0], inputs[0],
                     save_path=os.path.join(GLOBAL_OUTPUT_DIR, "labes_inputs_compare.png"),
                     csv_save_path=os.path.join(GLOBAL_OUTPUT_DIR, "labes_inputs_compare.csv"))
    compare_sequences(
        labels[0, mmu_index[-1] + 1 : eod_img_d[-1] + 1],
        inputs[0, mmu_index[-1] + 1 : eod_img_d[-1] + 1],
        save_path=os.path.join(GLOBAL_OUTPUT_DIR, "labes_inputs_compare_short.png"),
        csv_save_path=os.path.join(GLOBAL_OUTPUT_DIR, "labes_inputs_compare_short.csv")
    )
    visualize_sequence_intervals(labels[0], sptids_dict,
                                save_path=os.path.join(GLOBAL_OUTPUT_DIR, "labels_interval.png"))
    visualize_sequence_intervals(
        inputs[0], sptids_dict,
        save_path=os.path.join(GLOBAL_OUTPUT_DIR, "inputs_interval.png")
    )


def plot_attention_mask(attention_mask, save_path=None, title="Attention Mask",
                       cmap='viridis', show_colorbar=True, figsize=(12, 10)):
    """
    绘制2维attention mask矩阵
    
    :param attention_mask: 2维矩阵，支持格式：
                           - 0/1 格式 (numpy array, torch tensor, 或列表)
                           - True/False 格式 (numpy bool array, torch bool tensor)
    :param save_path: 保存路径，如果为None则使用全局根目录下的默认路径
    :param title: 图表标题
    :param cmap: 颜色映射，默认 'viridis'
    :param show_colorbar: 是否显示颜色条，默认True
    :param figsize: 图形大小，默认 (12, 10)
    :return: 显示attention mask热力图
    """
    # 设置保存路径
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(GLOBAL_OUTPUT_DIR, 'attention_mask.png')
    else:
        save_path = os.path.join(GLOBAL_OUTPUT_DIR, save_path)
    # 转换为numpy数组
    mask_array = _to_numpy(attention_mask)

    # 确保是2维数组
    if mask_array.ndim != 2:
        raise ValueError(f"attention_mask must be 2D, got {mask_array.ndim}D array")

    # 如果是布尔类型，转换为0/1
    if mask_array.dtype == bool:
        mask_array = mask_array.astype(int)

    # 创建图形
    plt.figure(figsize=figsize)

    # 绘制热力图
    im = plt.imshow(mask_array, cmap=cmap, aspect='auto', interpolation='nearest')

    # 设置标题和标签
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Target Position', fontsize=12)
    plt.ylabel('Source Position', fontsize=12)

    # 添加颜色条
    if show_colorbar:
        cbar = plt.colorbar(im, ax=plt.gca())
        cbar.set_label('Attention Mask Value', rotation=270, labelpad=20, fontsize=11)

    # 添加网格线（可选）
    plt.grid(False)

    # 添加图例说明
    unique_values = np.unique(mask_array)
    if len(unique_values) <= 10:
        legend_text = "Unique values: " + ", ".join(map(str, unique_values))
        plt.text(1.02, 0.98, legend_text, 
                transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 显示或保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention mask saved to: {save_path}")

    plt.tight_layout()
    plt.show()


def visualize_tensor_equality(
    tensor_list,
    figure_size=(16, 10),
    cmap='RdYlGn',
    title='Tensor Equality Similarity Heatmap with Statistics',
    save_path=None,
    element_level=True,
):
    """
    Visualize equality relationships between tensors in a list with element-level similarity statistics
    
    Args:
        tensor_list: List containing multiple tensors (supports tensor, numpy arrays, or lists)
        figure_size: Figure size (width, height)
        cmap: Color mapping scheme, 'RdYlGn' means red-yellow-green (green=equal, red=not equal)
        title: Figure title
        save_path: Path to save the image, if None then use global directory with default name
        element_level: If True, show element-level similarity percentage; if False, show binary equality
    
    Returns:
        fig, ax_heatmap, similarity_matrix: matplotlib figure, axis, and similarity matrix
    """
    # 设置保存路径
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(GLOBAL_OUTPUT_DIR, 'tensor_equality_heatmap.png')
    else:
        save_path = os.path.join(GLOBAL_OUTPUT_DIR, save_path)
    # Convert all tensors to numpy arrays
    numpy_list = [_to_numpy(t) for t in tensor_list]
    
    n = len(numpy_list)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((n, n), dtype=float)
    
    # Fill matrix, compare each pair of tensors
    for i in range(n):
        for j in range(n):
            if element_level:
                # Calculate element-level similarity
                if np.array_equal(numpy_list[i], numpy_list[j]):
                    similarity_matrix[i, j] = 100.0  # 100% similarity if tensors are identical
                elif numpy_list[i].shape == numpy_list[j].shape:
                    # Count equal elements for tensors with same shape
                    total_elements = numpy_list[i].size
                    equal_elements = np.sum(numpy_list[i] == numpy_list[j])
                    similarity_matrix[i, j] = (equal_elements / total_elements) * 100
                else:
                    similarity_matrix[i, j] = 0.0  # Different shapes, no element comparison
            else:
                # Binary equality (original behavior)
                similarity_matrix[i, j] = 100.0 if np.array_equal(numpy_list[i], numpy_list[j]) else 0.0
    
    # Statistics
    total_pairs = n * n
    if element_level:
        # For element-level: define "equal" as >50% similarity
        equal_threshold = 50.0
        equal_pairs = np.sum(similarity_matrix >= equal_threshold)
        not_equal_pairs = total_pairs - equal_pairs
        avg_similarity = np.mean(similarity_matrix)
        # Diagonal elements are always 100%
        off_diagonal_similarity = similarity_matrix[np.triu_indices(n, k=1)]
        avg_off_diagonal_similarity = np.mean(off_diagonal_similarity) if len(off_diagonal_similarity) > 0 else 0
    else:
        equal_threshold = 100.0
        equal_pairs = np.sum(similarity_matrix == equal_threshold)
        not_equal_pairs = total_pairs - equal_pairs
        avg_similarity = np.mean(similarity_matrix)
        off_diagonal_similarity = similarity_matrix[np.triu_indices(n, k=1)]
        avg_off_diagonal_similarity = np.mean(off_diagonal_similarity) if len(off_diagonal_similarity) > 0 else 0
    
    equal_percentage = (equal_pairs / total_pairs) * 100
    not_equal_percentage = (not_equal_pairs / total_pairs) * 100
    
    # 创建标签
    labels = [f'Tensor {i}\\nShape: {numpy_list[i].shape}'
              for i in range(n)]
    
    # Set beautiful style - use seaborn-v0_8-darkgrid
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    
    # Create main figure (heatmap)
    fig = plt.figure(figsize=figure_size)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 0.4])
    
    ax_heatmap = fig.add_subplot(gs[0, :2])  # Heatmap occupies 2/3 of top area
    
    # Draw heatmap
    if element_level:
        sns.heatmap(similarity_matrix,
                    annot=True,
                    cmap=cmap,
                    cbar_kws={'label': 'Element Similarity (%)'},
                    square=True,
                    linewidths=2,
                    linecolor='black',
                    xticklabels=labels,
                    yticklabels=labels,
                    fmt='.1f',
                    ax=ax_heatmap,
                    vmin=0, vmax=100)
    else:
        # Binary display: convert percentage to 0 or 1 for display
        display_matrix = (similarity_matrix == 100.0).astype(int)
        sns.heatmap(display_matrix,
                    annot=True,
                    cmap=cmap,
                    cbar_kws={'label': 'Equal (1) / Not Equal (0)'},
                    square=True,
                    linewidths=2,
                    linecolor='black',
                    xticklabels=labels,
                    yticklabels=labels,
                    fmt='.0f',
                    ax=ax_heatmap,
                    vmin=0, vmax=1)
    
    # 设置热力图标题和标签
    ax_heatmap.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax_heatmap.set_xlabel('Tensor Index', fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel('Tensor Index', fontsize=12, fontweight='bold')
    
    # 调整标签旋转
    ax_heatmap.set_xticklabels(labels, rotation=45, ha='right')
    ax_heatmap.set_yticklabels(labels, rotation=0)
    
    # 创建统计信息文本框
    ax_stats = fig.add_subplot(gs[0, 2])  # 统计信息占据右上方的1/3
    ax_stats.axis('off')
    
    if element_level:
        stats_text = f"""Statistics
{'='*25}
Total Tensors: {n}
Total Comps: {total_pairs}

Similar (>50%): {equal_pairs}
Percentage: {equal_percentage:.1f}%

Diff (<50%): {not_equal_pairs}
Percentage: {not_equal_percentage:.1f}%

Avg Similar: {avg_similarity:.1f}%
Avg (off-diag): {avg_off_diagonal_similarity:.1f}%

{'='*25}
"""
    else:
        stats_text = f"""Statistics
{'='*25}
Total Tensors: {n}
Total Comps: {total_pairs}

Equal Pairs: {equal_pairs}
Percentage: {equal_percentage:.1f}%

Not Equal: {not_equal_pairs}
Percentage: {not_equal_percentage:.1f}%

{'='*25}
"""
    
    ax_stats.text(0.05, 0.95, stats_text,
                 transform=ax_stats.transAxes,
                 fontsize=11,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue',
                          alpha=0.9, edgecolor='blue', linewidth=2))
    
    # 创建饼图显示占比
    ax_pie = fig.add_subplot(gs[1, 0])  # 饼图在左下角
    pie_data = [equal_pairs, not_equal_pairs]
    if element_level:
        pie_labels = ['Similar (>50%)', 'Different (<50%)']
    else:
        pie_labels = ['Equal', 'Not Equal']
    pie_colors = ['#90EE90', '#FFB6C1']  # Light green and light pink
    ax_pie.pie(pie_data, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
              startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax_pie.set_title('Similarity Percentage' if element_level else 'Equal vs Not Equal Percentage',
                     fontsize=12, fontweight='bold')
    
    # Create bar chart showing average similarity per tensor
    ax_bar = fig.add_subplot(gs[1, 1])  # Bar chart in middle-bottom
    if element_level:
        # Calculate average similarity with other tensors (excluding self)
        avg_similarities = []
        for i in range(n):
            other_similarities = [similarity_matrix[i, j] for j in range(n) if i != j]
            avg_similarities.append(np.mean(other_similarities) if other_similarities else 0)
        bar_data = avg_similarities
        ylabel = 'Avg Similarity (%)'
        bar_title = 'Avg Similarity per Tensor'
        # Use matplotlib colormap
        cmap = plt.get_cmap('RdYlGn')
        bar_colors = [cmap(s/100) for s in avg_similarities]
    else:
        equal_counts = np.sum((similarity_matrix == 100.0).astype(int), axis=1)  # Count of equal tensors
        bar_data = equal_counts
        ylabel = 'Equality Count'
        bar_title = 'Equality Count per Tensor'
        bar_colors = ['#90EE90'] * n
    
    bar_labels = [f'Num {i}' for i in range(n)]
    bars = ax_bar.bar(bar_labels, bar_data, color=bar_colors, edgecolor='green', linewidth=2)
    ax_bar.set_title(bar_title, fontsize=12, fontweight='bold')
    ax_bar.set_xlabel('Tensor Index', fontsize=10, fontweight='bold')
    ax_bar.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, bar_data):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:.1f}%' if element_level else f'{int(count)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 添加详细说明文本框
    ax_info = fig.add_subplot(gs[1, 2])  # 说明框在右下角
    ax_info.axis('off')
    
    if element_level:
        info_text = """Legend
{'='*25}
- Green: High Similarity
- Red: Low Similarity
- Diagonal: Self (100%)
- Values: Element Similarity
  Percentage
- Pie: Similarity Dist
- Bar: Avg Similarity
{'='*25}
"""
    else:
        info_text = """Legend
{'='*25}
- Green: Tensors Equal
- Red: Tensors Not Equal
- Diagonal: Self-comparison
- Pie: Overall Percentage
- Bar: Equality Count
{'='*25}
"""
    
    ax_info.text(0.05, 0.95, info_text,
                transform=ax_info.transAxes,
                fontsize=9,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                         alpha=0.9, edgecolor='orange', linewidth=2))
    
    # 调整整体布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    
    # Print statistics to console
    print("\n" + "="*50)
    print(f"Tensor Similarity Statistics (Total: {n} tensors)")
    print("="*50)
    print(f"Element-level mode: {element_level}")
    print(f"Total Comparison Pairs: {total_pairs}")
    if element_level:
        print(f"Similar Pairs (>50%): {equal_pairs} ({equal_percentage:.1f}%)")
        print(f"Different Pairs (<50%): {not_equal_pairs} ({not_equal_percentage:.1f}%)")
        print(f"Average Similarity: {avg_similarity:.1f}%")
        print(f"Average Off-Diagonal Similarity: {avg_off_diagonal_similarity:.1f}%")
        print("\nDetailed Similarity Matrix:")
        for i in range(n):
            for j in range(n):
                print(f"  Tensor {i} vs {j}: {similarity_matrix[i, j]:.1f}% similar")
    else:
        equal_counts = np.sum((similarity_matrix == 100.0).astype(int), axis=1)
        print(f"Equal Pairs: {equal_pairs} ({equal_percentage:.1f}%)")
        print(f"Not Equal Pairs: {not_equal_pairs} ({not_equal_percentage:.1f}%)")
        print("\nEquality Count per Tensor:")
        for i in range(n):
            print(f"  Tensor {i}: {equal_counts[i]} times")
    print("="*50)
    
    return fig, ax_heatmap, similarity_matrix


def plot_two_sequences_scatter(seq1, seq2, seq1_label="Sequence 1", seq2_label="Sequence 2",
                               seq1_color='blue', seq2_color='red',
                               seq1_marker='o', seq2_marker='s',
                               markersize=6, alpha=0.7,
                               title="Two Sequences Scatter Plot",
                               xlabel="Index", ylabel="Value",
                               save_path=None, figsize=(12, 8)):
    """
    在一个图形上绘制两个序列的散点图，使用不同颜色，不连线
    
    :param seq1: 第一个序列（支持 numpy array, torch tensor, 或列表）
    :param seq2: 第二个序列（支持 numpy array, torch tensor, 或列表）
    :param seq1_label: 第一个序列的标签，默认 "Sequence 1"
    :param seq2_label: 第二个序列的标签，默认 "Sequence 2"
    :param seq1_color: 第一个序列的颜色，默认蓝色
    :param seq2_color: 第二个序列的颜色，默认红色
    :param seq1_marker: 第一个序列的标记样式，默认圆圈 'o'
    :param seq2_marker: 第二个序列的标记样式，默认方块 's'
    :param markersize: 标记大小，默认6
    :param alpha: 透明度，默认0.7
    :param title: 图表标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param save_path: 保存路径，如果为None则使用全局根目录下的默认路径
    :param figsize: 图形大小，默认 (12, 8)
    :return: 显示散点图
    """
    # 设置保存路径
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(GLOBAL_OUTPUT_DIR, 'two_sequences_scatter.png')
    else:
        save_path = os.path.join(GLOBAL_OUTPUT_DIR, save_path)
    # 转换为numpy数组
    arr1 = _to_numpy(seq1)
    arr2 = _to_numpy(seq2)
    
    # 创建索引数组
    indices1 = np.arange(len(arr1))
    indices2 = np.arange(len(arr2))
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 绘制第一个序列
    plt.scatter(indices1, arr1, 
                color=seq1_color, marker=seq1_marker,
                s=markersize**2, alpha=alpha,
                label=seq1_label, edgecolors='black', linewidths=0.5)
    
    # 绘制第二个序列
    plt.scatter(indices2, arr2, 
                color=seq2_color, marker=seq2_marker,
                s=markersize**2, alpha=alpha,
                label=seq2_label, edgecolors='black', linewidths=0.5)
    
    # 设置标题和标签
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 添加图例
    plt.legend(loc='best', fontsize=10)
    
    # 添加统计信息
    stats_text = f"{seq1_label}:\n"
    stats_text += f"  Length: {len(arr1)}\n"
    stats_text += f"  Min: {arr1.min():.2f}\n"
    stats_text += f"  Max: {arr1.max():.2f}\n"
    stats_text += f"  Mean: {arr1.mean():.2f}\n\n"
    stats_text += f"{seq2_label}:\n"
    stats_text += f"  Length: {len(arr2)}\n"
    stats_text += f"  Min: {arr2.min():.2f}\n"
    stats_text += f"  Max: {arr2.max():.2f}\n"
    stats_text += f"  Mean: {arr2.mean():.2f}"
    
    plt.text(1.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 显示或保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


# 示例用法
if __name__ == "__main__":
    example_sequence = [1, 2, 3, 5, 2, 7, 2, 4, 5, 6, 2, 8, 2, 9, 2, 1, 3, 5, 7, 9]
    
    # 格式1：使用列表格式（原格式）
    example_targets_list = [
        [2, 5],  # 开始ID=2，结束ID=5
        [5, 9],
        [1, 2]
    ]
    
    # 格式2：使用字典格式（新格式）
    # 规则：||内第二个字母相同为一组，第三个字母s=start, e=end
    # soi和eoi：第二个字母都是'o'，一组 → [3, 5]
    # sod和eod：第二个字母都是'd'，一组 → [7, 9]
    # sos和eos：第二个字母都是'o'，一组 → [1, 2]
    # <|pad|>：无法成对，会用紫色星号标记
    example_targets_dict = {
        "<|soi|>": 3,  # start of image, 第三个字母's'表示start
        "<|eoi|>": 5,  # end of image, 第三个字母'e'表示end
        "<|sod|>": 7,  # start of dynamic
        "<|eod|>": 9,  # end of dynamic
        "<|sos|>": 1,  # start of sequence
        "<|eos|>": 2,  # end of sequence
        "<|sot|>": 4,  # start of text
        "<|eot|>": 6,  # end of text
        "<|pad|>": 0,  # 无法成对的token，会用紫色星号标记
    }
    
    # 示例3：包含无法成对token的字典
    example_targets_dict_with_single = {
        "<|soi|>": 3,
        "<|eoi|>": 5,
        "<|sod|>": 7,  #只有start，没有对应的eod
        "<|pad|>": 0,  # 无法成对的token
        "<|act|>": 8,  # 无法成对的token
    }
    
    print("=" * 50)
    print("示例1：使用列表格式（原格式）")
    print("=" * 50)
    
    # 示例1：使用列表格式渲染全部数据
    visualize_sequence_intervals(example_sequence, example_targets_list)
    
    # 示例2：使用列表格式只渲染[5,15]区间
    visualize_sequence_intervals(example_sequence, example_targets_list, range_limits=[5,15])
    
    print("\n" + "=" * 50)
    print("示例2：使用字典格式（新格式）")
    print("=" * 50)
    
    # 示例3：使用字典格式渲染全部数据
    visualize_sequence_intervals(example_sequence, example_targets_dict)
    
    # 示例4：使用字典格式只渲染[3,10]区间
    visualize_sequence_intervals(example_sequence, example_targets_dict, range_limits=[3,10])
    
    print("\n" + "=" * 50)
    print("示例3：使用字典格式（包含无法成对的token）")
    print("=" * 50)
    
    # 示例5：包含无法成对token的字典
    visualize_sequence_intervals(example_sequence, example_targets_dict_with_single)
    
    print("=" * 50)
    print("序列对比示例")
    print("=" * 50)
    
    # 序列对比示例
    seq_a = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17]
    seq_b = [1, 2, 4, 5, 8, 9, 12, 13, 16, 17]
    
    # 示例1：基本对比
    compare_sequences(seq_a, seq_b,
                     seq1_label="序列A",
                     seq2_label="序列B",
                     title="两个序列的数值对比")
    
    # 示例2：不显示标记，不高亮差异
    compare_sequences(seq_a, seq_b,
                     seq1_label="序列A",
                     seq2_label="序列B",
                     title="两个序列的对比（简洁版）",
                     show_markers=False,
                     highlight_diff=False,
                     save_path='sequence_comparison.png')
    
    # 示例3：更长的序列对比
    seq_long_1 = np.random.randn(50).cumsum() + 10
    seq_long_2 = seq_long_1 + np.random.randn(50) * 0.5
    
    compare_sequences(seq_long_1, seq_long_2,
                     seq1_label="原始序列",
                     seq2_label="噪声序列",
                     title="带噪声的序列对比",
                     save_path='long_sequence_comparison.png')
