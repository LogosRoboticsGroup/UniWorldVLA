"""
深度缓存工具函数
提供统一的缓存路径生成和token哈希计算功能
"""

import os
import hashlib
from typing import Optional


def get_depth_cache_path(depth_cache_dir: str, token_id: str, is_prev: bool = True,
                        use_subdirs: bool = True, naming_format: str = "hash_only") -> str:
    """
    根据token id生成深度数据缓存文件路径
    
    Args:
        depth_cache_dir: 缓存根目录
        token_id: token标识符
        is_prev: 是否为prev数据
        use_subdirs: 是否使用子目录结构（prev/next）
        naming_format: 文件命名格式，"hash_only"或"with_prefix"
        
    Returns:
        str: 缓存文件的完整路径
    """
    import os
    
    prefix = "prev" if is_prev else "next"
    token_hash = get_token_hash(token_id)
    
    if naming_format == "with_prefix":
        filename = f"{prefix}_depth_{token_hash}.pt"
        if use_subdirs:
            cache_path = os.path.join(depth_cache_dir, prefix)
            os.makedirs(cache_path, exist_ok=True)
            return os.path.join(cache_path, filename)
        else:
            return os.path.join(depth_cache_dir, filename)
    else:  # hash_only
        filename = f"{token_hash}.pt"
        if use_subdirs:
            cache_path = os.path.join(depth_cache_dir, prefix)
            os.makedirs(cache_path, exist_ok=True)
            return os.path.join(cache_path, filename)
        else:
            return os.path.join(depth_cache_dir, filename)


def get_token_hash(token_id: str) -> str:
    """
    生成token id的hash，用作文件名
    
    Args:
        token_id: token标识符
        
    Returns:
        str: token的hash值
    """
    # 目前直接返回token_id，保持向后兼容
    # 如需使用hash，可以取消下面的注释
    # return hashlib.md5(token_id.encode()).hexdigest()
    return token_id


def validate_cache_path(cache_path: str) -> bool:
    """
    验证缓存路径是否有效
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        bool: 路径是否有效
    """
    return os.path.exists(cache_path) and cache_path.endswith('.pt')


def get_cache_file_list(depth_cache_dir: str, is_prev: Optional[bool] = None) -> list:
    """
    获取缓存文件列表
    
    Args:
        depth_cache_dir: 缓存根目录
        is_prev: 如果为True只返回prev文件，False只返回next文件，None返回所有
        
    Returns:
        list: 缓存文件路径列表
    """
    cache_files = []
    
    if is_prev is None:
        subdirs = ['prev', 'next']
    else:
        subdirs = ['prev'] if is_prev else ['next']
    
    for subdir in subdirs:
        subdir_path = os.path.join(depth_cache_dir, subdir)
        if os.path.exists(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.pt'):
                    cache_files.append(os.path.join(subdir_path, file))
    
    return cache_files