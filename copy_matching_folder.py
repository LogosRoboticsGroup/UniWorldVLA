#!/usr/bin/env python3
import os
import shutil
import argparse

def copy_matching_folder(src_base, tgt_base, folder_name):
    """复制匹配的文件夹及其内容"""
    src_path = os.path.join(src_base, folder_name)
    tgt_path = os.path.join(tgt_base, folder_name)
    
    if not os.path.exists(src_path):
        print(f"错误: 源文件夹不存在 {src_path}")
        return False
        
    try:
        # 删除目标文件夹(如果存在)
        if os.path.exists(tgt_path):
            shutil.rmtree(tgt_path)
        # 递归复制        
        shutil.copytree(src_path, tgt_path)
        print(f"成功复制: {src_path} -> {tgt_path}")
        return True
    except Exception as e:
        print(f"复制失败: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='复制匹配的文件夹')
    parser.add_argument('src_base', help='源基础目录路径')
    parser.add_argument('tgt_base', help='目标基础目录路径')
    parser.add_argument('folder_name', help='要复制的文件夹名称')
    
    args = parser.parse_args()
    
    if not copy_matching_folder(args.src_base, args.tgt_base, args.folder_name):
        exit(1)