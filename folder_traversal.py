import os

def adapt_raw_scene_blobs(root_folder_path):
    """
    遍历指定文件夹的二级子目录，生成以二级子目录为键，
    值为包含[当前目录, 一级目录, 二级目录]的列表的字典
    
    Args:
        root_folder_path (str): 根文件夹路径
    
    Returns:
        dict: 以二级子目录名为键，值为[当前目录, 一级目录, 二级目录]列表的字典
    """
    folder_dict = {}
    
    # 检查根文件夹是否存在
    if not os.path.exists(root_folder_path):
        print(f"错误：路径 '{root_folder_path}' 不存在")
        return folder_dict
    
    # 检查是否为文件夹
    if not os.path.isdir(root_folder_path):
        print(f"错误：'{root_folder_path}' 不是一个文件夹")
        return folder_dict
    
    # 获取根目录的基准名称
    root_name = os.path.basename(root_folder_path)
    
    try:
        # 遍历一级子目录
        for first_level_item in os.listdir(root_folder_path):
            first_level_path = os.path.join(root_folder_path, first_level_item)
            
            # 检查是否为文件夹
            if os.path.isdir(first_level_path):
                # 遍历二级子目录
                for second_level_item in os.listdir(first_level_path):
                    second_level_path = os.path.join(first_level_path, second_level_item)
                    
                    # 检查是否为文件夹
                    if os.path.isdir(second_level_path):
                        # 以二级目录名为键，值为[当前目录, 一级目录, 二级目录]的列表
                        folder_dict[second_level_item] = [root_name, first_level_item, second_level_item]
    except PermissionError:
        print(f"错误：没有权限访问 '{root_folder_path}'")
    except Exception as e:
        print(f"遍历文件夹时发生错误：{e}")
    
    return folder_dict

# 示例用法
if __name__ == "__main__":
    # 指定要遍历的根文件夹路径
    root_path = "."  # 当前目录，您可以修改为任何您想遍历的路径
    
    # 调用函数
    result = adapt_raw_scene_blobs(root_path)
    
    # 打印结果
    print("二级子目录 -> [当前目录, 一级目录, 二级目录] 字典:")
    for subfolder, path_list in result.items():
        print(f"  '{subfolder}' -> {path_list}")
    
    print(f"\n总共找到 {len(result)} 个二级子目录")