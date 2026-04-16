from pathlib import Path
import pickle

def build_scene_index(root):
    """
    root: nuplan_scene_blobs 路径
    return: dict[str, tuple[str, str]]  # (camera_path, lidar_path)
    """
    root = Path(root)
    scene2path = {}

    for split_dir in root.iterdir():  # train / val / test
        if not split_dir.is_dir():
            continue

        # 先找 camera
        for camera_dir in split_dir.glob("*camera*"):
            if not camera_dir.is_dir():
                continue

            for scene_dir in camera_dir.iterdir():
                if not scene_dir.is_dir():
                    continue
                scene_id = scene_dir.name
                # 如果已经有 lidar 记录，保留它
                lidar_path = scene2path.get(scene_id, ('', ''))[1]
                scene2path[scene_id] = (f"{split_dir.name}/{camera_dir.name}", lidar_path)

        # 再找 lidar
        for lidar_dir in split_dir.glob("*lidar*"):
            if not lidar_dir.is_dir():
                continue

            for scene_dir in lidar_dir.iterdir():
                if not scene_dir.is_dir():
                    continue
                scene_id = scene_dir.name
                # 如果已经有 camera 记录，保留它
                camera_path = scene2path.get(scene_id, ('', ''))[0]
                scene2path[scene_id] = (camera_path, f"{split_dir.name}/{lidar_dir.name}")

    return scene2path

def save_scene_index(root, out_file):
    path_mapping = build_scene_index(root)
    with open(out_file, "wb") as f:
        pickle.dump(path_mapping, f)

    print(f"Saved {len(path_mapping)} image paths to {out_file}")

root = "dataset/navsim/nuplan_scene_blobs"
out_file = "dataset/navsim/scene_index.pkl"

save_scene_index(root, out_file)