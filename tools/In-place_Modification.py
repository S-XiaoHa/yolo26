#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic Light 数据集就地清洗与重映射脚本
功能：
1. 读取原有的 19 类标签，过滤掉不需要的类（如 black, countdown）。
2. 将保留的类别映射为全新的 0~12 号 ID。
3. 若某张图过滤后标签为空，则物理删除该 txt 以及对应的图片文件。
4. 原地更新 classes.txt。
"""

import os
from tqdm import tqdm

# ================= 配置区域 =================
DATASET_DIR = "/home/jzyh/xbzl/traffic_light/data_v1_filtered"

# 这是上一次跑完脚本后，目前 TXT 文件里实际对应的 19 个旧类别（顺序绝对不能错）
OLD_CLASSES = [
    "round_black", "round_red", "round_yellow", "round_green",
    "up_red", "up_yellow", "up_green",
    "left_black", "left_red", "left_yellow", "left_green",
    "right_red", "right_yellow", "right_green",
    "countdown_black", "countdown_red", "countdown_yellow", "countdown_green",
    "unknown"
]

# 这是你现在决定真正要训练的 12 个新类别
NEW_KEEP_CLASSES = [
    "round_red", "round_yellow", "round_green",
    "up_red", "up_yellow", "up_green",
    "left_red", "left_yellow", "left_green",
    "right_red", "right_yellow", "right_green",
]

# 1. 建立新类别的 ID 字典 (0~12)
NEW_CLASS_TO_ID = {cls: i for i, cls in enumerate(NEW_KEEP_CLASSES)}

# 2. 建立 [旧ID -> 新ID] 的映射表
# 只有在 NEW_KEEP_CLASSES 里的类才会出现在这个字典里
OLD_ID_TO_NEW_ID = {}
for old_id, cls_name in enumerate(OLD_CLASSES):
    if cls_name in NEW_KEEP_CLASSES:
        OLD_ID_TO_NEW_ID[old_id] = NEW_CLASS_TO_ID[cls_name]

def clean_and_remap_dataset():
    stats = {"kept_files": 0, "deleted_empty": 0, "lines_deleted": 0}
    
    phases = ["train", "val"]
    
    for phase in phases:
        labels_dir = os.path.join(DATASET_DIR, "labels", phase)
        images_dir = os.path.join(DATASET_DIR, "images", phase)
        
        if not os.path.exists(labels_dir):
            continue
            
        label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt") and f != "classes.txt"]
        print(f"\n--- 正在清洗 {phase} 集 (共 {len(label_files)} 个标签文件) ---")
        
        for label_file in tqdm(label_files, desc=f"  {phase} 清洗进度", unit="文件"):
            label_path = os.path.join(labels_dir, label_file)
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            file_modified = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                old_id = int(parts[0])
                
                # 如果这个旧 ID 在我们需要的映射表里，说明是要保留的类
                if old_id in OLD_ID_TO_NEW_ID:
                    new_id = OLD_ID_TO_NEW_ID[old_id]
                    # 替换开头的 ID，保留后面的坐标 (x, y, w, h)
                    new_line = f"{new_id} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
                    if new_id != old_id:
                        file_modified = True
                else:
                    # 如果不在映射表里（比如是 _black 或 countdown），直接抛弃这行
                    stats["lines_deleted"] += 1
                    file_modified = True

            # 检查清洗后的结果
            if not new_lines:
                # 标签变为空了 -> 彻底删除标签和对应的图片
                os.remove(label_path)
                
                # 寻找并删除对应的图片（支持多种后缀，因为你之前可能有 png 和 jpg）
                img_base = os.path.splitext(label_file)[0]
                img_deleted = False
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    img_path = os.path.join(images_dir, img_base + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        img_deleted = True
                        break
                
                stats["deleted_empty"] += 1
                # 终端实时输出被删掉的帧
                print(f"\n  [丢弃] {label_file} 内所有红绿灯均被过滤，已删除该帧！")
                
            elif file_modified:
                # 标签没空，但内容改变了（ID重映射了，或者删掉了部分不要的框），覆写回去
                with open(label_path, 'w') as f:
                    f.writelines(new_lines)
                stats["kept_files"] += 1
            else:
                stats["kept_files"] += 1

        # 更新当前文件夹下的 classes.txt
        classes_path = os.path.join(labels_dir, "classes.txt")
        with open(classes_path, "w") as f:
            f.write("\n".join(NEW_KEEP_CLASSES))

    print(f"\n{'='*45}")
    print(f"清洗与重映射完成！")
    print(f"保留且有效的帧数: {stats['kept_files']} 帧")
    print(f"因标签全部被过滤而删除的帧数: {stats['deleted_empty']} 帧")
    print(f"共计剔除了多少个无效红绿灯框: {stats['lines_deleted']} 个")
    print(f"现在的分类数已成功降维至: 12 类")
    print(f"{'='*45}")

if __name__ == "__main__":
    clean_and_remap_dataset()