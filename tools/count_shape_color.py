import os
from pathlib import Path

# ==================== 1. 路径配置 ====================
ROOT_PATH = "/home/jzyh/xbzl/traffic_light/data_v1_rebalanced"

# YOLO 默认会将 images/ 替换为 labels/ 来读取标签
TRAIN_LABEL_DIR = os.path.join(ROOT_PATH, "labels/train")
VAL_LABEL_DIR = os.path.join(ROOT_PATH, "labels/val")

# ==================== 2. 类别映射 ====================
# 对应 shape: class_id // 3
SHAPES = ["round", "up", "left", "right", "turn_around"]
# 对应 color: class_id % 3
COLORS = ["red", "yellow", "green"]

def count_labels(label_dir):
    # 初始化计数器
    shape_counts = {shape: 0 for shape in SHAPES}
    color_counts = {color: 0 for color in COLORS}
    
    label_path = Path(label_dir)
    if not label_path.exists():
        print(f"⚠️ 警告: 找不到路径 {label_dir}，请检查标签文件夹是否存在。")
        return shape_counts, color_counts

    # 遍历目录下所有 .txt 标签文件
    txt_files = list(label_path.glob("*.txt"))
    
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # YOLO 标签格式通常为: class_id x_center y_center width height
                parts = line.split()
                try:
                    class_id = int(parts[0])
                    
                    # 利用数学规律拆分 0-14 的 class_id
                    shape_idx = class_id // 3
                    color_idx = class_id % 3
                    
                    # 累加计数
                    if 0 <= shape_idx < len(SHAPES):
                        shape_counts[SHAPES[shape_idx]] += 1
                    if 0 <= color_idx < len(COLORS):
                        color_counts[COLORS[color_idx]] += 1
                        
                except (ValueError, IndexError):
                    # 略过可能存在的空行或格式错误行
                    continue
                    
    return shape_counts, color_counts, len(txt_files)

def main():
    for split, label_dir in [("Train (训练集)", TRAIN_LABEL_DIR), ("Val (验证集)", VAL_LABEL_DIR)]:
        print(f"\n==================== {split} ====================")
        shape_counts, color_counts, file_count = count_labels(label_dir)
        
        print(f"总计扫描标签文件数: {file_count} 个")
        print("\n【形状统计 (Shapes)】")
        for shape, count in shape_counts.items():
            print(f"  - {shape:<15} : {count} 个")
            
        print("\n【颜色统计 (Colors)】")
        for color, count in color_counts.items():
            print(f"  - {color:<15} : {count} 个")

if __name__ == "__main__":
    main()