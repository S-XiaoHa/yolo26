#!/usr/bin/env python3
"""
独立统计脚本 - 直接扫描原始数据集，统计保留类别的标注数量
不依赖已处理的数据集，可直接在原始数据上运行.
"""

import os
from collections import defaultdict

# ================= 配置区域（与 dataset_prepare.py 保持一致）=================
SOURCE_DIR = "/home/jzyh/xbzl/traffic_light/0825"  # 原始数据集根目录

# 原始 37 个类别的完整定义
ORIGINAL_CLASSES = [
    "round_red",
    "round_yellow",
    "round_green",
    "up_red",
    "up_yellow",
    "up_green",
    "left_red",
    "left_yellow",
    "left_green",
    "right_red",
    "right_yellow",
    "right_green",
]

# 需要保留的类别列表（顺序即为新 ID 顺序）
KEEP_CLASSES = [
    "round_red",
    "round_yellow",
    "round_green",
    "up_red",
    "up_yellow",
    "up_green",
    "left_red",
    "left_yellow",
    "left_green",
    "right_red",
    "right_yellow",
    "right_green",
]


# 构建原始ID -> 新ID 的映射表（用于统计）
def build_class_map():
    class_map = {}
    missing = []
    for new_id, cls_name in enumerate(KEEP_CLASSES):
        if cls_name in ORIGINAL_CLASSES:
            old_id = ORIGINAL_CLASSES.index(cls_name)
            class_map[old_id] = new_id
        else:
            missing.append(cls_name)
    if missing:
        raise ValueError(f"以下类别在原始类别列表中不存在，请检查: {missing}")
    return class_map


def scan_and_count():
    class_map = build_class_map()
    # 用于计数：key = new_id, value = 标注框总数
    total_counts = defaultdict(int)

    print(f"正在深度扫描原始数据集: {SOURCE_DIR}")
    total_labels_processed = 0
    total_annotations_kept = 0

    # 递归遍历所有包含 'pic' 和 'yolo' 的文件夹对
    for root, dirs, files in os.walk(SOURCE_DIR):
        if "pic" in dirs:
            images_dir = os.path.join(root, "pic")
            labels_dir = os.path.join(root, "yolo")
            if not os.path.exists(labels_dir):
                print(f"警告: 在 {root} 发现 pic 但未发现配套的 yolo 文件夹，跳过")
                continue

            # 遍历 labels_dir 下的所有 .txt 文件
            for lbl_file in os.listdir(labels_dir):
                if not lbl_file.endswith(".txt") or lbl_file == "classes.txt":  # 添加排除条件
                    continue
                lbl_path = os.path.join(labels_dir, lbl_file)
                # 检查对应的图片是否存在（可选，保持严格）
                img_base = os.path.splitext(lbl_file)[0]
                img_path = None
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    candidate = os.path.join(images_dir, img_base + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path is None:
                    print(f"警告: 标签文件 {lbl_path} 没有找到对应的图片，跳过")
                    continue

                total_labels_processed += 1
                # 读取标签文件，统计保留类别的标注
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            try:
                                old_id = int(parts[0])
                                if old_id in class_map:
                                    new_id = class_map[old_id]
                                    total_counts[new_id] += 1
                                    total_annotations_kept += 1
                            except ValueError:
                                continue

    print(f"扫描完成！共处理 {total_labels_processed} 个标签文件，保留标注总数: {total_annotations_kept}")
    return total_counts


def main():
    counts = scan_and_count()

    # 输出统计结果到 txt 文件（放在当前目录或指定路径）
    output_file = os.path.join(os.path.dirname(__file__), "original_stats.txt")
    with open(output_file, "w") as f:
        f.write("Traffic Light Dataset Statistics (Original Data, Filtered Classes Only)\n")
        f.write(f"{'New ID':<8} {'Class Name':<30} {'Annotation Count':<15}\n")
        f.write("-" * 55 + "\n")
        for new_id, class_name in enumerate(KEEP_CLASSES):
            cnt = counts.get(new_id, 0)
            f.write(f"{new_id:<8} {class_name:<30} {cnt:<15}\n")
        f.write("-" * 55 + "\n")
        f.write(f"Total annotations (kept): {sum(counts.values())}\n")

    print(f"\n统计结果已保存至: {output_file}")
    # 控制台打印预览
    print("\n" + "=" * 55)
    print(f"{'New ID':<8} {'Class Name':<30} {'Count':<10}")
    print("-" * 55)
    for new_id, class_name in enumerate(KEEP_CLASSES):
        cnt = counts.get(new_id, 0)
        print(f"{new_id:<8} {class_name:<30} {cnt:<10}")
    print("=" * 55)


if __name__ == "__main__":
    main()
