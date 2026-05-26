#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
已清洗数据集统计脚本
直接扫描 data_v1_filtered/labels 下的 train 和 val 文件夹
统计当前 12 个类别的真实框（BBox）数量
"""

import os
from collections import defaultdict
from tqdm import tqdm

# ================= 配置区域 =================
# 已经清洗完毕、可直接用于训练的数据集目录
DATASET_DIR = "/home/jzyh/xbzl/traffic_light/data_v1_filtered"
# DATASET_DIR = "/home/jzyh/code/ultralytics/data_augment"


# 当前实际保留的 13 个类别（严格对应你数据集里 0~12 的 ID）
CURRENT_CLASSES = [
    "round_red", "round_yellow", "round_green",
    "up_red", "up_yellow", "up_green",
    "left_red", "left_yellow", "left_green",
    "right_red", "right_yellow", "right_green",
    "turn_around_red", "turn_around_yellow", "turn_around_green"
]

def scan_and_count():
    # counts 结构: { phase: { class_id: count } }
    counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "total": defaultdict(int)
    }
    
    total_labels_processed = 0
    phases = ["train", "val"]

    for phase in phases:
        labels_dir = os.path.join(DATASET_DIR, "labels", phase)
        if not os.path.exists(labels_dir):
            print(f"警告: 未找到 {phase} 标签目录 {labels_dir}")
            continue
            
        label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt") and f != "classes.txt"]
        
        for lbl_file in tqdm(label_files, desc=f"扫描 {phase} 集", unit="文件"):
            lbl_path = os.path.join(labels_dir, lbl_file)
            total_labels_processed += 1
            
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # 标准 YOLO 格式至少 5 列 (id x y w h)
                        try:
                            cls_id = int(parts[0])
                            # 累加计数
                            counts[phase][cls_id] += 1
                            counts["total"][cls_id] += 1
                        except ValueError:
                            continue
                            
    return counts, total_labels_processed

def main():
    print(f"开始扫描数据集: {DATASET_DIR}\n")
    counts, total_files = scan_and_count()
    
    # 检查是否有越界 ID（用于防错）
    out_of_bounds = {k: v for k, v in counts["total"].items() if k >= len(CURRENT_CLASSES)}
    if out_of_bounds:
        print(f"\n[⚠️ 严重警告] 发现超出 0-12 范围的异常类别 ID: {out_of_bounds}")
        print("请检查清洗脚本是否正确执行！\n")

    # 输出统计结果到 txt 文件
    output_file = os.path.join(os.path.dirname(__file__), "current_dataset_stats.txt")
    with open(output_file, 'w') as f:
        f.write("Traffic Light Dataset Statistics (Filtered 13 Classes)\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'ID':<4} | {'Class Name':<20} | {'Train':<10} | {'Val':<10} | {'Total':<10}\n")
        f.write("-" * 65 + "\n")
        
        for cls_id, class_name in enumerate(CURRENT_CLASSES):
            train_cnt = counts["train"].get(cls_id, 0)
            val_cnt = counts["val"].get(cls_id, 0)
            total_cnt = counts["total"].get(cls_id, 0)
            f.write(f"{cls_id:<4} | {class_name:<20} | {train_cnt:<10} | {val_cnt:<10} | {total_cnt:<10}\n")
            
        f.write("-" * 65 + "\n")
        f.write(f"Total Annotations: Train={sum(counts['train'].values())}, Val={sum(counts['val'].values())}, All={sum(counts['total'].values())}\n")
        f.write(f"Total Label Files Parsed: {total_files}\n")

    # 控制台打印预览
    print("\n" + "=" * 65)
    print(f"{'ID':<4} | {'Class Name':<20} | {'Train':<10} | {'Val':<10} | {'Total':<10}")
    print("-" * 65)
    for cls_id, class_name in enumerate(CURRENT_CLASSES):
        train_cnt = counts["train"].get(cls_id, 0)
        val_cnt = counts["val"].get(cls_id, 0)
        total_cnt = counts["total"].get(cls_id, 0)
        print(f"{cls_id:<4} | {class_name:<20} | {train_cnt:<10} | {val_cnt:<10} | {total_cnt:<10}")
    print("=" * 65)
    
    print(f"\n扫描完成！共解析 {total_files} 个文件。")
    print(f"总计保留有效红绿灯框: {sum(counts['total'].values())} 个")
    print(f"统计报告已保存至: {output_file}")

if __name__ == "__main__":
    main()