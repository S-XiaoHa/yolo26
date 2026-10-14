#!/usr/bin/env python3
"""
Traffic Light 数据集整合处理脚本 - 智能裁剪 + Letterbox 版.

核心改进：
2. 自动丢弃空标签：重映射或过滤后，若标签文件为空，则不拷贝图片和标签，并在终端输出提示。
3. 终端实时反馈：输出无效标注的原因及被删除的帧。
4. 使用 tqdm 显示实时处理进度条。
"""

import os
import random
import shutil

import cv2
from tqdm import tqdm  # 新增导入

# ================= 配置区域 =================
SOURCE_DIR = "/home/jzyh/xbzl/traffic_light/0825"
OUTPUT_DIR = "/home/jzyh/xbzl/traffic_light/data_v1_filtered"

TARGET_W, TARGET_H = 1920, 1536

ORIGINAL_CLASSES = [
    "round_black",
    "round_red",
    "round_yellow",
    "round_green",
    "up_black",
    "up_red",
    "up_yellow",
    "up_green",
    "down_black",
    "down_red",
    "down_yellow",
    "down_green",
    "left_black",
    "left_red",
    "left_yellow",
    "left_green",
    "right_black",
    "right_red",
    "right_yellow",
    "right_green",
    "turn_around_black",
    "turn_around_red",
    "turn_around_yellow",
    "turn_around_green",
    "prohibit_black",
    "prohibit_red",
    "prohibit_yellow",
    "prohibit_green",
    "countdown_black",
    "countdown_red",
    "countdown_yellow",
    "countdown_green",
    "script_black",
    "script_red",
    "script_yellow",
    "script_green",
    "unknown",
]

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

# 检查缺失类别
missing = [cls for cls in KEEP_CLASSES if cls not in ORIGINAL_CLASSES]
if missing:
    raise ValueError(f"以下类别在原始类别列表中不存在，请检查: {missing}")

CLASS_MAP = {ORIGINAL_CLASSES.index(cls): i for i, cls in enumerate(KEEP_CLASSES)}


def validate_bbox(x, y, w, h, img_name):
    """校验坐标是否合法."""
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        print(f"  [无效坐标] 越界: {img_name} -> x:{x:.3f}, y:{y:.3f}, w:{w:.3f}, h:{h:.3f}")
        return False
    if w <= 0 or h <= 0:
        print(f"  [无效几何] 宽高<=0: {img_name}")
        return False
    return True


def process_image_and_label(img_path, label_path, out_img_path, out_label_path):
    """处理单张图片和对应的标签文件 返回值: 处理状态 ("success", "empty", "error").
    """
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [错误] 无法读取图片: {img_path}")
        return "error"

    orig_h, orig_w = img.shape[:2]

    with open(label_path) as f:
        lines = f.readlines()

    out_lines = []
    need_resize = not (orig_w == TARGET_W and orig_h == TARGET_H)

    # Letterbox 转换参数 (严谨整型对齐版)
    ratio, dw, dh = 1.0, 0, 0
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0

    if need_resize:
        ratio = min(TARGET_H / orig_h, TARGET_W / orig_w)
        new_unpad = (round(orig_w * ratio), round(orig_h * ratio))

        pad_w = TARGET_W - new_unpad[0]
        pad_h = TARGET_H - new_unpad[1]

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        dw = pad_left
        dh = pad_top

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            old_id = int(parts[0])
            if old_id in CLASS_MAP:
                new_id = CLASS_MAP[old_id]
                x, y, w, h = map(float, parts[1:5])

                if need_resize:
                    x = (x * orig_w * ratio + dw) / TARGET_W
                    y = (y * orig_h * ratio + dh) / TARGET_H
                    w = (w * orig_w * ratio) / TARGET_W
                    h = (h * orig_h * ratio) / TARGET_H

                if validate_bbox(x, y, w, h, img_name):
                    out_lines.append(f"{new_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except ValueError:
            print(f"  [错误] 标签含非数字内容: {img_name}")
            continue

    if not out_lines:
        print(f"  [过滤] 标签为空(无匹配类或无效标注)，已丢弃该帧: {img_name}")
        return "empty"

    # 写入图片和标签
    if need_resize:
        out_img_path = os.path.splitext(out_img_path)[0] + ".png"
        new_unpad = (round(orig_w * ratio), round(orig_h * ratio))
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img_final = cv2.copyMakeBorder(
            img_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        cv2.imwrite(out_img_path, img_final)
    else:
        shutil.copy2(img_path, out_img_path)

    with open(out_label_path, "w") as f:
        f.writelines(out_lines)

    return "success"


def organize_dataset():
    # 清理并重建输出目录
    if os.path.exists(OUTPUT_DIR):
        print(f"正在清理旧目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, p), exist_ok=True)

    random.seed(42)
    all_files = []
    print(f"正在深度扫描源目录: {SOURCE_DIR}")

    for root, dirs, files in os.walk(SOURCE_DIR):
        if "pic" in dirs:
            images_dir = os.path.join(root, "pic")
            labels_dir = os.path.join(root, "yolo")
            if not os.path.exists(labels_dir):
                continue

            for f in os.listdir(images_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(images_dir, f)
                    lbl_path = os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt")
                    if os.path.exists(lbl_path):
                        all_files.append({"image": img_path, "label": lbl_path})

    print(f"扫描完毕！初始数量: {len(all_files)}")

    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.85)
    phases = {"train": all_files[:split_idx], "val": all_files[split_idx:]}

    stats = {"success": 0, "empty": 0, "error": 0}

    for phase, files in phases.items():
        print(f"\n--- 开始处理 {phase} 集 (共 {len(files)} 张) ---")
        # 使用 tqdm 包装 files，自动显示进度条
        for f_info in tqdm(files, desc=f"  {phase} 进度", unit="张"):
            out_img = os.path.join(OUTPUT_DIR, "images", phase, os.path.basename(f_info["image"]))
            out_lbl = os.path.join(OUTPUT_DIR, "labels", phase, os.path.basename(f_info["label"]))

            status = process_image_and_label(f_info["image"], f_info["label"], out_img, out_lbl)
            stats[status] += 1

    # 生成 classes.txt
    for p in ["train", "val"]:
        with open(os.path.join(OUTPUT_DIR, "labels", p, "classes.txt"), "w") as f:
            f.write("\n".join(KEEP_CLASSES))

    print(f"\n{'=' * 40}")
    print("处理完成！")
    print(f"成功导出: {stats['success']} 帧")
    print(f"因标签为空/无效丢弃: {stats['empty']} 帧")
    print(f"因读取错误失败: {stats['error']} 帧")
    print(f"数据集路径: {OUTPUT_DIR}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    organize_dataset()
