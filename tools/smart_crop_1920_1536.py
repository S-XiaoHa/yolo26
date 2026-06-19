#!/usr/bin/env python3
"""
数据集处理脚本 v6 - 智能裁剪 + 多进程极速 + 数据追溯审计版
新增功能：
1. 统计 8M (3840x2160) 成功转换为 3M (1920x1536) 的数量
2. 统计并记录因红绿灯跨度过大而产生黑边 (Letterbox降级) 的具体图片路径
3. 输出日志文件 black_border_images.txt.
"""

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

# ================= 配置区域 =================
SOURCE_DIR = Path("/home/jzyh/xbzl/traffic_light/0825")
OUTPUT_DIR = Path("/home/jzyh/xbzl/traffic_light/data_v1_filtered")

TARGET_W, TARGET_H = 1920, 1536
TARGET_RATIO = TARGET_W / TARGET_H  # 1.25

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
    "turn_around_red",
    "turn_around_yellow",
    "turn_around_green",
]

CLASS_MAP = {ORIGINAL_CLASSES.index(cls): i for i, cls in enumerate(KEEP_CLASSES)}


def validate_bbox(x, y, w, h):
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        return False
    if w <= 0 or h <= 0:
        return False
    return True


def compute_smart_crop(boxes_normalized, orig_w, orig_h):
    ideal_h = orig_h  # 2160
    ideal_w = int(ideal_h * TARGET_RATIO)  # 2700

    if not boxes_normalized:
        cx1 = (orig_w - ideal_w) // 2
        return (cx1, 0, cx1 + ideal_w, orig_h)

    min_x = min(x - w / 2 for _, x, y, w, h in boxes_normalized) * orig_w
    max_x = max(x + w / 2 for _, x, y, w, h in boxes_normalized) * orig_w
    center_x = (min_x + max_x) / 2
    bbox_span = max_x - min_x
    req_w = bbox_span + orig_w * 0.03
    crop_w = max(ideal_w, req_w)
    crop_w = min(crop_w, orig_w)

    cx1 = int(center_x - crop_w / 2)
    cx2 = int(cx1 + crop_w)

    if cx1 < 0:
        cx1 = 0
        cx2 = int(crop_w)
    elif cx2 > orig_w:
        cx2 = orig_w
        cx1 = int(orig_w - crop_w)

    return (int(cx1), 0, int(cx2), orig_h)


def compute_letterbox_params(crop_w, crop_h):
    ratio = min(TARGET_W / crop_w, TARGET_H / crop_h)
    new_w = int(crop_w * ratio)
    new_h = int(crop_h * ratio)
    pad_left = (TARGET_W - new_w) // 2
    pad_top = (TARGET_H - new_h) // 2
    return ratio, pad_left, pad_top, new_w, new_h


# 将核心处理逻辑独立为一个可以被多进程调用的纯函数
def worker_task(args):
    img_path, label_path, out_img_path, out_label_path = args

    # 默认状态
    is_8m = False
    has_black_borders = False

    img = cv2.imread(str(img_path))
    if img is None:
        return ("error", False, False, str(img_path))

    orig_h, orig_w = img.shape[:2]

    # 判断是否为 8M 图像 (3840x2160)
    if orig_w == 3840 and orig_h == 2160:
        is_8m = True

    with open(str(label_path)) as f:
        lines = f.readlines()

    valid_boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_id = int(parts[0])
            if old_id in CLASS_MAP:
                new_id = CLASS_MAP[old_id]
                x, y, w, h = map(float, parts[1:5])
                if validate_bbox(x, y, w, h):
                    valid_boxes.append((new_id, x, y, w, h))
        except ValueError:
            continue

    if not valid_boxes:
        return ("empty", is_8m, False, str(img_path))

    x1, y1, x2, y2 = compute_smart_crop(valid_boxes, orig_w, orig_h)
    crop_w = x2 - x1
    crop_h = y2 - y1

    img_cropped = img[y1:y2, x1:x2]

    ratio, pad_left, pad_top, new_w, new_h = compute_letterbox_params(crop_w, crop_h)

    pad_right = TARGET_W - new_w - pad_left
    pad_bottom = TARGET_H - new_h - pad_top

    # 判定是否产生了黑边 (上下左右只要有任何一边大于0，即代表产生了退化黑边)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        has_black_borders = True

    img_resized = cv2.resize(img_cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    img_final = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    out_lines = []
    for new_id, x, y, w, h in valid_boxes:
        px, py = x * orig_w, y * orig_h
        pw, ph = w * orig_w, h * orig_h

        px_new = (px - x1) * ratio + pad_left
        py_new = (py - y1) * ratio + pad_top
        pw_new = pw * ratio
        ph_new = ph * ratio

        x_new = px_new / TARGET_W
        y_new = py_new / TARGET_H
        w_new = pw_new / TARGET_W
        h_new = ph_new / TARGET_H

        if validate_bbox(x_new, y_new, w_new, h_new):
            out_lines.append(f"{new_id} {x_new:.6f} {y_new:.6f} {w_new:.6f} {h_new:.6f}\n")

    if not out_lines:
        return ("empty", is_8m, False, str(img_path))

    cv2.imwrite(str(out_img_path), img_final)
    with open(str(out_label_path), "w") as f:
        f.writelines(out_lines)

    # 返回丰富的统计状态
    return ("success", is_8m, has_black_borders, str(img_path))


def main():
    print("=" * 60)
    print("Traffic Light Dataset Processor v6 (数据审计版🚀)")
    print("=" * 60)

    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 扩展了统计字典
    stats = {
        "success": 0,
        "empty": 0,
        "error": 0,
        "8m_converted": 0,  # 成功处理的8M图数量
        "black_borders": 0,  # 发生退化黑边的图片数量
    }

    # 用于记录黑边图片的路径
    black_border_files_log = []

    print(f"正在深度扫描 {SOURCE_DIR} ...")

    all_img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
        all_img_paths.extend(list(SOURCE_DIR.rglob(ext)))

    valid_pairs = []
    for img_path in all_img_paths:
        lbl_path = img_path.with_suffix(".txt")
        if not lbl_path.exists() and "pic" in img_path.parts:
            parts = list(img_path.parts)
            new_parts = ["yolo" if p == "pic" else p for p in parts]
            lbl_path = Path(*new_parts).with_suffix(".txt")

        if not lbl_path.exists() and "images" in img_path.parts:
            parts = list(img_path.parts)
            new_parts = ["labels" if p == "images" else p for p in parts]
            lbl_path = Path(*new_parts).with_suffix(".txt")

        if lbl_path.exists():
            valid_pairs.append((img_path, lbl_path))

    print(f"✅ 共找到 {len(valid_pairs)} 对有效的图像-标签文件。")
    if not valid_pairs:
        return

    random.seed(42)
    random.shuffle(valid_pairs)

    split_ratio = 0.85
    split_idx = int(len(valid_pairs) * split_ratio)
    datasets = {"train": valid_pairs[:split_idx], "val": valid_pairs[split_idx:]}

    # ================= 🚀 多进程狂飙引擎 =================
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"\n🚀 启动多进程加速引擎，分配核心数: {max_workers}")

    for split, pairs in datasets.items():
        print(f"\n处理 {split} 数据集 (共 {len(pairs)} 张)...")

        task_args = []
        for img_path, label_path in pairs:
            out_img = OUTPUT_DIR / "images" / split / img_path.name
            out_lbl = OUTPUT_DIR / "labels" / split / f"{img_path.stem}.txt"
            task_args.append((img_path, label_path, out_img, out_lbl))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_task, arg) for arg in task_args]

            # 解析多进程返回的丰富状态
            for future in tqdm(as_completed(futures), total=len(task_args), desc=f"{split} 处理"):
                status, is_8m, has_black_borders, orig_img_path = future.result()

                # 记录基础状态 (success/empty/error)
                stats[status] += 1

                # 记录成功案例中的详细指标
                if status == "success":
                    if is_8m:
                        stats["8m_converted"] += 1
                    if has_black_borders:
                        stats["black_borders"] += 1
                        black_border_files_log.append(orig_img_path)

    # 导出包含黑边的文件日志
    log_file_path = OUTPUT_DIR / "black_border_images.txt"
    with open(log_file_path, "w") as f:
        f.write(f"以下 {len(black_border_files_log)} 张图片因为红绿灯横跨过宽，触发了 Letterbox 退化（带有黑边）：\n")
        f.write("-" * 80 + "\n")
        for path in black_border_files_log:
            f.write(f"{path}\n")

    print("\n" + "=" * 60)
    print("⚡ 处理完成！统计概览:")
    print(f"  ✅ 成功处理总计: {stats['success']} 张")
    print(f"      ├─ 其中 8M (3840x2160) 成功缩放至 3M 的图片: {stats['8m_converted']} 张")
    print(f"      └─ 完美 0 黑边缩放图片: {stats['success'] - stats['black_borders']} 张")
    print(f"      └─ 妥协退化产生黑边的图片: {stats['black_borders']} 张")
    print(f"  ⚠ 无目标(标签全是被过滤类): {stats['empty']} 张")
    print(f"  ❌ 错误(图片损坏): {stats['error']} 张")
    print(f"\n📂 输出目录: {OUTPUT_DIR}")
    print(f"📄 黑边追溯日志已生成: {log_file_path}")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
