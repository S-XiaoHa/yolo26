#!/usr/bin/env python3
"""
YOLO 智能按需 Copy-Paste 增强脚本 V5 - 指定素材目录 + 直接指定增强数量版.

核心改动：
1. 从指定的素材目录 (--source_dir) 中读取要粘贴的目标物体
2. 背景图片从原始数据集 (--data_dir) 中随机选取
3. 直接使用 --augment "class_id:num,class_id:num,..." 指定每个类别要增强的数量
4. 保留多进程加速功能
python tools/copypaste_augment_v5.py \
    --source_dir /path/to/source_dataset \
    --data_dir /path/to/original_dataset \
    --output_dir /path/to/output \
    --augment "1:500,4:300,7:200"

"""

import argparse
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

# ================= 类别定义 =================
CURRENT_CLASSES = [
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


# ================= 工具函数 =================
def xywh_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    """将归一化的 (x_center, y_center, w, h) 转换为像素坐标的 (x1, y1, x2, y2)."""
    cx, cy = x_center * img_w, y_center * img_h
    bw, bh = w * img_w, h * img_h
    return (
        max(0, int(cx - bw / 2)),
        max(0, int(cy - bh / 2)),
        min(img_w, int(cx + bw / 2)),
        min(img_h, int(cy + bh / 2)),
    )


def xyxy_to_xywh_norm(x1, y1, x2, y2, img_w, img_h):
    """将像素坐标的 (x1, y1, x2, y2) 转换为归一化的 (x_center, y_center, w, h)."""
    return ((x1 + x2) / 2) / img_w, ((y1 + y2) / 2) / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h


def compute_iou(box1, box2):
    """计算两个框的 IoU."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    inter_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    union_area = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter_area
    return inter_area / union_area if union_area > 0 else 0


def parse_label_file(label_path):
    """解析 YOLO 格式的标签文件."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def write_label_file(label_path, boxes):
    """写入 YOLO 格式的标签文件."""
    with open(label_path, "w") as f:
        for cls, x_center, y_center, w, h in boxes:
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def find_image(img_dir, stem):
    """查找匹配的图片文件，支持多种扩展名."""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        img_path = Path(img_dir) / f"{stem}{ext}"
        if img_path.exists():
            return str(img_path)
    return None


# ================= 多进程 Worker =================
def worker_task(args):
    """独立的进程工作单元：生成一张增强样本."""
    # 强制隔离随机种子，避免 fork 模式下多进程随机序列一致
    random.seed(os.getpid() + int(cv2.getTickCount()))

    cls, mat_record_str, bg_lbl_path, bg_img_dir, out_img_path, out_lbl_path, max_iou = args

    # --- 1. 裁剪源物体 ---
    mat_record = mat_record_str.split(",")
    src_img = cv2.imread(mat_record[1])
    if src_img is None:
        return "skipped"

    sh, sw = src_img.shape[:2]
    src_x, src_y, src_w, src_h = map(float, mat_record[2:])

    x1, y1, x2, y2 = xywh_to_xyxy(src_x, src_y, src_w, src_h, sw, sh)
    crop = src_img[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return "skipped"

    # --- 2. 读取背景图片 ---
    bg_stem = Path(bg_lbl_path).stem
    bg_img_path = find_image(bg_img_dir, bg_stem)
    if bg_img_path is None:
        return "skipped"

    bg_img = cv2.imread(bg_img_path)
    if bg_img is None:
        return "skipped"

    bg_h, bg_w = bg_img.shape[:2]
    existing_boxes = parse_label_file(bg_lbl_path)

    # --- 3. 上下文感知：合法放置点寻找 ---
    placed = False
    candidates = []
    margin = 5

    # 策略 A：伴随现有红绿灯生成
    if existing_boxes:
        for ex_cls, ex_x, ex_y, ex_w, ex_h in existing_boxes:
            ex_x1, ex_y1, ex_x2, ex_y2 = xywh_to_xyxy(ex_x, ex_y, ex_w, ex_h, bg_w, bg_h)
            candidates.append((ex_x2 + margin, ex_y1))
            candidates.append((ex_x1 - crop_w - margin, ex_y1))
            candidates.append((ex_x1, ex_y2 + margin))
            candidates.append((ex_x1, ex_y1 - crop_h - margin))

    random.shuffle(candidates)

    # 策略 B：兜底策略（在画面上半部分随机找）
    for _ in range(10):
        candidates.append(
            (random.randint(0, max(1, bg_w - crop_w)), random.randint(0, max(1, int(bg_h * 0.5) - crop_h)))
        )

    # --- 4. 遍历候选点，找到第一个合法位置执行粘贴 ---
    for cx, cy in candidates:
        if cx < 0 or cy < 0 or cx + crop_w > bg_w or cy + crop_h > bg_h:
            continue

        new_bbox = (cx, cy, cx + crop_w, cy + crop_h)

        overlap = False
        for ex_cls, ex_x, ex_y, ex_w, ex_h in existing_boxes:
            ex_bbox = xywh_to_xyxy(ex_x, ex_y, ex_w, ex_h, bg_w, bg_h)
            if compute_iou(new_bbox, ex_bbox) > max_iou:
                overlap = True
                break

        if not overlap:
            bg_img[cy : cy + crop_h, cx : cx + crop_w] = crop
            new_label = xyxy_to_xywh_norm(cx, cy, cx + crop_w, cy + crop_h, bg_w, bg_h)
            existing_boxes.append((cls, *new_label))
            placed = True
            break

    # --- 5. 保存结果 ---
    if placed:
        cv2.imwrite(out_img_path, bg_img)
        write_label_file(out_lbl_path, existing_boxes)
        return "success"
    else:
        return "skipped"


# ================= 核心类 =================
class SmartAugmenterV5:
    def __init__(self, args):
        self.source_dir = Path(args.source_dir)  # 素材目录（要粘贴的目标物体来源）
        self.data_dir = Path(args.data_dir)  # 原始数据集目录（背景图片来源）
        self.out_dir = Path(args.output_dir)
        self.max_iou = args.max_iou
        self.workers = args.workers

        # 解析增强配置: "class_id:num,class_id:num,..."
        # 例如: "1:500,4:300,7:200"
        self.augment_targets = parse_augment_config(args.augment)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    def build_source_library(self, classes_to_aug):
        """从指定的素材目录构建树材库索引."""
        print(f"\n🔍 正在扫描素材目录: {self.source_dir}")
        materials = {cls: [] for cls in classes_to_aug}

        source_labels_dir = self.source_dir / "labels" / "train"
        source_img_dir = self.source_dir / "images" / "train"

        # 如果素材目录结构不同，尝试其他路径
        if not source_labels_dir.exists():
            source_labels_dir = self.source_dir / "labels"
            source_img_dir = self.source_dir / "images"

        if not source_labels_dir.exists():
            print(f"❌ 素材目录不存在: {source_labels_dir}")
            return materials

        for lbl_file in source_labels_dir.glob("*.txt"):
            if lbl_file.name == "classes.txt":
                continue

            img_path = find_image(source_img_dir, lbl_file.stem)
            if img_path is None:
                continue

            for cls, x, y, w, h in parse_label_file(str(lbl_file)):
                if cls in classes_to_aug:
                    record = f"{cls},{img_path},{x},{y},{w},{h}"
                    materials[cls].append(record)

        total = sum(len(v) for v in materials.values())
        print(f"✅ 素材库构建完成，共找到 {total} 个可用素材")
        for cls in classes_to_aug:
            if materials[cls]:
                print(f"   类别 {cls} ({CURRENT_CLASSES[cls]}): {len(materials[cls])} 个素材")

        return materials

    def get_background_files(self):
        """从原始数据集中获取背景图片列表."""
        print(f"\n📋 正在扫描背景图片目录: {self.data_dir}")
        bg_labels_dir = self.data_dir / "labels" / "train"
        bg_files = []

        for lbl_file in bg_labels_dir.glob("*.txt"):
            if lbl_file.name == "classes.txt":
                continue
            bg_files.append(str(lbl_file))

        print(f"✅ 共找到 {len(bg_files)} 张背景图片")
        return bg_files

    def run(self):
        # 打印增强计划
        print("=" * 60)
        print("📊 YOLO 智能 Copy-Paste 增强脚本 V5")
        print("=" * 60)
        print(f"📁 素材目录 (源物体): {self.source_dir}")
        print(f"📁 背景目录 (原始数据集): {self.data_dir}")
        print(f"📁 输出目录: {self.out_dir}")
        print("🎯 增强目标:")

        classes_to_aug = []
        total_needed = 0
        for cls, count in self.augment_targets:
            class_name = CURRENT_CLASSES[cls] if cls < len(CURRENT_CLASSES) else f"未知类({cls})"
            print(f"   类别 {cls} ({class_name}): 增强 {count} 个")
            classes_to_aug.append(cls)
            total_needed += count

        print(f"   合计: {total_needed} 个增强样本")
        print(f"⚙️  最大 IoU 阈值: {self.max_iou}")
        print("=" * 60)

        if not classes_to_aug:
            print("🎉 无需增强！")
            return

        # 构建树材库（从指定素材目录）
        materials = self.build_source_library(classes_to_aug)

        # 检查素材是否充足
        for cls in classes_to_aug:
            needed = self.augment_targets[cls]
            available = len(materials[cls])
            if available == 0:
                print(f"⚠️ 类别 {cls} ({CURRENT_CLASSES[cls]}) 在素材目录中没有找到素材，跳过！")
            elif available < needed:
                print(
                    f"⚠️ 类别 {cls} ({CURRENT_CLASSES[cls]}) 素材不足: 需要 {needed} 个，只有 {available} 个（会重复使用）"
                )

        # 获取背景图片列表（从原始数据集）
        bg_files = self.get_background_files()
        if not bg_files:
            print("❌ 背景图片目录为空，无法继续！")
            return

        # 构建任务列表
        task_args = []
        global_task_counter = 0

        print("\n📦 正在构建并发任务队列...")
        for cls in classes_to_aug:
            needed = self.augment_targets[cls]
            if not materials[cls]:
                continue

            for _ in range(needed):
                global_task_counter += 1
                # 从素材库中随机选择一个素材
                mat_record_str = random.choice(materials[cls])
                # 从原始数据集中随机选择一张背景图
                bg_lbl_path = random.choice(bg_files)
                bg_img_dir = str(self.data_dir / "images" / "train")

                # 预分配唯一的文件名
                out_name = f"aug_cls{cls}_{global_task_counter:06d}"
                out_img_path = str(self.out_dir / "images" / "train" / f"{out_name}.jpg")
                out_lbl_path = str(self.out_dir / "labels" / "train" / f"{out_name}.txt")

                task_args.append(
                    (cls, mat_record_str, bg_lbl_path, bg_img_dir, out_img_path, out_lbl_path, self.max_iou)
                )

        print(f"✅ 任务队列构建完成: {len(task_args)} 个任务")

        # 决定进程数
        if self.workers <= 0:
            max_workers = max(1, multiprocessing.cpu_count() - 2)
        else:
            max_workers = self.workers

        print(f"\n🚀 启动 V5 多进程增强引擎 (分配核心数: {max_workers})")

        total_generated = 0
        total_skipped = 0

        # 执行任务
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_task, arg) for arg in task_args]

            for future in tqdm(as_completed(futures), total=len(task_args), desc="增强进度"):
                status = future.result()
                if status == "success":
                    total_generated += 1
                else:
                    total_skipped += 1

        # 打印结果统计
        print("\n" + "=" * 60)
        print("✨ 增强完成！结果统计:")
        print("=" * 60)
        print(f"✅ 成功生成: {total_generated} 张")
        if total_skipped > 0:
            print(f"⚠️ 放弃跳过: {total_skipped} 个（IoU 限制或读取失败）")
        print(f"📊 成功率: {total_generated / len(task_args) * 100:.1f}%" if task_args else "N/A")
        print(f"\n📁 增强数据已保存至: {self.out_dir}")
        print(f"   图片: {self.out_dir / 'images' / 'train'}")
        print(f"   标签: {self.out_dir / 'labels' / 'train'}")
        print("=" * 60)


# ================= 辅助函数 =================
def parse_augment_config(augment_str):
    """解析增强配置字符串 格式: "class_id:num,class_id:num,..." 示例: "1:500,4:300,7:200" 返回: [(class_id, num), ...].
    """
    targets = {}
    parts = augment_str.split(",")
    for part in parts:
        part = part.strip()
        if ":" not in part:
            print(f"⚠️ 跳过无效配置: {part} (格式应为 class_id:num)")
            continue
        cls_str, num_str = part.split(":", 1)
        try:
            cls_id = int(cls_str.strip())
            num = int(num_str.strip())
            if cls_id in targets:
                print(f"⚠️ 类别 {cls_id} 重复指定，使用最新的值: {num}")
            targets[cls_id] = num
        except ValueError:
            print(f"⚠️ 跳过无效配置: {part}")

    return [(cls, num) for cls, num in targets.items()]


# ================= 主入口 =================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="YOLO 智能按需 Copy-Paste 增强脚本 V5 - 指定素材目录版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法：从素材目录取目标，贴到原始数据集图片上
  python tools/copypaste_augment_v5.py \\
      --source_dir /path/to/source_dataset \\
      --data_dir /path/to/original_dataset \\
      --output_dir /path/to/output \\
      --augment "1:500,4:300,7:200"

  # 参数说明:
  # --augment "1:500,4:300,7:200" 表示:
  #   类别1 (round_yellow): 增强 500 个
  #   类别4 (up_yellow):    增强 300 个
  #   类别7 (left_yellow):  增强 200 个

  #
        """,
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="素材目录：从中提取要粘贴的目标物体（需包含 images/train 和 labels/train 子目录）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="原始数据集目录：从中随机选取背景图片（需包含 images/train 和 labels/train 子目录）",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="增强数据输出目录")
    parser.add_argument(
        "--augment",
        type=str,
        required=True,
        help='增强配置，格式: "class_id:num,class_id:num,..."  示例: "1:500,4:300,7:200"',
    )
    parser.add_argument("--max_iou", type=float, default=0.0, help="最大允许的 IoU 阈值 (默认 0.0，即不允许任何重叠)")
    parser.add_argument("--workers", type=int, default=0, help="并行进程数 (0=自动推断为 cpu_count-2, 1=强制单线程)")

    args = parser.parse_args()
    augmenter = SmartAugmenterV5(args)
    augmenter.run()
