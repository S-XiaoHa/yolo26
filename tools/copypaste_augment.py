#!/usr/bin/env python3
"""
YOLO 智能按需 Copy-Paste 增强脚本 (V4 多进程加速 + 综合审计日志版).

核心改进：
1. 引入 ProcessPoolExecutor 实现多进程极速生成
2. 预分配文件名，彻底解决多进程下的文件名冲突风险
3. 进程级随机种子隔离，保证生成的样本高度多样化
python tools/copypaste_augment.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/output \
    --base_count 16402 \
    --ratios "1.0,0.2,1.0,0.2,0.15,0.2,0.6,0.15,0.4,0.2,0.15,0.3" \
    --max_iou 0.0 \
    --workers 0

"""

import argparse
import multiprocessing
import os
import random
from collections import defaultdict
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

    cls, mat_record_str, bg_lbl_path, img_dir, out_img_path, out_lbl_path, max_iou = args

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

    # 读取背景图片
    bg_stem = Path(bg_lbl_path).stem
    bg_img_path = find_image(img_dir, bg_stem)
    if bg_img_path is None:
        return "skipped"

    bg_img = cv2.imread(bg_img_path)
    if bg_img is None:
        return "skipped"

    bg_h, bg_w = bg_img.shape[:2]
    existing_boxes = parse_label_file(bg_lbl_path)

    # ---- 上下文感知：合法放置点寻找 ----
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

    # 策略 B：兜底策略
    for _ in range(10):
        candidates.append(
            (random.randint(0, max(1, bg_w - crop_w)), random.randint(0, max(1, int(bg_h * 0.5) - crop_h)))
        )

    # 遍历候选点，找到第一个合法位置执行粘贴
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

    if placed:
        cv2.imwrite(out_img_path, bg_img)
        write_label_file(out_lbl_path, existing_boxes)
        return "success"
    else:
        return "skipped"


# ================= 核心类 =================
class SmartAugmenterV4:
    def __init__(self, args):
        self.data_dir = Path(args.data_dir)
        self.out_dir = Path(args.output_dir)
        self.base_count = args.base_count
        self.max_iou = args.max_iou
        self.workers = args.workers

        ratio_strs = args.ratios.split(",")
        if len(ratio_strs) != 12:
            raise ValueError("必须严格提供 12 个类别的比例，用逗号分隔！")
        self.ratios = [float(r) for r in ratio_strs]

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    def scan_dataset(self):
        """统计数据集中每个类别的框数量."""
        counts = defaultdict(int)
        labels_dir = self.data_dir / "labels" / "train"
        label_files = list(labels_dir.glob("*.txt"))

        for lbl_file in label_files:
            if lbl_file.name == "classes.txt":
                continue
            for box in parse_label_file(str(lbl_file)):
                counts[box[0]] += 1
        return counts

    def generate_report(self, current_counts, deficit, after_counts, report_name="augmentation_report.txt"):
        """生成详细的审计对比日志."""
        total_before = sum(current_counts.values())
        total_after = sum(after_counts.values())

        with open(self.out_dir / report_name, "w", encoding="utf-8") as f:
            f.write("📊 数据集增强综合审计报告 (Augmentation Audit Report)\n")
            f.write("=" * 110 + "\n")
            f.write(f"基准数量 (Base Count): {self.base_count}\n")
            f.write(f"增强前总框数 (Train): {total_before}\n")
            f.write(f"预期增强后总数: {total_after} (+{total_after - total_before})\n")
            f.write(f"最大 IoU 阈值: {self.max_iou}\n")
            f.write("=" * 110 + "\n")

            header = f"{'ID':<3} | {'类别名':<15} | {'设定比例':<8} | {'目标数量':<8} | {'增强前数量':<10} | {'前比例':<8} | {'需增强(次)':<10} | {'增强后预期':<10} | {'后比例':<8}"
            f.write(header + "\n")
            f.write("-" * 110 + "\n")

            for cls_id in range(12):
                name = CURRENT_CLASSES[cls_id]
                ratio = self.ratios[cls_id]

                if ratio == 1.0:
                    target_str = "SKIP"
                else:
                    target_str = str(int(self.base_count * ratio))

                before_cnt = current_counts[cls_id]
                before_pct = f"{(before_cnt / total_before * 100):.1f}%" if total_before > 0 else "0%"
                need_aug = deficit[cls_id]
                after_cnt = after_counts[cls_id]
                after_pct = f"{(after_cnt / total_after * 100):.1f}%" if total_after > 0 else "0%"

                line = f"{cls_id:<3} | {name:<15} | {ratio:<8.2f} | {target_str:<8} | {before_cnt:<10} | {before_pct:<8} | {need_aug:<10} | {after_cnt:<10} | {after_pct:<8}"
                f.write(line + "\n")
            f.write("=" * 110 + "\n")

    def build_material_library(self, classes_to_aug):
        """构建素材库索引."""
        print("\n🔍 正在扫描并构建素材库索引...")
        library_path = self.out_dir / "material_library.txt"
        materials = {cls: [] for cls in classes_to_aug}

        labels_dir = self.data_dir / "labels" / "train"
        img_dir = self.data_dir / "images" / "train"

        with open(library_path, "w", encoding="utf-8") as f:
            f.write("class_id,img_path,x_center,y_center,w,h\n")
            for lbl_file in labels_dir.glob("*.txt"):
                if lbl_file.name == "classes.txt":
                    continue

                img_path = find_image(img_dir, lbl_file.stem)
                if img_path is None:
                    continue

                for cls, x, y, w, h in parse_label_file(str(lbl_file)):
                    if cls in classes_to_aug:
                        record = f"{cls},{img_path},{x},{y},{w},{h}"
                        f.write(record + "\n")
                        materials[cls].append(record)
        print(f"✅ 素材库索引已生成并保存至: {library_path}")
        return materials

    def run(self):
        print("📊 正在统计原始数据集分布...")
        current_counts = self.scan_dataset()

        if self.base_count <= 0:
            self.base_count = max(current_counts.values())
            print(f"自动推断 Base Count 为当前最大类数量: {self.base_count}")

        # 计算差值
        deficit = {}
        after_counts = current_counts.copy()
        classes_to_aug = []

        for cls in range(12):
            ratio = self.ratios[cls]
            current = current_counts[cls]

            if ratio == 1.0:
                deficit[cls] = 0
                after_counts[cls] = current
                continue

            target = int(self.base_count * ratio)
            need = max(0, target - current)
            deficit[cls] = need
            after_counts[cls] = current + need
            if need > 0:
                classes_to_aug.append(cls)

        self.generate_report(current_counts, deficit, after_counts, "augmentation_report.txt")
        print(f"📄 增强计划报告已生成: {self.out_dir / 'augmentation_report.txt'}")

        if not classes_to_aug:
            print("🎉 当前数据已全部满足设定比例，无需增强！")
            return

        # 构建素材库
        materials = self.build_material_library(classes_to_aug)
        bg_files = list((self.data_dir / "labels" / "train").glob("*.txt"))

        # 构建任务列表
        task_args = []
        global_task_counter = 0

        print("\n📦 正在构建并发任务队列...")
        for cls in classes_to_aug:
            needed = deficit[cls]
            if not materials[cls]:
                print(f"⚠️ 类别 {CURRENT_CLASSES[cls]} 没有源素材，跳过！")
                continue

            for _ in range(needed):
                global_task_counter += 1
                mat_record_str = random.choice(materials[cls])
                bg_lbl_path = str(random.choice(bg_files))
                img_dir = str(self.data_dir / "images" / "train")

                # 预分配唯一的文件名
                out_name = f"aug_cls{cls}_{global_task_counter:06d}"
                out_img_path = str(self.out_dir / "images" / "train" / f"{out_name}.jpg")
                out_lbl_path = str(self.out_dir / "labels" / "train" / f"{out_name}.txt")

                task_args.append((cls, mat_record_str, bg_lbl_path, img_dir, out_img_path, out_lbl_path, self.max_iou))

        # 决定进程数
        if self.workers <= 0:
            max_workers = max(1, multiprocessing.cpu_count() - 2)
        else:
            max_workers = self.workers

        print(f"\n🚀 启动 V4 多进程增强引擎 (分配核心数: {max_workers})")

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

        print(f"\n✨ 增强结束！共成功生成 {total_generated} 张新图片。")
        if total_skipped > 0:
            print(f"⚠️ 放弃了 {total_skipped} 个样本（可能由于 IoU 限制或读取失败）")
        print(f"📁 详细审计报告与增强图集位于: {self.out_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="YOLO 智能按需 Copy-Paste 增强脚本 (V4)")
    parser.add_argument("--data_dir", type=str, required=True, help="原始数据集目录")
    parser.add_argument("--output_dir", type=str, required=True, help="增强数据输出目录")
    parser.add_argument("--base_count", type=int, default=16402, help="计算目标的基准数量(默认使用round_green的量)")
    parser.add_argument(
        "--ratios",
        type=str,
        default="1.0,0.2,1.0,0.2,0.15,0.2,0.6,0.15,0.4,0.2,0.15,0.3",
        help="严格对应0-11类的比例，用逗号分隔",
    )
    parser.add_argument("--max_iou", type=float, default=0.0, help="最大允许的 IoU 阈值 (默认 0.0)")
    parser.add_argument("--workers", type=int, default=0, help="并行进程数 (0=自动推断, 1=强制单线程)")

    args = parser.parse_args()
    augmenter = SmartAugmenterV4(args)
    augmenter.run()
