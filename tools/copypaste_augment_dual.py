#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 智能按需 Copy-Paste 增强脚本 (V20 宏观属性分析防过拟合版)

核心改进：
1. 属性解耦分析：在报告中额外输出“形状(Shapes)”与“颜色(Colors)”两大宏观属性的分布变化。
2. 保守型防过拟合配比：全线降低长尾增强倍率(控制在2~5倍)，防止过度生成导致的特征过拟合。
3. 纯净度优先级队列：优先消耗 0 大类的极品背景，大类零膨胀。
4. 增量合并机制：安全后缀命名，直接合并至原训练集，触发对比正则化。
"""

import argparse
import os
import random
import cv2
import numpy as np
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# ================= 类别定义 =================
CURRENT_CLASSES = [
    "round_red", "round_yellow", "round_green",
    "up_red", "up_yellow", "up_green",
    "left_red", "left_yellow", "left_green",
    "right_red", "right_yellow", "right_green",
    "turn_around_red", "turn_around_yellow", "turn_around_green"
]

MAX_PASTES_PER_IMG = 5

# ================= 工具函数 =================
def xywh_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    cx, cy = x_center * img_w, y_center * img_h
    bw, bh = w * img_w, h * img_h
    return max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)), min(img_w, int(cx + bw / 2)), min(img_h, int(cy + bh / 2))

def xyxy_to_xywh_norm(x1, y1, x2, y2, img_w, img_h):
    return ((x1 + x2) / 2) / img_w, ((y1 + y2) / 2) / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h

def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    inter_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    union_area = (x2_1 - x1_1)*(y2_1 - y1_1) + (x2_2 - x1_2)*(y2_2 - y1_2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def parse_label_file(label_path):
    boxes = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    except Exception:
        pass
    return boxes

def write_label_file(label_path, boxes):
    with open(label_path, "w") as f:
        for cls, x_center, y_center, w, h in boxes:
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def is_vertical(w, h):
    return h > w * 1.2

def parse_single_file_for_count(file_path):
    counts = defaultdict(int)
    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    counts[int(parts[0])] += 1
    except Exception:
        pass
    return (file_path, counts)

# ================= 多进程 Worker =================
def worker_task(args):
    chunk, bg_sample_pairs, out_dir_str, max_iou, task_id = args
    random.seed(os.getpid() + int(cv2.getTickCount()))
    
    parsed_materials = []
    for cls, mat_record_str in chunk:
        mat_record = mat_record_str.split(',')
        src_img = cv2.imread(mat_record[1])
        if src_img is None: continue
        sh, sw = src_img.shape[:2]
        src_x, src_y, src_w, src_h = map(float, mat_record[2:])
        x1, y1, x2, y2 = xywh_to_xyxy(src_x, src_y, src_w, src_h, sw, sh)
        crop = src_img[y1:y2, x1:x2].copy()
        crop_h, crop_w = crop.shape[:2]
        if crop_h > 0 and crop_w > 0:
            parsed_materials.append({
                "cls": cls, "crop": crop, 
                "h": crop_h, "w": crop_w, 
                "is_vert": is_vertical(crop_w, crop_h),
                "area": crop_w * crop_h
            })

    if not parsed_materials: return []
    margin = 10 

    for bg_lbl_path, bg_img_path in bg_sample_pairs:
        existing_boxes = parse_label_file(bg_lbl_path)
        if not existing_boxes: continue 
            
        bg_img = cv2.imread(bg_img_path)
        if bg_img is None: continue
        bg_h, bg_w = bg_img.shape[:2]
        
        success_this_bg = []
        for mat in parsed_materials:
            p1_candidates = []
            p2_candidates = []
            
            for ex_cls, ex_x, ex_y, ex_w, ex_h in existing_boxes:
                ex_h_px = int(ex_h * bg_h)
                ex_w_px = int(ex_w * bg_w)
                if ex_h_px == 0 or ex_w_px == 0: continue
                ex_is_vert = is_vertical(ex_w_px, ex_h_px)
                ex_area = ex_w_px * ex_h_px
                ex_x1, ex_y1, ex_x2, ex_y2 = xywh_to_xyxy(ex_x, ex_y, ex_w, ex_h, bg_w, bg_h)
                
                if mat["is_vert"] == ex_is_vert:
                    scale_ratio = ex_h_px / mat["h"]
                    if 0.65 <= scale_ratio <= 1.45:
                        p1_candidates.append((ex_x2 + margin, ex_y1))
                        p1_candidates.append((ex_x1 - mat["w"] - margin, ex_y1))
                else:
                    area_ratio = ex_area / mat["area"]
                    if 0.5 <= area_ratio <= 2.0:
                        p2_candidates.append((ex_x2 + margin, ex_y1))
                        p2_candidates.append((ex_x1 - mat["w"] - margin, ex_y1))

            placed = False
            for cands in [p1_candidates, p2_candidates]:
                if placed or not cands: continue
                random.shuffle(cands)
                for cx, cy in cands:
                    if cx < 0 or cy < 0 or cx + mat["w"] > bg_w or cy + mat["h"] > bg_h: continue
                    new_bbox = (cx, cy, cx + mat["w"], cy + mat["h"])
                    
                    overlap = False
                    for ex_cls, ex_x, ex_y, ex_w, ex_h in existing_boxes:
                        ex_bbox = xywh_to_xyxy(ex_x, ex_y, ex_w, ex_h, bg_w, bg_h)
                        if compute_iou(new_bbox, ex_bbox) > max_iou:
                            overlap = True; break
                            
                    if not overlap:
                        bg_img[cy:cy+mat["h"], cx:cx+mat["w"]] = mat["crop"]
                        new_label = xyxy_to_xywh_norm(cx, cy, cx+mat["w"], cy+mat["h"], bg_w, bg_h)
                        existing_boxes.append((mat["cls"], *new_label))
                        success_this_bg.append(mat["cls"])
                        placed = True; break

        if success_this_bg:
            bg_stem = Path(bg_lbl_path).stem
            out_name = f"{bg_stem}_aug_{task_id:06d}"
            
            out_img_path = str(Path(out_dir_str) / "train" / "pic" / f"{out_name}.jpg")
            out_lbl_path = str(Path(out_dir_str) / "train" / "yolo" / f"{out_name}.txt")

            cv2.imwrite(out_img_path, bg_img)
            write_label_file(out_lbl_path, existing_boxes)
            return success_this_bg
            
    return []

# ================= 核心类 =================
class SmartAugmenterV20:
    def __init__(self, args):
        self.data_dir = Path(args.data_dir)
        self.out_dir = Path(args.output_dir)
        self.base_count = args.base_count
        self.max_iou = args.max_iou  
        self.workers = args.workers
        self.max_major_per_bg = args.max_major_per_bg 
        
        ratio_strs = args.ratios.split(",")
        if len(ratio_strs) != 15:
            raise ValueError("必须严格提供 15 个类别的比例，用逗号分隔！")
        self.ratios = [float(r) for r in ratio_strs]
        
        (self.out_dir / "train" / "pic").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "train" / "yolo").mkdir(parents=True, exist_ok=True)
            
        self.img_cache = {}
        self.major_classes = set(c for c, r in enumerate(self.ratios) if r == 1.0)

    def preload_image_cache(self):
        img_dir = self.data_dir / "images" / "train"
        if not img_dir.exists(): return
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            for img_path in img_dir.glob(f"*{ext}"):
                self.img_cache[img_path.stem] = str(img_path)

    def scan_dataset_fast(self):
        global_counts = defaultdict(int)
        labels_dir = self.data_dir / "labels" / "train"
        label_files = []
        if labels_dir.exists():
            label_files.extend([str(f) for f in labels_dir.glob("*.txt") if f.name != "classes.txt"])
        
        max_workers = max(1, multiprocessing.cpu_count() - 2)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(parse_single_file_for_count, label_files), 
                                total=len(label_files), desc="⚡ 极速扫描与背景提纯 (仅Train)"))
            
        bg_histogram = defaultdict(int)
        pure_bg_info = [] 
        
        for file_path, counts in results:
            for k, v in counts.items():
                global_counts[k] += v
                
            total_boxes = sum(counts.values())
            if total_boxes == 0:
                bg_histogram[-1] += 1
                continue
                
            major_boxes = sum(counts[c] for c in self.major_classes if c in counts)
            if major_boxes >= 4:
                bg_histogram[4] += 1
            else:
                bg_histogram[major_boxes] += 1
                
            if major_boxes <= self.max_major_per_bg:
                pure_bg_info.append((major_boxes, file_path))
                
        self.pure_bg_info = pure_bg_info
        self.bg_histogram = bg_histogram
        return global_counts

    def build_meta_report(self, before_counts, augment_counts, title_prefix=""):
        """生成形状与颜色的宏观大类属性分析报表"""
        lines = []
        lines.append("\n" + "=" * 120)
        lines.append(f"🌟 形状与颜色大类属性分析 ({title_prefix})")
        lines.append("=" * 120)

        shape_map = {
            "round": [0, 1, 2],
            "up": [3, 4, 5],
            "left": [6, 7, 8],
            "right": [9, 10, 11],
            "turn_around": [12, 13, 14]
        }
        color_map = {
            "red": [0, 3, 6, 9, 12],
            "yellow": [1, 4, 7, 10, 13],
            "green": [2, 5, 8, 11, 14]
        }

        lines.append("【形状统计 (Shapes)】")
        for shape, ids in shape_map.items():
            b_cnt = sum(before_counts.get(i, 0) for i in ids)
            a_cnt = sum(augment_counts.get(i, 0) for i in ids)
            lines.append(f"  - {shape:<15} : {b_cnt:<6} -> {b_cnt + a_cnt:<6} (+{a_cnt})")

        lines.append("\n【颜色统计 (Colors)】")
        for color, ids in color_map.items():
            b_cnt = sum(before_counts.get(i, 0) for i in ids)
            a_cnt = sum(augment_counts.get(i, 0) for i in ids)
            lines.append(f"  - {color:<15} : {b_cnt:<6} -> {b_cnt + a_cnt:<6} (+{a_cnt})")

        lines.append("=" * 120 + "\n")
        return "\n".join(lines)

    def generate_report(self, current_counts, deficit, after_counts, report_name="augmentation_report.txt"):
        total_before = sum(current_counts.values())
        total_after = sum(after_counts.values())
        
        report_lines = []
        report_lines.append("📊 数据集增强综合审计报告 (V20 宏观属性分析版)")
        report_lines.append("=" * 120)
        report_lines.append(f"基准数量 (Base Count): {self.base_count}")
        report_lines.append(f"增强前总框数 (仅Train): {total_before}")
        report_lines.append(f"预期增强后总数: {total_after} (+{total_after - total_before})")
        report_lines.append("=" * 120)
        
        report_lines.append("\n🖼️ 【Train纯净背景池】提纯分析报告:")
        report_lines.append(f"设定的大类(Ratio=1.0)为: {[CURRENT_CLASSES[c] for c in self.major_classes]}")
        report_lines.append(f" -> 包含 0 个大类目标: {self.bg_histogram[0]:>6} 张 (⭐ 绝对优先消耗区)")
        report_lines.append(f" -> 包含 1 个大类目标: {self.bg_histogram[1]:>6} 张 (✔️ 降级保底备用区)")
        report_lines.append(f" -> 包含 ≥2 个大类目标: {sum(v for k, v in self.bg_histogram.items() if k >= 2):>6} 张 (均淘汰)")
        report_lines.append(f"✅ 提纯完毕！最终入选【纯净背景池】的图片共计: {len(self.pure_bg_info)} 张\n")

        header = f"{'ID':<3} | {'类别名':<18} | {'设定比例':<8} | {'目标数量':<8} | {'增强前数量':<10} | {'需增强(次)':<10}"
        report_lines.append(header)
        report_lines.append("-" * 120)
        
        for cls_id in range(15):
            name = CURRENT_CLASSES[cls_id]
            ratio = self.ratios[cls_id]
            target_str = "SKIP" if ratio == 1.0 else str(int(self.base_count * ratio))
            before_cnt = current_counts[cls_id]
            need_aug = deficit[cls_id]
            line = f"{cls_id:<3} | {name:<18} | {ratio:<8.3f} | {target_str:<8} | {before_cnt:<10} | {need_aug:<10}"
            report_lines.append(line)
        
        # 附加属性报告 (预期)
        meta_report = self.build_meta_report(current_counts, deficit, "Expected / 预期计划")
        report_lines.append(meta_report)
        
        full_report = "\n".join(report_lines)
        print("\n" + full_report)
        with open(self.out_dir / report_name, 'w', encoding='utf-8') as f:
            f.write(full_report + "\n")

    def build_material_library_fast(self, classes_to_aug):
        library_path = self.out_dir / "material_library.txt"
        materials = {cls: [] for cls in classes_to_aug}
        
        labels_dir = self.data_dir / "labels" / "train"
        label_files = []
        if labels_dir.exists():
            label_files.extend([f for f in labels_dir.glob("*.txt") if f.name != "classes.txt"])
        
        with open(library_path, 'w', encoding='utf-8') as f:
            f.write("class_id,img_path,x_center,y_center,w,h\n")
            for lbl_file in tqdm(label_files, desc="🔍 极速构建素材库索引"):
                img_path = self.img_cache.get(lbl_file.stem)
                if not img_path: continue
                
                for cls, x, y, w, h in parse_label_file(str(lbl_file)):
                    if cls in classes_to_aug:
                        record = f"{cls},{img_path},{x},{y},{w},{h}"
                        f.write(record + "\n")
                        materials[cls].append(record)
        return materials

    def run(self):
        print("💾 正在预加载Train图片路径到内存...")
        self.preload_image_cache()

        current_counts = self.scan_dataset_fast()
        if self.base_count <= 0: self.base_count = max(current_counts.values())

        deficit = {}
        after_counts = current_counts.copy()
        classes_to_aug = []
        
        for cls in range(15):
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
        if not classes_to_aug:
            print("🎉 当前数据已全部满足设定比例，无需增强！")
            return
            
        materials = self.build_material_library_fast(classes_to_aug)
        
        bg_groups = defaultdict(list)
        for major_count, lbl_file in self.pure_bg_info:
            bg_groups[major_count].append(lbl_file)
            
        ordered_bgs = []
        for count in sorted(bg_groups.keys()):
            bgs = bg_groups[count]
            random.shuffle(bgs)
            ordered_bgs.extend(bgs)

        valid_bg_pairs = []
        for lbl_file in ordered_bgs:
            stem = Path(lbl_file).stem
            img_path = self.img_cache.get(stem)
            if img_path:
                valid_bg_pairs.append((lbl_file, img_path))
        
        max_workers = max(1, multiprocessing.cpu_count() - 2) if self.workers <= 0 else self.workers
        print(f"\n🚀 启动 V20 保守安全增强引擎 (核心数: {max_workers})")

        success_counts = {cls: 0 for cls in classes_to_aug}
        total_skipped_chunks = 0
        target_success_total = sum(deficit.values())
        global_task_counter = 0
        bg_index = 0

        def make_task_args(active_classes, current_deficit, current_success):
            nonlocal global_task_counter, bg_index
            global_task_counter += 1
            
            total_remaining = sum(current_deficit[c] - current_success[c] for c in active_classes)
            num_pastes = min(MAX_PASTES_PER_IMG, total_remaining)
            
            chosen_classes = []
            if num_pastes > 0:
                distinct_pool = list(active_classes)
                random.shuffle(distinct_pool)
                chosen_classes.extend(distinct_pool[:num_pastes])
                while len(chosen_classes) < num_pastes:
                    chosen_classes.append(random.choice(active_classes))
            
            chunk = []
            for c in chosen_classes:
                mat = random.choice(materials[c])
                chunk.append((c, mat))
                
            bg_sample_pairs = []
            for i in range(min(10, len(valid_bg_pairs))):
                bg_sample_pairs.append(valid_bg_pairs[(bg_index + i) % len(valid_bg_pairs)])
            
            bg_index = (bg_index + 1) % len(valid_bg_pairs)
            
            return (chunk, bg_sample_pairs, str(self.out_dir), self.max_iou, global_task_counter)

        active_futures = set()

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            active_classes = [c for c in classes_to_aug if success_counts[c] < deficit[c]]
            if active_classes:
                for i in range(max_workers * 2):
                    active_futures.add(executor.submit(worker_task, make_task_args(active_classes, deficit, success_counts)))

            with tqdm(total=target_success_total, desc="🎯 目标生成进度") as pbar:
                while active_futures:
                    done, active_futures = concurrent.futures.wait(
                        active_futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        try:
                            success_list = future.result()
                            if success_list:
                                for c in success_list:
                                    if success_counts[c] < deficit[c]:
                                        success_counts[c] += 1
                                        pbar.update(1)
                            else:
                                total_skipped_chunks += 1
                        except Exception:
                            total_skipped_chunks += 1
                            
                        active_classes = [c for c in classes_to_aug if success_counts[c] < deficit[c]]
                        pbar.set_postfix({"🗑️淘汰盲盒": total_skipped_chunks})
                        
                        if active_classes:
                            active_futures.add(executor.submit(worker_task, make_task_args(active_classes, deficit, success_counts)))

        total_generated = sum(success_counts.values())
        
        report_tail = []
        report_tail.append("\n\n🏁 实际增强执行结果 (V20)")
        report_tail.append("=" * 120)
        report_tail.append(f"实际成功生成的目标框数: {total_generated}")
        report_tail.append(f"最终生成的独立图片文件数量: {global_task_counter - total_skipped_chunks} 张")
        report_tail.append("=" * 120)
        
        header = f"{'ID':<3} | {'类别名':<18} | {'原始数量':<10} | {'实际成功增强(次)':<14} | {'最终纯目标量':<10}"
        report_tail.append(header)
        report_tail.append("-" * 120)
        
        for cls_id in range(15):
            name = CURRENT_CLASSES[cls_id]
            orig_cnt = current_counts[cls_id]
            success_cnt = success_counts.get(cls_id, 0)
            final_cnt = orig_cnt + success_cnt
            report_tail.append(f"{cls_id:<3} | {name:<18} | {orig_cnt:<10} | {success_cnt:<14} | {final_cnt:<10}")
        
        # 附加属性报告 (实际)
        actual_meta_report = self.build_meta_report(current_counts, success_counts, "Actual Execution / 实际执行")
        report_tail.append(actual_meta_report)
            
        full_tail = "\n".join(report_tail)
        print(full_tail)
        
        with open(self.out_dir / 'augmentation_report.txt', 'a', encoding='utf-8') as f:
            f.write(full_tail + "\n")

        print(f"\n✨ 增强结束！请将 {self.out_dir}/train 目录下的增量数据，直接合并进你的训练集！")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="YOLO 智能按需 Copy-Paste 增强脚本 (V20)")
    parser.add_argument("--data_dir", type=str, required=True, help="原始数据集目录")
    parser.add_argument("--output_dir", type=str, required=True, help="增强数据输出目录")
    parser.add_argument("--base_count", type=int, default=16402, help="基准数量")
    # 【核心调整】全面调低防过拟合比例 (left_green因为自身数量充足直接设为1.0跳过)
    parser.add_argument("--ratios", type=str, 
                        default="1.0,0.12,1.0,0.08,0.01,0.13,1.0,0.05,1.0,0.05,0.004,0.13,0.06,0.008,0.03", 
                        help="严格对应15类的防过拟合比例")
    parser.add_argument("--max_major_per_bg", type=int, default=1, 
                        help="单张背景图允许包含的最大“大类”目标数量，越小背景越纯净。默认=1")
    parser.add_argument("--max_iou", type=float, default=0.0, help="最大允许的 IoU 阈值")
    parser.add_argument("--workers", type=int, default=0, help="并行进程数")
    
    args = parser.parse_args()
    augmenter = SmartAugmenterV20(args) 
    augmenter.run()