#!/usr/bin/env python3
"""
YOLO 数据集标签冲突安全扫描工具 (提取审查版)
功能：
1. 扫描所有的标签文件，寻找物理冲突 (IoU > 0.9 或 IoM > 0.9)
2. 将发现冲突的图片和对应 txt 标签，原封不动地复制到 OUTPUT_DIR
3. 生成详尽的《conflict_review_report.txt》供人工对照复核
4.cp -r /home/jzyh/xbzl/traffic_light/data_v1_ct/labels/* /home/jzyh/xbzl/traffic_light/data_v1_filtered/labels/.
"""

import shutil
from pathlib import Path

# ================= 配置区域 =================
SOURCE_DIR = Path("/home/jzyh/xbzl/traffic_light/data_v1_filtered")
OUTPUT_DIR = Path("/home/jzyh/xbzl/traffic_light/data_v1_ct")
OUTPUT_REPORT = "conflict_review_report.txt"  # 生成的审核报告名称

IOU_THRESH = 0.9  # 重叠度阈值
IOM_THRESH = 0.9  # 交小比阈值(大框套小框)
# ==========================================

# 类别字典，用于让报告更容易看懂
CLASS_MAP = {
    0: "round_red",
    1: "round_yellow",
    2: "round_green",
    3: "up_red",
    4: "up_yellow",
    5: "up_green",
    6: "left_red",
    7: "left_yellow",
    8: "left_green",
    9: "right_red",
    10: "right_yellow",
    11: "right_green",
    12: "turn_around_red",
    13: "turn_around_yellow",
    14: "turn_around_green",
}


def compute_overlap(box1, box2):
    """计算两个框的 IoU 和 IoM."""
    _, x1_c, y1_c, w1, h1 = box1[:5]
    _, x2_c, y2_c, w2, h2 = box2[:5]

    b1_x1, b1_y1 = x1_c - w1 / 2, y1_c - h1 / 2
    b1_x2, b1_y2 = x1_c + w1 / 2, y1_c + h1 / 2

    b2_x1, b2_y1 = x2_c - w2 / 2, y2_c - h2 / 2
    b2_x2, b2_y2 = x2_c + w2 / 2, y2_c + h2 / 2

    inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area, b2_area = w1 * h1, w2 * h2

    union_area = b1_area + b2_area - inter_area
    min_area = min(b1_area, b2_area)

    iou = inter_area / union_area if union_area > 0 else 0
    iom = inter_area / min_area if min_area > 0 else 0

    return iou, iom


def scan_label_file(txt_path):
    """扫描单个文件，返回发现的冲突列表."""
    conflicts = []

    with open(txt_path) as f:
        lines = f.readlines()

    if len(lines) < 2:
        return conflicts

    boxes = []
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append(
                [int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), line.strip(), idx]
            )

    # 暴力比对两两框
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou, iom = compute_overlap(boxes[i], boxes[j])

            if iou > IOU_THRESH or iom > IOM_THRESH:
                conflict_type = "高度重合 (IoU)" if iou > IOU_THRESH else "嵌套包含 (IoM)"

                cls_i_name = CLASS_MAP.get(boxes[i][0], f"未知类别({boxes[i][0]})")
                cls_j_name = CLASS_MAP.get(boxes[j][0], f"未知类别({boxes[j][0]})")

                detail = {
                    "type": conflict_type,
                    "iou": round(iou, 4),
                    "iom": round(iom, 4),
                    "box1": f"Line {boxes[i][6] + 1}: [{cls_i_name}] {boxes[i][5]}",
                    "box2": f"Line {boxes[j][6] + 1}: [{cls_j_name}] {boxes[j][5]}",
                }
                conflicts.append(detail)

    return conflicts


def get_image_path(lbl_path, phase_img_dir):
    """根据标签路径，寻找对应的图片（支持多种扩展名）."""
    stem = lbl_path.stem
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
        img_path = phase_img_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def generate_audit_report():
    phases = ["train", "val"]
    all_reports = []
    total_conflict_files = 0
    total_conflict_pairs = 0

    print(f"🚀 启动安全扫描任务: {SOURCE_DIR}")
    print(f"👀 正在寻找 IoU > {IOU_THRESH} 或 IoM > {IOM_THRESH} 的嫌疑框...")

    # 初始化输出主目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for phase in phases:
        src_lbl_dir = SOURCE_DIR / "labels" / phase
        src_img_dir = SOURCE_DIR / "images" / phase

        if not src_lbl_dir.exists():
            continue

        # 在 OUTPUT_DIR 中建立对应的 train/val 结构
        out_lbl_dir = OUTPUT_DIR / "labels" / phase
        out_img_dir = OUTPUT_DIR / "images" / phase

        lbl_files = [f for f in src_lbl_dir.iterdir() if f.suffix == ".txt" and f.name != "classes.txt"]

        for txt_path in lbl_files:
            conflicts = scan_label_file(txt_path)

            if conflicts:
                total_conflict_files += 1
                total_conflict_pairs += len(conflicts)

                # 记录报告信息
                all_reports.append(f"\n{'=' * 40}")
                all_reports.append(f"📁 发现冲突文件: {txt_path.name} ({phase})")
                for idx, c in enumerate(conflicts):
                    all_reports.append(f"  --- 冲突对 #{idx + 1} ---")
                    all_reports.append(f"  🚨 冲突类型: {c['type']} (IoU: {c['iou']}, IoM: {c['iom']})")
                    all_reports.append(f"  A 框: {c['box1']}")
                    all_reports.append(f"  B 框: {c['box2']}")

                # ================= 执行复制操作 =================
                img_path = get_image_path(txt_path, src_img_dir)

                if img_path:
                    # 确保输出子目录存在
                    out_lbl_dir.mkdir(parents=True, exist_ok=True)
                    out_img_dir.mkdir(parents=True, exist_ok=True)

                    # 物理复制图片和标签
                    shutil.copy(str(img_path), str(out_img_dir / img_path.name))
                    shutil.copy(str(txt_path), str(out_lbl_dir / txt_path.name))
                else:
                    print(f"⚠️ 警告: 找到冲突标签 {txt_path.name}，但在 {src_img_dir} 中未找到对应图片！")

    # 写入报告（报告直接生成在 OUTPUT_DIR 根目录下）
    report_path = OUTPUT_DIR / OUTPUT_REPORT
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("📊 数据集标签冲突审查与隔离报告\n")
        rf.write(f"数据源目录: {SOURCE_DIR}\n")
        rf.write(f"隔离输出目录: {OUTPUT_DIR}\n")
        rf.write(f"受影响文件数: {total_conflict_files} 个\n")
        rf.write(f"发现冲突对数: {total_conflict_pairs} 对\n")
        rf.write("说明: 存在冲突的图片和标签已被复制到此目录下。您可直接在此目录下使用标注工具进行修复。\n")
        rf.write("\n" + "\n".join(all_reports))

    # 将 classes.txt 也顺便复制过去，方便标注工具读取
    for phase in phases:
        src_classes = SOURCE_DIR / "labels" / phase / "classes.txt"
        out_classes_dir = OUTPUT_DIR / "labels" / phase
        if src_classes.exists() and out_classes_dir.exists():
            shutil.copy(str(src_classes), str(out_classes_dir / "classes.txt"))

    print("\n✅ 扫描及提取完成！")
    print(f"⚠️ 共发现 {total_conflict_files} 个存在冲突的文件，包含 {total_conflict_pairs} 对冲突框。")
    print(f"📂 冲突文件已全部提取至: {OUTPUT_DIR}")
    print(f"📄 审查报告已生成: {report_path}")
    print("👉 下一步建议：直接用标注软件打开隔离目录，修改完毕后，将隔离目录中的 txt 文件复制回原数据集覆盖即可！")


if __name__ == "__main__":
    generate_audit_report()
