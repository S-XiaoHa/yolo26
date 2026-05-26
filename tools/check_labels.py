import cv2
import os

def draw_yolo_boxes(img_path, txt_path, out_path):
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None: 
        print(f"❌ 错误：找不到图片，请检查路径是否正确 -> {img_path}")
        return
    h, w = img.shape[:2]

    # 2. 读取标签
    if not os.path.exists(txt_path): 
        print(f"❌ 错误：找不到标签文件，请检查路径 -> {txt_path}")
        return
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("⚠️ 警告：标签文件是空的（该图没有目标）。")

    # 3. 反推像素坐标并画框
    for line in lines:
        parts = line.strip().split()
        cls_id = parts[0]
        cx, cy, bw, bh = map(float, parts[1:5])
        
        # 算回真实像素坐标
        px_cx, px_cy = int(cx * w), int(cy * h)
        px_bw, px_bh = int(bw * w), int(bh * h)
        
        x1 = int(px_cx - px_bw / 2)
        y1 = int(px_cy - px_bh / 2)
        x2 = int(px_cx + px_bw / 2)
        y2 = int(px_cy + px_bh / 2)

        # 画个绿色的框，线宽 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"cls:{cls_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存结果
    cv2.imwrite(out_path, img)
    print(f"✅ 已成功生成验证图: {os.path.abspath(out_path)}")

# --- 请把下面这两个路径修改为你当前的真实文件 ---
# 注意：img 的路径里是 images，txt 的路径里是 labels！

test_img = "/home/jzyh/code/ultralytics/data/images/train/image_1756955056.png"  
test_txt = "/home/jzyh/code/ultralytics/data/labels/train/image_1756955056.txt" 

draw_yolo_boxes(test_img, test_txt, "verify_check.jpg")