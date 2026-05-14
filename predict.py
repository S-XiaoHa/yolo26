import os

from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 加载你刚刚训练出来的 last.pt 权重
    # 注意：如果你之前重新跑生成了 exp5，这里路径可能需要改成 exp5/weights/last.pt
    weight_path = "runs/detect/26m/train/exp4/weights/best.pt"

    if not os.path.exists(weight_path):
        print(f"找不到权重文件: {weight_path}，请检查路径！")
        exit()

    model = YOLO(weight_path)

    # 2. 准备你要测试的图片或文件夹路径
    # 可以是一张图片的绝对路径，也可以是一个包含多张图片的文件夹
    source_path = "./data/images/val"

    # 3. 执行推理预测
    results = model.predict(
        source=source_path,
        imgsz=[1536, 1920],  # 必须和你训练时保持一致，保证特征尺度相同！
        # conf=0.25,         # 置信度阈值：低于 0.25 的红绿灯预测框会被过滤掉
        # iou=0.45,          # NMS 交并比阈值：控制重叠框的消除
        save=True,  # 关键参数：设为 True，会把画好预测框的图片保存下来
        save_txt=False,  # 是否保存 txt 格式的预测结果
        show=False,  # 如果在无 UI 界面的服务器上，必须设为 False
        device="0",  # 推理只需要一张卡即可
    )

    print("✅ 推理完成！请去 runs/detect/predict2/ 目录下查看画好框的可视化图片。")
