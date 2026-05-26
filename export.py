from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载你训练好的最优权重 
    model = YOLO("runs/detect/26m/train/exp/weights/epoch55.pt")

    # 2. 执行导出
    success = model.export(
        format="onnx",      # 指定导出格式为 ONNX
        imgsz=[1536, 1920],   # 写你训练时设置推理尺寸！
        half=False,         # 是否开启 FP16 半精度导出 (如果你要在 TensorRT 部署，建议先出 FP32，让 TRT 去量化)
        dynamic=False,      # 是否开启动态轴 (Dynamic Axes)。建议设为 False 写死尺寸，推理引擎优化更好
        simplify=True,      # 【强烈建议开启】调用 onnx-simplifier 简化冗余的算子，极大提高后续部署的成功率
        opset=12,            # ONNX 的算子集版本，通常 11 或 12 兼容性最好
        nms=False,
        batch=1,             #单帧推理
        end2end=True,       # 是否使用端到端模式导出
        device='0'
    )
    
    print(f"✅ ONNX 导出状态: {success}")