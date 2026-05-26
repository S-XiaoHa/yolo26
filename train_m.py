"""Ultralytics YOLO26 dual-task training script for traffic light detection.

Maps 15-class labels (5 shapes x 3 colors) onto two independent classification
heads (Detect26.cv3 for shape, Detect26.cv4 for color). Loss is computed by
v8DetectionLoss26 and is wired into the model via init_criterion monkey-patching.
"""


from ultralytics import YOLO


# 加上这一句导入您的自定义 Trainer
from my_trainer import DualTaskTrainer


if __name__ == "__main__":
    model = YOLO("/home/jzyh/code/ultralytics/runs/detect/26m/train1/exp3/weights/epoch60.pt")

    model.train(
        data="./data/mydata_dual.yaml",
        task="detect",
        end2end=True,
        cache="disk",
        epochs=170,
        imgsz=1920,
        seed=24,
        single_cls=False,
        deterministic=True,
        profile=False,
        batch=8,
        pretrained=True,
        patience=50,
        cos_lr=True,
        save_period=5,
        val=True,
        workers=8,
        device="0,1,2,3",
        # Augmentation
        # mosaic=0.392,
        mosaic=0.3,
        mixup=0.01,
        cutmix=0.00082,
        copy_paste=0.0,
        # scale=0.3,
        scale=0.2,
        fliplr=0.0,
        flipud=0.0,
        translate=0.275,
        perspective=0.0,
        hsv_h=0.001,
        hsv_s=0.453,
        hsv_v=0.294,
        bgr=0.0,
        # Loss weights
        box=9.48,
        # cls=1.5,
        cls=1.0,
        dfl=0.96,
        # Optimizer / LR
        weight_decay=0.00027,
        # warmup_epochs=0.99,
        warmup_epochs=0.99,
        warmup_momentum=0.54064,
        warmup_bias_lr=0.05684,
        # lr0=0.00038,
        lr0=0.00098,           # 从 0.00038 提升，给随机初始化的分类头足够的动力
        # lrf=0.882,
        lrf=0.2,            
        momentum=0.948,
        shear=0.0,
        degrees=0.0,
        close_mosaic=20,
        multi_scale=0.0,
        optimizer="MuSGD",
        resume=False,
        amp=True,
        project="26m/train",
        name="exp",
        trainer=DualTaskTrainer,
    )
