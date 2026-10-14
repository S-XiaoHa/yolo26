# import warnings
# warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 放弃从零训练，直接加载预训练权重进行微调 (极大幅度提升收敛速度和特征泛化能力)
    model = YOLO("yolo26s.pt")
    data = "./MyData/mydata.yaml"
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
    # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    # model.load('yolov11n.pt') # 是否加载预训练权重,

    model.train(
        data=data,
        task="detect",
        end2end=True,  # 保持开启：YOLO26 专属的无 NMS 端到端训练
        cache=False,  # 建议关闭或设为 'disk'，防止 2 万张高分辨率图撑爆内存
        epochs=300,  # 增加轮次：38个细粒度类别需要充分学习
        imgsz=1280,  # 关键提升：针对红绿灯小目标，必须拉高分辨率 (若显存不足可降至 960)
        batch=32,  # 因分辨率提升，适当降低 batch 避免显卡 OOM (4卡跑32或64)
        pretrained=True,  # 开启预训练权重加载
        patience=50,  # 开启早停：50轮没有提升则自动停止，防止过拟合
        single_cls=False,
        deterministic=True,
        profile=True,
        workers=8,
        device="0,1,2,3",
        save_period=5,
        val=True,
        # Optimizer and Learning Rate
        optimizer="MuSGD",  # 保持 MuSGD，对红绿灯长尾类别非常有效
        lr0=0.01,  # 恢复正常初始学习率
        lrf=0.01,  # 最终学习率比例 (0.01 * 0.01 = 0.0001)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,  # 增加 warmup 轮次，让早期收敛更平稳
        # Augmentation Pipeline (红绿灯定制版)
        mosaic=1.0,  # 开启 100% 马赛克，对小目标极其重要
        mixup=0.0,  # 建议关闭：重叠的透明红绿灯在现实中不存在，会引入严重噪声
        copy_paste=0.0,
        scale=0.5,  # 缩小缩放幅度：红绿灯本身就很小，过度缩小会丢失目标
        fliplr=0.0,  # 绝对禁止！不能翻转，否则左转/右转箭头标签全部作废
        translate=0.1,
        hsv_h=0.0,  # 绝对禁止！不能改变色相，绿灯不能变色
        hsv_s=0.5,  # 可以改变饱和度 (模拟大雾或褪色)
        hsv_v=0.4,  # 可以改变亮度 (模拟逆光或夜晚)
        bgr=0.0,
        # Loss Weights
        box=7.5,
        cls=0.5,  # 类别损失比例
        # dfl=1.5,          # 提醒：YOLO26 已经移除了 DFL 模块，此参数通常失效，可删除
        # Key design choices across all sizes
        shear=0.0,  # 关闭剪切变换
        degrees=0.0,  # 关闭旋转：红绿灯永远是横平竖直的
        close_mosaic=10,
        project="26s_traffic_lights",
        name="train_1280_noflip",
    )
