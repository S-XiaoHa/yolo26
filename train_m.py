# import warnings
# warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == "__main__":
    # model = YOLO("./ultralytics/cfg/models/26/yolo26m.yaml")
    model = YOLO("yolo26m.pt")
    # model = YOLO("runs/detect/26m/train/exp4/weights/last.pt")
    data = "./data/mydata.yaml"
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
    # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    # model.load('yolov11n.pt') # 是否加载预训练权重,
    model.train(
        data=data,
        # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
        task="detect",
        end2end=True,  # 是否使用端到端训练
        cache="disk",  # 将所有图片缓存到 RAM 中。极大提升训练速度，但极其消耗内存
        # cache=False,
        epochs=170,
        # imgsz=640,
        imgsz=1920,  # 告诉框架：这批图的最大边长不要超过 1920
        # rect=True,    # 告诉框架：开启矩形训练！
        # imgsz=[1536, 1920],
        # imgsz=1536, # 关键提升：针对红绿灯小目标，必须拉高分辨率
        seed=24,
        single_cls=False,  # 是否是单类别检测
        deterministic=True,  # 是否使用确定性训练
        profile=False,  # 是否分析每个步骤的时间和内存占用onnx或tesorrt
        # batch=-1,
        batch=16,
        pretrained=True,
        patience=50,  # EarlyStopping 的耐心值，连续 50 个 epoch 验证指标没有提升就停止训练
        cos_lr=False,
        save_period=5,
        val=True,  # 培训期间不 验证/测试
        workers=8,
        device="0,1,2,3",
        # device='0',
        # Augmentation Pipeline
        # mosaic=0.992,# 0.992马赛克增强，有 99.2% 的概率，系统会随机挑选 4 张不同的图片，把它们缩放并拼接成 1 张大图丢给模型去学
        mosaic=0.392,
        mixup=0.01,  # 0.01 有 1% 的概率，系统会把两张完全不同的图片，像幻灯片一样设置一定的透明度（Alpha 通道）叠在一起
        cutmix=0.00082,  # 0.1 有 10% 的概率，系统会把两张完全不同的图片，像拼图一样裁剪成块状并组合在一起
        # copy_paste=0.304,# 40.4% 的概率触发复制粘贴增强
        copy_paste=0.0,
        # 注意：纯检测任务中 copy_paste 默认使用 flip 模式（翻转复制），不增加新的类别实例
        scale=0.3,  # 启用 30% 的缩放增强（随机缩小或放大）
        # fliplr=0.304,
        fliplr=0.0,  # 绝对禁止！不能翻转，否则左转/右转箭头标签全部作废
        flipud=0.0,
        translate=0.275,
        perspective=0.0,
        # hsv_h=0.013,
        hsv_h=0.0,  # 绝对禁止！不能改变色相，绿灯不能变色
        hsv_s=0.353,
        hsv_v=0.194,
        bgr=0.0,
        # Loss Weights
        # box=9.83, # 损失的权重比例
        # cls=0.65,
        # dfl=0.96, # 分布焦点损失
        box=7.5,  # 从 9.83 适当下调。告诉网络：框画得差不多就行了，别死磕边缘了。
        cls=1.5,  # 从 0.65 提升两倍多！严厉警告网络：看清楚里面的箭头！认错形状后果很严重！
        dfl=0.96,  # 保持官方不变。红绿灯边缘清晰，配合端到端特性，这个值非常合理。
        # Optimizer and Learning Rate
        weight_decay=0.00027,  # 权重衰减惩罚项，用于防止过拟合。
        warmup_epochs=0.99,
        warmup_momentum=0.54064,
        warmup_bias_lr=0.05684,
        lr0=0.00038,
        lrf=0.882,
        momentum=0.948,
        # Key design choices across all sizes
        shear=0.0,
        # degrees=0.001,
        degrees=0.0,  # 关闭旋转：红绿灯永远是横平竖直的
        close_mosaic=10,  # 在最后 10 个 Epoch 关闭 Mosaic 增强
        # multi_scale=0.25,
        multi_scale=0.0,  # 1% 的概率在每个 epoch 随机调整输入图像大小，增强模型对不同分辨率的适应能力
        optimizer="MuSGD",  # using SGD 优化器 默认为auto建议大家使用固定的.
        resume=False,
        # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
        amp=True,  # 如果出现训练损失为Nan可以关闭amp
        project="26m/train1",
        name="exp",
    )
