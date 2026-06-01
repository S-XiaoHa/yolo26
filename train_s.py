#import warnings
#warnings.filterwarnings('ignore')


from ultralytics import YOLO
from my_trainer import DualTaskTrainer

if __name__ == '__main__':
	# model = YOLO("./ultralytics/cfg/models/26/yolo26s.yaml")
    model = YOLO("runs/detect/26s/train/exp/weights/epoch50.pt")
    data = "./data/mydata_dual.yaml"
	# 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
	# 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么就想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
	# model.load('yolov11n.pt') # 是否加载预训练权重,
    model.train(data=data,
	            # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
	            task='detect',
                end2end=True, # 是否使用端到端训练
				cache="disk",# 将所有图片缓存到 RAM 中。极大提升训练速度，但极其消耗内存
                epochs=150,
	            # imgsz=640,
                # imgsz=[1536, 1920],
                imgsz=1920, # 关键提升：针对红绿灯小目标，必须拉高分辨率 
                seed=24,
	            single_cls=False,  # 是否是单类别检测
                deterministic=True,  # 是否使用确定性训练
                profile=False,  # 是否分析每个步骤的时间和内存占用onnx或tesorrt
	            # batch=-1,
                batch=16,
	            pretrained=True,
	            patience=0,
	            cos_lr=True,
	            save_period=5,
	            val=True,   #培训期间不 验证/测试
	            workers=8,
	            device='0,1,2,3',
                # device='0',
                #Augmentation Pipeline
                mosaic=0.28,# 0.992马赛克增强，有 99.2% 的概率，系统会随机挑选 4 张不同的图片，把它们缩放并拼接成 1 张大图丢给模型去学
                mixup=0.005, #0.05 有 5% 的概率，系统会把两张完全不同的图片，像幻灯片一样设置一定的透明度（Alpha 通道）叠在一起
				cutmix=0.00082,
                close_mosaic=20,# 在最后 10 个 Epoch 关闭 Mosaic 增强
                # copy_paste=0.404,# 40.4% 的概率触发复制粘贴增强
                copy_paste = 0.0,# 无Mask 掩码，不增强
                # scale=0.9,
                scale=0.2,   # 缩小缩放幅度：红绿灯本身就很小，过度缩小会丢失目标
                # fliplr=0.304,
                fliplr=0.0,    # 绝对禁止！不能翻转，否则左转/右转箭头标签全部作废
                flipud=0.0,
                translate=0.275,
                perspective=0.0,
                # hsv_h=0.013,
                hsv_h=0.001,   # 禁止改变色相 (严禁红变绿)
                hsv_s=0.453, # 允许饱和度变化 (模拟不同天气)
                hsv_v=0.294, # 允许亮度变化 (模拟昼夜)
                bgr=0.0,
                #Loss Weights
                # box=9.83, # 损失的权重比例
                # box=7.5
                box=9.48,
                cls=1.0,
                dfl=0.96, # 分布焦点损失
                #Optimizer and Learning Rate
                weight_decay=0.00027,# 权重衰减惩罚项，用于防止过拟合。
                # warmup_epochs=0.99,
                warmup_epochs=1.99,
                warmup_momentum=0.54064,
                warmup_bias_lr=0.05684,
                # lr0=0.00038,
                lr0=0.001,
                # lrf=0.882,
                lrf=0.2,
                momentum=0.948,
                #Key design choices across all sizes
                shear=0.0,
                # degrees=0.001,
                degrees=0.0,   # 关闭旋转：红绿灯永远是横平竖直的
                
                # multi_scale=0.25,
                multi_scale=0.0, # 1% 的概率在每个 epoch 随机调整输入图像大小，增强模型对不同分辨率的适应能力
	            optimizer='MuSGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
	            
                resume=True,
                # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
	            amp=True,  # 如果出现训练损失为Nan可以关闭amp
	            project='26s/train',
	            name='exp',
                trainer=DualTaskTrainer,
	            )