import os
import torch

class Config:
    # 类别（5类）
    classes = ['person', 'cat', 'dog', 'car', 'bicycle']
    num_classes = len(classes)

    # 训练超参数
    epochs = 50
    batch_size = 4
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 5e-4

    # 优化器
    optimizer_type = 'SGD'

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 随机种子
    seed = 42

    # 保存路径
    base_save_dir = './saved_models'
    result_save_dir = './results'

    # 输入图像大小
    img_size = 640  # YOLOv5默认是640x640

    # YOLOv5预训练配置
    pretrained_yolov5 = True
    yolov5_variant = 'yolov5s'
    
    model_save_path = './saved_models'
    
    # VOC2007固定下载路径
    voc_root = os.path.join('.', 'data', 'VOC2007')

    # 初始化（比如创建必要的目录）
    @staticmethod
    def init():
        os.makedirs(Config.base_save_dir, exist_ok=True)
        os.makedirs(Config.result_save_dir, exist_ok=True)
        

# 在import的时候就初始化
Config.init()


