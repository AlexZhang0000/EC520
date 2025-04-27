import torch

class Config:
    # 类别（5类）
    classes = ['person', 'cat', 'dog', 'car', 'bicycle']
    num_classes = len(classes)

    # 训练超参数
    epochs = 100
    batch_size = 16
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 5e-4

    # 优化器
    optimizer_type = 'SGD'

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机种子
    seed = 42

    # 保存路径
    model_save_path = './saved_models'
    result_save_path = './results'

    # 输入图像大小
    img_size = 640  # YOLOv5默认是640x640

    # YOLOv5预训练配置
    pretrained_yolov5 = True
    yolov5_variant = 'yolov5s'

    # VOC2007固定下载路径
    voc_root = './data/VOC2007'

