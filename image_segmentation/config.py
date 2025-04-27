import torch
import os

class Config:
    # 路径
    data_root = './data/VOCdevkit'  # Pascal VOC官方结构
    model_save_path = './saved_models'
    result_save_path = './results'

    # 训练参数
    num_epochs = 100
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-4
    seed = 42

    # 类别数
    num_classes = 5  # 背景、人、猫、狗、车

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 类别映射
    class_map = {
        0: 'background',
        1: 'person',
        2: 'cat',
        3: 'dog',
        4: 'car'
    }



