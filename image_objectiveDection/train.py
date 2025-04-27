import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from config import Config
from dataloader import get_loader
from model import YOLOv5Backbone
from utils import compute_map, set_seed

def train(train_distortion=None):
    set_seed(Config.seed)

    print(f"Using device: {Config.device}")

    # 数据
    train_loader = get_loader(batch_size=Config.batch_size, mode='train', distortion=train_distortion, pin_memory=True)
    val_loader = get_loader(batch_size=Config.batch_size, mode='val', distortion=None, pin_memory=True)

    # 模型
    model = YOLOv5Backbone(num_classes=Config.num_classes).to(Config.device)

    # 损失函数
    bce_loss = nn.BCEWithLogitsLoss()
    ciou_loss = nn.MSELoss()  # 简化版替代CIoU（注意：正式版建议替换）

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate, momentum=Config.momentum, weight_decay=Config.weight_decay)

    # 保存最好mAP
    best_map = 0.0

    for epoch in range(1, Config.epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, targets, labels in train

