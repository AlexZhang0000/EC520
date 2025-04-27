import torch
import os

class Config:
    # Basic settings
    project_name = 'image_segmentation'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

    # Dataset
    data_root = './Data'  # 自动下载 VOC 2012 数据到这个文件夹
    num_classes = 21  # VOC有20类 + 背景
    
    # Training
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Save paths
    model_save_path = './saved_models'
    results_save_path = './results'

    # Create directories if not exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(results_save_path, exist_ok=True)
