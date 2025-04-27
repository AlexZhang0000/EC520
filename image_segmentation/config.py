import torch
import os

class Config:
    # Basic settings
    project_name = 'image_segmentation'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

    # Dataset
    data_root = './Data'
    num_classes = 21  # VOC有20类+背景

    # Training
    batch_size = 16  # ✅ 保持较大batch size
    num_epochs = 200  # ✅ 改成200轮
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Save paths
    model_save_path = './saved_models'
    results_save_path = './results'

    # Ensure directories
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(results_save_path, exist_ok=True)

