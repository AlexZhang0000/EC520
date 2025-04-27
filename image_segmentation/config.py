import torch
import os

class Config:
    project_name = 'image_segmentation_ade20k_distortion'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

    # Dataset
    data_root = './Data/ADE20K'
    num_classes = 4  # 背景、建筑、树、人

    # Training
    batch_size = 8
    num_epochs = 200
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Save paths
    model_save_path = './saved_models'
    results_save_path = './results'

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(results_save_path, exist_ok=True)


