import torch
import os

class Config:
    project_name = 'image_segmentation_camvid'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

    # Dataset
    data_root = './Data/CamVid'
    num_classes = 11  # ✅CamVid只有11类！

    # Training
    batch_size = 16
    num_epochs = 200
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Save paths
    model_save_path = './saved_models'
    results_save_path = './results'

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(results_save_path, exist_ok=True)
