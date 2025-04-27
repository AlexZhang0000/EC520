import torch

class Config:
    # Data paths
    train_data_dir = './Data'  # 你的训练数据路径
    test_data_dir = './Data'   # 测试数据路径

    # Training parameters
    num_epochs = 200
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    # Model parameters
    num_classes = 10  # CIFAR-10是10类

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Random seed (可选，如果想要复现实验)
    seed = 42

    # Save paths
    model_save_path = './saved_models'
    results_save_path = './results'


