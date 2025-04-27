import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from dataloader import get_loader
from model import ResNet18

def evaluate():
    # Load test data
    test_loader = get_loader(Config.test_data_dir, batch_size=Config.batch_size, mode='test')

    # Create model
    model = ResNet18(num_classes=Config.num_classes).to(Config.device)

    # Load best saved model
    model_path = os.path.join(Config.model_save_path, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(Config.device)
            labels = labels.to(Config.device)

            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    acc = 100. * (all_preds == all_labels).sum() / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {acc:.2f}%")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(Config.results_save_path, 'confusion_matrix.png'))
    plt.close()

    # Save test results
    with open(os.path.join(Config.results_save_path, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.2f}%\n")
        f.write(f"F1 Score: {f1:.4f}\n")

if __name__ == '__main__':
    evaluate()

