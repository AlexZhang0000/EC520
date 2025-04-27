import os
import torch
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from dataloader import get_loader
from model import ResNet18
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_distortion', type=str, default=None,
                        help="Distortion format for test data: gaussianblur:kernel_size,sigma/...")
    parser.add_argument('--model_suffix', type=str, default='_clean',
                        help="Suffix for model file to load, e.g., _clean, _distorted_gaussianblur_5_2.0")
    return parser.parse_args()

def make_suffix(distortion, mode='test'):
    if distortion is None:
        return ''
    clean_distortion = distortion.replace('/', '_').replace(':', '_').replace(',', '_')
    return f"_{mode}_{clean_distortion}"

def evaluate():
    args = parse_args()

    distortion_suffix = make_suffix(args.test_distortion, 'test')

    test_loader = get_loader(Config.test_data_dir, batch_size=Config.batch_size, mode='test', distortion=args.test_distortion)

    model = ResNet18(num_classes=Config.num_classes).to(Config.device)
    model_path = os.path.join(Config.model_save_path, f'best_model{args.model_suffix}.pth')
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

    acc = 100. * (all_preds == all_labels).sum() / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {acc:.2f}%")
    print(f"F1 Score: {f1:.4f}")

    # Save results
    os.makedirs(Config.results_save_path, exist_ok=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(Config.results_save_path, f'confusion_matrix{distortion_suffix}.png'))
    plt.close()

    with open(os.path.join(Config.results_save_path, f'test_results{distortion_suffix}.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.2f}%\n")
        f.write(f"F1 Score: {f1:.4f}\n")

if __name__ == '__main__':
    evaluate()


