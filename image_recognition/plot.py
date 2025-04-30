import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid')
colors = {'clean': 'black', 'light': 'green', 'medium': 'orange', 'heavy': 'red'}

# ========= 1. Clean-only Training Curve =========
df_clean = pd.read_csv('./results/train_log_clean.csv')
df_clean['model'] = 'clean'
df_clean['epoch'] = df_clean['epoch'].astype(int)

plt.figure(figsize=(10,5))
sns.lineplot(data=df_clean, x='epoch', y='train_acc', label='Train Accuracy', color='blue')
plt.title("Clean Model - Epoch vs Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Train Accuracy (%)")
plt.grid(True)
plt.savefig('./results/clean_train_acc.png')
plt.close()

plt.figure(figsize=(10,5))
sns.lineplot(data=df_clean, x='epoch', y='val_acc', label='Validation Accuracy', color='purple')
plt.title("Clean Model - Epoch vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.grid(True)
plt.savefig('./results/clean_val_acc.png')
plt.close()

# ========= 2. Distortion Training Curve Comparison =========
def plot_group(distortion_name, files):
    dfs = []
    for label, path in files.items():
        df = pd.read_csv(path)
        df['epoch'] = df['epoch'].astype(int)
        df['model'] = label
        dfs.append(df)
    df_all = pd.concat(dfs)

    plt.figure(figsize=(10,5))
    sns.lineplot(data=df_all, x='epoch', y='train_acc', hue='model', palette=colors)
    plt.title(f"Train Accuracy - {distortion_name.capitalize()} vs Clean")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy (%)")
    plt.grid(True)
    plt.savefig(f'./results/{distortion_name}_train_acc_compare.png')
    plt.close()

    plt.figure(figsize=(10,5))
    sns.lineplot(data=df_all, x='epoch', y='val_acc', hue='model', palette=colors)
    plt.title(f"Validation Accuracy - {distortion_name.capitalize()} vs Clean")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True)
    plt.savefig(f'./results/{distortion_name}_val_acc_compare.png')
    plt.close()

# Plot all groups
plot_group('blur', {
    'clean': './results/train_log_clean.csv',
    'light': './results/train_log_distorted_gaussianblur_3_0.8.csv',
    'medium': './results/train_log_distorted_gaussianblur_5_2.0.csv',
    'heavy': './results/train_log_distorted_gaussianblur_9_4.0.csv'
})
plot_group('noise', {
    'clean': './results/train_log_clean.csv',
    'light': './results/train_log_distorted_gaussiannoise_0_0.02.csv',
    'medium': './results/train_log_distorted_gaussiannoise_0_0.1.csv',
    'heavy': './results/train_log_distorted_gaussiannoise_0_0.25.csv'
})
plot_group('aliasing', {
    'clean': './results/train_log_clean.csv',
    'light': './results/train_log_distorted_aliasing_2.csv',
    'medium': './results/train_log_distorted_aliasing_4.csv',
    'heavy': './results/train_log_distorted_aliasing_8.csv'
})
plot_group('jpeg', {
    'clean': './results/train_log_clean.csv',
    'light': './results/train_log_distorted_jpegcompression_60.csv',
    'medium': './results/train_log_distorted_jpegcompression_20.csv',
    'heavy': './results/train_log_distorted_jpegcompression_5.csv'
})
