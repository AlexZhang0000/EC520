import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def load_log(tag, label):
    path = f'./results/train_log_{tag}.csv'
    df = pd.read_csv(path)
    df['epoch'] = df['epoch'].astype(int)
    df['model'] = label
    return df

# === Clean only ===
df_clean = load_log('clean', 'clean')
plt.figure(figsize=(10,6))
sns.lineplot(data=df_clean, x='epoch', y='val_map', color='black', label='clean')
plt.title("Validation mAP - Clean Only")
plt.xlabel("Epoch")
plt.ylabel("Validation mAP")
plt.grid(True)
plt.savefig("plot_clean.png")
plt.close()

# === Distorted ===
df_aliasing = load_log('aliasing_8', 'aliasing_heavy')
df_jpeg = load_log('jpegcompression_5', 'jpeg_heavy')
df_all = pd.concat([df_clean, df_aliasing, df_jpeg])

palette = {
    'clean': 'black',
    'aliasing_heavy': 'red',
    'jpeg_heavy': 'red'
}

plt.figure(figsize=(10,6))
sns.lineplot(data=df_all, x='epoch', y='val_map', hue='model', palette=palette)
plt.title("Validation mAP - Clean vs Aliasing/JPEG Heavy")
plt.xlabel("Epoch")
plt.ylabel("Validation mAP")
plt.grid(True)
plt.savefig("plot_aliasing_jpeg.png")
plt.close()

