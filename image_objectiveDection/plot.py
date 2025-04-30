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

# === 1. Clean Only ===
df_clean = load_log('clean', 'clean')
plt.figure(figsize=(10,6))
sns.lineplot(data=df_clean, x='epoch', y='val_map', color='black', label='clean')
plt.title("Validation mAP (Clean Only)")
plt.xlabel("Epoch")
plt.ylabel("Validation mAP")
plt.grid(True)
plt.savefig("plot_clean.png")
plt.close()

# === 2. Heavy Distortions ===
df_blur = load_log('gaussianblur_9_4.0', 'blur_heavy')
df_noise = load_log('gaussiannoise_0_0.25', 'noise_heavy')
df_alias = load_log('aliasing_8', 'aliasing_heavy')
df_jpeg = load_log('jpegcompression_5', 'jpeg_heavy')

df_all = pd.concat([df_clean, df_blur, df_noise, df_alias, df_jpeg])
sns_palette = {
    'clean': 'black',
    'blur_heavy': 'red',
    'noise_heavy': 'red',
    'aliasing_heavy': 'red',
    'jpeg_heavy': 'red'
}

plt.figure(figsize=(10,6))
sns.lineplot(data=df_all, x='epoch', y='val_map', hue='model', palette=sns_palette)
plt.title("Validation mAP - Clean vs Heavy Distortions")
plt.xlabel("Epoch")
plt.ylabel("Validation mAP")
plt.grid(True)
plt.savefig("plot_heavy_all.png")
plt.close()
