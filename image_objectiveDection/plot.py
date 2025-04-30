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

# === 2. Blur ===
df_blur1 = load_log('gaussianblur_3_0.8', 'blur_light')
df_blur2 = load_log('gaussianblur_5_2.0', 'blur_medium')
df_blur3 = load_log('gaussianblur_9_4.0', 'blur_heavy')
df_all_blur = pd.concat([df_clean, df_blur1, df_blur2, df_blur3])
plt.figure(figsize=(10,6))
sns.lineplot(data=df_all_blur, x='epoch', y='val_map', hue='model',
             palette={'clean':'black', 'blur_light':'green', 'blur_medium':'gold', 'blur_heavy':'red'})
plt.title("Validation mAP - Gaussian Blur")
plt.xlabel("Epoch")
plt.ylabel("Validation mAP")
plt.grid(True)
plt.savefig("plot_blur.png")
plt.close()

# === 3. Gaussian Noise ===
df_noise1 = load_log('gaussiannoise_0_0.02', 'noise_light')
df_noise2 = load_log('gaussiannoise_0_0.1', 'noise_medium')
df_noise3 = load_log('gaussiannoise_0_0.25', 'noise_heavy')
df_all_noise = pd.concat([df_clean, df_noise1, df_noise2, df_noise3])
plt.figure(figsize=(10,6))
sns.lineplot(data=df_all_noise, x='epoch', y='val_map', hue='model',
             palette={'clean':'black', 'noise_light':'green', 'noise_medium':'gold', 'noise_heavy':'red'})
plt.title("Validation mAP - Gaussian Noise")
plt.xlabel("Epoch")
plt.ylabel("Validation mAP")
plt.grid(True)
plt.savefig("plot_noise.png")
plt.close()

# === 4. Aliasing ===
df_alias1 = load_log('aliasing_2', 'aliasing_light')
df_alias2 = load_log('aliasing_4', 'aliasing_medium')
df_alias3 = load_log('aliasing_8', 'aliasing_heavy')
df_all_alias = pd.concat([df_clean, df_alias1, df_alias2, df_alias3])
plt.figure(figsize=(10,6))
sns.lineplot(data=df_all_alias, x='epoch', y='val_map', hue='model',
             palette={'clean':'black', 'aliasing_light':'green', 'aliasing_medium':'gold', 'aliasing_heavy':'red'})
plt.title("Validation mAP - Aliasing")
plt.xlabel("Epoch")
plt.ylabel("Validation mAP")
plt.grid(True)
plt.savefig("plot_aliasing.png")
plt.close()

# === 5. JPEG Compression ===
df_jpeg1 = load_log('jpegcompression_60', 'jpeg_light')
df_jpeg2 = load_log('jpegcompression_20', 'jpeg_medium')
df_jpeg3 = load_log('jpegcompression_5', 'jpeg_heavy')
df_all_jpeg = pd.concat([df_clean, df_jpeg1, df_jpeg2, df_jpeg3])
plt.figure(figsize=(10,6))
sns.lineplot(data=df_all_jpeg, x='epoch', y='val_map', hue='model',
             palette={'clean':'black', 'jpeg_light':'green', 'jpeg_medium':'gold', 'jpeg_heavy':'red'})
plt.title("Validation mAP - JPEG Compression")
plt.xlabel("Epoch")
plt.ylabel("Validation mAP")
plt.grid(True)
plt.savefig("plot_jpeg.png")
plt.close()
