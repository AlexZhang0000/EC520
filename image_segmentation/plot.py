import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def load_log(tag, label):
    path = f'./results/train_log_{tag}.csv'
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return None
    df = pd.read_csv(path)
    df['epoch'] = df['epoch'].astype(int)
    df['model'] = label
    return df

def plot_and_save(df_list, palette, title, filename):
    df_all = pd.concat([df for df in df_list if df is not None])
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_all, x='epoch', y='val_miou', hue='model', palette=palette)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Validation mIoU")
    plt.grid(True)
    plt.savefig(filename)
    print(f"✅ Saved: {filename}")
    plt.close()

# === Clean Only ===
df_clean = load_log('clean', 'clean')
if df_clean is not None:
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_clean, x='epoch', y='val_miou', color='black', label='clean')
    plt.title("Validation mIoU (Clean Only)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation mIoU")
    plt.grid(True)
    plt.savefig("plot_clean.png")
    print("✅ Saved: plot_clean.png")
    plt.close()

# === Blur ===
df_blur1 = load_log('gaussianblur_3,0.8', 'blur_light')
df_blur2 = load_log('gaussianblur_5,2.0', 'blur_medium')
df_blur3 = load_log('gaussianblur_9,4.0', 'blur_heavy')
plot_and_save([df_clean, df_blur1, df_blur2, df_blur3],
              palette={'clean': 'black', 'blur_light': 'green', 'blur_medium': 'gold', 'blur_heavy': 'red'},
              title='Validation mIoU - Blur',
              filename='plot_blur.png')

# === Noise ===
df_noise1 = load_log('gaussiannoise_0,0.02', 'noise_light')
df_noise2 = load_log('gaussiannoise_0,0.1', 'noise_medium')
df_noise3 = load_log('gaussiannoise_0,0.25', 'noise_heavy')
plot_and_save([df_clean, df_noise1, df_noise2, df_noise3],
              palette={'clean': 'black', 'noise_light': 'green', 'noise_medium': 'gold', 'noise_heavy': 'red'},
              title='Validation mIoU - Gaussian Noise',
              filename='plot_noise.png')

# === Aliasing ===
df_alias1 = load_log('aliasing_2', 'aliasing_light')
df_alias2 = load_log('aliasing_4', 'aliasing_medium')
df_alias3 = load_log('aliasing_8', 'aliasing_heavy')
plot_and_save([df_clean, df_alias1, df_alias2, df_alias3],
              palette={'clean': 'black', 'aliasing_light': 'green', 'aliasing_medium': 'gold', 'aliasing_heavy': 'red'},
              title='Validation mIoU - Aliasing',
              filename='plot_aliasing.png')

# === JPEG ===
df_jpeg1 = load_log('jpegcompression_60', 'jpeg_light')
df_jpeg2 = load_log('jpegcompression_20', 'jpeg_medium')
df_jpeg3 = load_log('jpegcompression_5', 'jpeg_heavy')
plot_and_save([df_clean, df_jpeg1, df_jpeg2, df_jpeg3],
              palette={'clean': 'black', 'jpeg_light': 'green', 'jpeg_medium': 'gold', 'jpeg_heavy': 'red'},
              title='Validation mIoU - JPEG Compression',
              filename='plot_jpeg.png')

