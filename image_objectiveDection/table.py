import os
import pandas as pd

# 提取 mAP（或 mIoU）从 TXT 文件中
def get_miou_from_txt(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if 'Val mAP:' in line:
                try:
                    return float(line.strip().split(':')[1])
                except:
                    return None
    return None

# distortion 等级划分
categories = {
    "light": [
        ("gaussianblur:3,0.8", "_distorted_gaussianblur_3_0.8"),
        ("gaussiannoise:0,0.02", "_distorted_gaussiannoise_0_0.02"),
        ("aliasing:2", "_distorted_aliasing_2"),
        ("jpegcompression:60", "_distorted_jpegcompression_60")
    ],
    "medium": [
        ("gaussianblur:5,2.0", "_distorted_gaussianblur_5_2.0"),
        ("gaussiannoise:0,0.1", "_distorted_gaussiannoise_0_0.1"),
        ("aliasing:4", "_distorted_aliasing_4"),
        ("jpegcompression:20", "_distorted_jpegcompression_20")
    ],
    "heavy": [
        ("gaussianblur:9,4.0", "_distorted_gaussianblur_9_4.0"),
        ("gaussiannoise:0,0.25", "_distorted_gaussiannoise_0_0.25"),
        ("aliasing:8", "_distorted_aliasing_8"),
        ("jpegcompression:5", "_distorted_jpegcompression_5")
    ]
}

def compute_robustness(dist_type, distortions):
    rows = []
    acc_clean_on_clean = get_miou_from_txt('./results/test_log_clean.txt')

    for test_d, model_suffix in distortions:
        tag = test_d.replace(":", "_").replace(",", "_")

        acc_clean_on_dist = get_miou_from_txt(f'./results/test_log_clean_{tag}.txt')
        acc_model_on_clean = get_miou_from_txt(f'./results/test_log{model_suffix}_clean.txt')
        acc_model_on_dist = get_miou_from_txt(f'./results/test_log{model_suffix}_{tag}.txt')

        robustness = None
        if acc_clean_on_clean and acc_model_on_dist:
            robustness = round(acc_model_on_dist / acc_clean_on_clean, 3)

        row = {
            'Distortion Type': test_d,
            'CleanModel-CleanData': acc_clean_on_clean,
            'CleanModel-DistData': acc_clean_on_dist,
            'DistModel-CleanData': acc_model_on_clean,
            'DistModel-DistData': acc_model_on_dist,
            'Robustness (DistModel/DistData ÷ CleanModel/CleanData)': robustness
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    save_path = f'./results/robustness_{dist_type}.csv'
    df.to_csv(save_path, index=False)
    print(f"✅ Saved {save_path}")

# 运行所有
compute_robustness("light", categories["light"])
compute_robustness("medium", categories["medium"])
compute_robustness("heavy", categories["heavy"])
