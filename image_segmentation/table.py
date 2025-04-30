import os
import pandas as pd

# 从 result txt 中提取 mIoU
def get_miou_from_txt(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if 'mIoU:' in line:
                return float(line.strip().split(':')[1])
    return None

# distortion 分级定义
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
    acc_clean_on_clean = get_miou_from_txt('./results/result_best_model_clean.txt')

    for test_d, model_suffix in distortions:
        tag = test_d.replace(":", "_").replace(",", "_")

        acc_clean_on_dist = get_miou_from_txt(f'./results/result_best_model_{tag}.txt')
        acc_model_on_clean = get_miou_from_txt(f'./results/result_best_model{model_suffix}_clean.txt')
        acc_model_on_dist = get_miou_from_txt(f'./results/result_best_model{model_suffix}_{tag}.txt')

        row = {
            'Distortion': test_d,
            'CleanModel-CleanData': acc_clean_on_clean,
            'CleanModel-DistData': acc_clean_on_dist,
            'DistModel-CleanData': acc_model_on_clean,
            'DistModel-DistData': acc_model_on_dist,
            'Robustness (DistModel/DistData ÷ CleanModel/CleanData)': 
                round(acc_model_on_dist / acc_clean_on_clean, 3) if acc_clean_on_clean and acc_model_on_dist else None
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    save_path = f'./results/robustness_{dist_type}.csv'
    df.to_csv(save_path, index=False)
    print(f"✅ Saved {save_path}")

# Run all
compute_robustness("light", categories["light"])
compute_robustness("medium", categories["medium"])
compute_robustness("heavy", categories["heavy"])
