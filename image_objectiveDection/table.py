import os
import pandas as pd

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

distortions = [
    ("aliasing:8", "_distorted_aliasing_8"),
    ("jpegcompression:5", "_distorted_jpegcompression_5")
]

acc_clean_on_clean = get_miou_from_txt('./results/test_log_clean.txt')
rows = []

for test_d, model_suffix in distortions:
    tag = test_d.replace(":", "_").replace(",", "_")
    acc_clean_on_dist = get_miou_from_txt(f'./results/test_log_clean_{tag}.txt')
    acc_model_on_clean = get_miou_from_txt(f'./results/test_log{model_suffix}_clean.txt')
    acc_model_on_dist = get_miou_from_txt(f'./results/test_log{model_suffix}_{tag}.txt')

    robustness = round(acc_model_on_dist / acc_clean_on_clean, 3) if acc_clean_on_clean and acc_model_on_dist else None

    rows.append({
        'Distortion': test_d,
        'CleanModel-CleanData': acc_clean_on_clean,
        'CleanModel-DistData': acc_clean_on_dist,
        'DistModel-CleanData': acc_model_on_clean,
        'DistModel-DistData': acc_model_on_dist,
        'Robustness (DistModel/DistData ÷ CleanModel/CleanData)': robustness
    })

pd.DataFrame(rows).to_csv('./results/robustness_aliasing_jpeg.csv', index=False)
print("✅ Saved ./results/robustness_aliasing_jpeg.csv")
