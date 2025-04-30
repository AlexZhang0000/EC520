#!/bin/bash
set -e

echo "📊 Generating robustness CSVs..."

python <<EOF
import os
import pandas as pd

def get_miou(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if 'mIoU:' in line:
                return float(line.strip().split(':')[1])
    return None

def distortion_tag(d):  # e.g. "gaussianblur:3,0.8" → "gaussianblur_3_0.8"
    return d.replace(":", "_").replace(",", "_")

categories = {
    "light": [
        "gaussianblur:3,0.8",
        "gaussiannoise:0,0.02",
        "aliasing:2",
        "jpegcompression:60"
    ],
    "medium": [
        "gaussianblur:5,2.0",
        "gaussiannoise:0,0.1",
        "aliasing:4",
        "jpegcompression:20"
    ],
    "heavy": [
        "gaussianblur:9,4.0",
        "gaussiannoise:0,0.25",
        "aliasing:8",
        "jpegcompression:5"
    ]
}

def compute_robustness(dist_list, level):
    rows = []
    clean_clean = get_miou('./results/result_clean_clean.txt')

    for d in dist_list:
        tag = distortion_tag(d)
        clean_dist = get_miou(f'./results/result_clean_{tag}.txt')
        dist_clean = get_miou(f'./results/result_{tag}_clean.txt')
        dist_dist = get_miou(f'./results/result_{tag}_{tag}.txt')

        row = {
            'Distortion': d,
            'CleanModel-CleanData': clean_clean,
            'CleanModel-DistData': clean_dist,
            'DistModel-CleanData': dist_clean,
            'DistModel-DistData': dist_dist,
            'Robustness (DistModel/DistData ÷ CleanModel/CleanData)': round(dist_dist / clean_clean, 3) if dist_dist and clean_clean else None
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f'./results/robustness_{level}.csv', index=False)
    print(f"✅ Saved ./results/robustness_{level}.csv")

for level in ["light", "medium", "heavy"]:
    compute_robustness(categories[level], level)
EOF

echo "✅ All robustness CSVs generated."
