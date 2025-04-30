#!/bin/bash

# Distortion参数
declare -a distortions=(
    "gaussianblur:3,0.8"
    "gaussianblur:5,2.0"
    "gaussianblur:9,4.0"
    "gaussiannoise:0,0.02"
    "gaussiannoise:0,0.1"
    "gaussiannoise:0,0.25"
    "aliasing:2"
    "aliasing:4"
    "aliasing:8"
    "jpegcompression:60"
    "jpegcompression:20"
    "jpegcompression:5"
)

# ============ TRAINING ============
echo "🚀 Training clean model..."
python train.py

for d in "${distortions[@]}"; do
    echo "🚀 Training with distortion: $d"
    python train.py --train_distortion "$d"
done

# ============ TESTING ============

# 1. clean model + clean data
echo "🧪 Testing clean model on clean data"
python test.py --model_path saved_models/best_model.pth

# 2. clean model + 12 distorted datasets
for d in "${distortions[@]}"; do
    echo "🧪 Testing clean model on distorted data: $d"
    python test.py --model_path saved_models/best_model.pth --distortion "$d"
done

# 3. 12 distorted models + clean data
for d in "${distortions[@]}"; do
    suffix=$(echo "$d" | sed 's/:/_/g' | sed 's/\//_/g')
    echo "🧪 Testing distorted model ($d) on clean data"
    python test.py --model_path "saved_models/best_model_distorted_${suffix}.pth"
done

# 4. 12 distorted models + corresponding distorted data
for d in "${distortions[@]}"; do
    suffix=$(echo "$d" | sed 's/:/_/g' | sed 's/\//_/g')
    echo "🧪 Testing distorted model ($d) on same distorted data"
    python test.py --model_path "saved_models/best_model_distorted_${suffix}.pth" --distortion "$d"
done

echo "✅ All training and testing complete."
