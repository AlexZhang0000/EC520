#!/bin/bash
set -e

echo "=============== TRAINING PHASE ==============="

# --- Clean training ---
 python train.py


# --- Aliasing ---
 python train.py --train_distortion aliasing:8

# --- JPEG Compression ---
 python train.py --train_distortion jpegcompression:5

echo "=============== TESTING PHASE ==============="

# Clean model → clean data
python test.py --model_suffix _clean

# Clean model → distorted datasets
python test.py --model_suffix _clean --test_distortion aliasing:8
python test.py --model_suffix _clean --test_distortion jpegcompression:5

# Distorted model → clean data
python test.py --model_suffix _aliasing:8
python test.py --model_suffix _jpegcompression:5

# Distorted model → same distortion dataset
python test.py --model_suffix _aliasing:8 --test_distortion aliasing:8
python test.py --model_suffix _jpegcompression:5 --test_distortion jpegcompression:5

echo "✅ All training and testing complete."
