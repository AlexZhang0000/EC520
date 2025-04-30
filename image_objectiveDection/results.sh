#!/bin/bash
set -e

echo "=============== TRAINING PHASE ==============="

# --- Clean training ---
python train.py

# --- Gaussian Blur ---
python train.py --train_distortion gaussianblur:9,4.0

# --- Gaussian Noise ---
python train.py --train_distortion gaussiannoise:0,0.25

# --- Aliasing ---
python train.py --train_distortion aliasing:8

# --- JPEG Compression ---
python train.py --train_distortion jpegcompression:5

echo "=============== TESTING PHASE ==============="

# Clean model → clean data
python test.py --model_suffix _clean

# Clean model → distorted datasets

python test.py --model_suffix _clean --test_distortion gaussianblur:9,4.0

python test.py --model_suffix _clean --test_distortion gaussiannoise:0,0.25

python test.py --model_suffix _clean --test_distortion aliasing:8

python test.py --model_suffix _clean --test_distortion jpegcompression:5

# Distorted model → clean data

python test.py --model_suffix _distorted_gaussianblur_9_4.0

python test.py --model_suffix _distorted_gaussiannoise_0_0.25

python test.py --model_suffix _distorted_aliasing_8

python test.py --model_suffix _distorted_jpegcompression_5

# Distorted model → same distortion dataset

python test.py --model_suffix _distorted_gaussianblur_9_4.0 --test_distortion gaussianblur:9,4.0

python test.py --model_suffix _distorted_gaussiannoise_0_0.25 --test_distortion gaussiannoise:0,0.25

python test.py --model_suffix _distorted_aliasing_8 --test_distortion aliasing:8

python test.py --model_suffix _distorted_jpegcompression_5 --test_distortion jpegcompression:5

echo "✅ All training and testing complete."
