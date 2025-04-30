#!/bin/bash
set -e

echo "=============== TRAINING PHASE ==============="

# --- Clean training ---
# python train.py

# --- Gaussian Blur ---
# python train.py --train_distortion gaussianblur:3,0.8
# python train.py --train_distortion gaussianblur:5,2.0
# python train.py --train_distortion gaussianblur:9,4.0

# --- Gaussian Noise ---
# python train.py --train_distortion gaussiannoise:0,0.02
# python train.py --train_distortion gaussiannoise:0,0.1
# python train.py --train_distortion gaussiannoise:0,0.25

# --- Aliasing ---
# python train.py --train_distortion aliasing:2
# python train.py --train_distortion aliasing:4
# python train.py --train_distortion aliasing:8

# --- JPEG Compression ---
# python train.py --train_distortion jpegcompression:60
# python train.py --train_distortion jpegcompression:20
# python train.py --train_distortion jpegcompression:5

echo "=============== TESTING PHASE ==============="

# Clean model → Clean data
python test.py --model_path saved_models/best_model.pth

# Clean model → Distorted data
python test.py --model_path saved_models/best_model.pth --distortion gaussianblur:3,0.8
python test.py --model_path saved_models/best_model.pth --distortion gaussianblur:5,2.0
python test.py --model_path saved_models/best_model.pth --distortion gaussianblur:9,4.0
python test.py --model_path saved_models/best_model.pth --distortion gaussiannoise:0,0.02
python test.py --model_path saved_models/best_model.pth --distortion gaussiannoise:0,0.1
python test.py --model_path saved_models/best_model.pth --distortion gaussiannoise:0,0.25
python test.py --model_path saved_models/best_model.pth --distortion aliasing:2
python test.py --model_path saved_models/best_model.pth --distortion aliasing:4
python test.py --model_path saved_models/best_model.pth --distortion aliasing:8
python test.py --model_path saved_models/best_model.pth --distortion jpegcompression:60
python test.py --model_path saved_models/best_model.pth --distortion jpegcompression:20
python test.py --model_path saved_models/best_model.pth --distortion jpegcompression:5

# Distorted model → Clean data
python test.py --model_path saved_models/best_model_gaussianblur_3,0.8.pth
python test.py --model_path saved_models/best_model_gaussianblur_5,2.0.pth
python test.py --model_path saved_models/best_model_gaussianblur_9,4.0.pth
python test.py --model_path saved_models/best_model_gaussiannoise_0,0.02.pth
python test.py --model_path saved_models/best_model_gaussiannoise_0,0.1.pth
python test.py --model_path saved_models/best_model_gaussiannoise_0,0.25.pth
python test.py --model_path saved_models/best_model_aliasing_2.pth
python test.py --model_path saved_models/best_model_aliasing_4.pth
python test.py --model_path saved_models/best_model_aliasing_8.pth
python test.py --model_path saved_models/best_model_jpegcompression_60.pth
python test.py --model_path saved_models/best_model_jpegcompression_20.pth
python test.py --model_path saved_models/best_model_jpegcompression_5.pth

# Distorted model → Distorted data
python test.py --model_path saved_models/best_model_gaussianblur_3,0.8.pth --distortion gaussianblur:3,0.8
python test.py --model_path saved_models/best_model_gaussianblur_5,2.0.pth --distortion gaussianblur:5,2.0
python test.py --model_path saved_models/best_model_gaussianblur_9,4.0.pth --distortion gaussianblur:9,4.0
python test.py --model_path saved_models/best_model_gaussiannoise_0,0.02.pth --distortion gaussiannoise:0,0.02
python test.py --model_path saved_models/best_model_gaussiannoise_0,0.1.pth --distortion gaussiannoise:0,0.1
python test.py --model_path saved_models/best_model_gaussiannoise_0,0.25.pth --distortion gaussiannoise:0,0.25
python test.py --model_path saved_models/best_model_aliasing_2.pth --distortion aliasing:2
python test.py --model_path saved_models/best_model_aliasing_4.pth --distortion aliasing:4
python test.py --model_path saved_models/best_model_aliasing_8.pth --distortion aliasing:8
python test.py --model_path saved_models/best_model_jpegcompression_60.pth --distortion jpegcompression:60
python test.py --model_path saved_models/best_model_jpegcompression_20.pth --distortion jpegcompression:20
python test.py --model_path saved_models/best_model_jpegcompression_5.pth --distortion jpegcompression:5


echo "✅ All training and testing complete."

