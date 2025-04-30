#!/bin/bash
set -e

echo "🟩 TRAINING PHASE"
# 1. Train clean model
# python train.py

# 2. Train Gaussian Blur models
# python train.py --train_distortion gaussianblur:3,0.8
# python train.py --train_distortion gaussianblur:5,2.0
# python train.py --train_distortion gaussianblur:9,4.0

# 3. Train Gaussian Noise models
# python train.py --train_distortion gaussiannoise:0,0.02
# python train.py --train_distortion gaussiannoise:0,0.1
# python train.py --train_distortion gaussiannoise:0,0.25

# 4. Train Aliasing models
# python train.py --train_distortion aliasing:2
# python train.py --train_distortion aliasing:4
# python train.py --train_distortion aliasing:8

# 5. Train JPEG Compression models
# python train.py --train_distortion jpegcompression:60
# python train.py --train_distortion jpegcompression:20
# python train.py --train_distortion jpegcompression:5

echo "🟦 TESTING PHASE"
# 1. Test clean model on clean data
python test.py --model_suffix _clean

# 2. Test clean model on all distortions
python test.py --model_suffix _clean --test_distortion gaussianblur:3,0.8
python test.py --model_suffix _clean --test_distortion gaussianblur:5,2.0
python test.py --model_suffix _clean --test_distortion gaussianblur:9,4.0

python test.py --model_suffix _clean --test_distortion gaussiannoise:0,0.02
python test.py --model_suffix _clean --test_distortion gaussiannoise:0,0.1
python test.py --model_suffix _clean --test_distortion gaussiannoise:0,0.25

python test.py --model_suffix _clean --test_distortion aliasing:2
python test.py --model_suffix _clean --test_distortion aliasing:4
python test.py --model_suffix _clean --test_distortion aliasing:8

python test.py --model_suffix _clean --test_distortion jpegcompression:60
python test.py --model_suffix _clean --test_distortion jpegcompression:20
python test.py --model_suffix _clean --test_distortion jpegcompression:5

# 3. Test distorted models on clean data
python test.py --model_suffix _distorted_gaussianblur_3_0.8
python test.py --model_suffix _distorted_gaussianblur_5_2.0
python test.py --model_suffix _distorted_gaussianblur_9_4.0

python test.py --model_suffix _distorted_gaussiannoise_0_0.02
python test.py --model_suffix _distorted_gaussiannoise_0_0.1
python test.py --model_suffix _distorted_gaussiannoise_0_0.25

python test.py --model_suffix _distorted_aliasing_2
python test.py --model_suffix _distorted_aliasing_4
python test.py --model_suffix _distorted_aliasing_8

python test.py --model_suffix _distorted_jpegcompression_60
python test.py --model_suffix _distorted_jpegcompression_20
python test.py --model_suffix _distorted_jpegcompression_5

# 4. Test distorted models on their own distortions
python test.py --model_suffix _distorted_gaussianblur_3_0.8 --test_distortion gaussianblur:3,0.8
python test.py --model_suffix _distorted_gaussianblur_5_2.0 --test_distortion gaussianblur:5,2.0
python test.py --model_suffix _distorted_gaussianblur_9_4.0 --test_distortion gaussianblur:9,4.0

python test.py --model_suffix _distorted_gaussiannoise_0_0.02 --test_distortion gaussiannoise:0,0.02
python test.py --model_suffix _distorted_gaussiannoise_0_0.1 --test_distortion gaussiannoise:0,0.1
python test.py --model_suffix _distorted_gaussiannoise_0_0.25 --test_distortion gaussiannoise:0,0.25

python test.py --model_suffix _distorted_aliasing_2 --test_distortion aliasing:2
python test.py --model_suffix _distorted_aliasing_4 --test_distortion aliasing:4
python test.py --model_suffix _distorted_aliasing_8 --test_distortion aliasing:8

python test.py --model_suffix _distorted_jpegcompression_60 --test_distortion jpegcompression:60
python test.py --model_suffix _distorted_jpegcompression_20 --test_distortion jpegcompression:20
python test.py --model_suffix _distorted_jpegcompression_5 --test_distortion jpegcompression:5

echo "✅ All training and testing completed."
