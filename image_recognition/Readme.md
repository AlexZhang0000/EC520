# Distortion Usage Guide

This document shows how to train and test with different distortions in the project.

Each distortion has a training command and a testing command.

---

## 1. Gaussian Blur

- **Training Command**:

```bash
python train.py --train_distortion gaussianblur:5,2.0
```

- **Testing Command**:

```bash
python test.py --model_suffix _distorted_gaussianblur_5_2.0 --test_distortion gaussianblur:5,2.0
```

- **Description**:  
  Apply Gaussian blur with kernel size 5 and sigma 2.0 to images.

---

## 2. Gaussian Noise

- **Training Command**:

```bash
python train.py --train_distortion gaussiannoise:0,0.1
```

- **Testing Command**:

```bash
python test.py --model_suffix _distorted_gaussiannoise_0_0.1 --test_distortion gaussiannoise:0,0.1
```

- **Description**:  
  Add random Gaussian noise to images with mean 0 and standard deviation 0.1.

---

## 3. Aliasing

- **Training Command**:

```bash
python train.py --train_distortion aliasing:4
```

- **Testing Command**:

```bash
python test.py --model_suffix _distorted_aliasing_4 --test_distortion aliasing:4
```

- **Description**:  
  Downscale image by factor 4 and then upscale back using nearest neighbor interpolation, causing aliasing artifacts.

---

## 4. JPEG Compression

- **Training Command**:

```bash
python train.py --train_distortion jpegcompression:20
```

- **Testing Command**:

```bash
python test.py --model_suffix _distorted_jpegcompression_20 --test_distortion jpegcompression:20
```

- **Description**:  
  Apply JPEG compression to images with quality factor 20, introducing compression artifacts.

---

## Notes

- You can also test clean models without distortion:

```bash
python test.py --model_suffix _clean
```

- Multiple distortions can be combined using `/`, for example:

```bash
python train.py --train_distortion gaussianblur:5,2.0/gaussiannoise:0,0.1
```
