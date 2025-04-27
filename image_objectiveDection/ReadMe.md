# Distortion Usage Guide

This document explains how to **train** and **test** your object detection project (YOLOv5 + Pascal VOC 5-class subset) under different distortions.

---

## 📋 Table of Contents

- [1. Gaussian Blur](#1-gaussian-blur)
- [2. Gaussian Noise](#2-gaussian-noise)
- [3. Aliasing](#3-aliasing)
- [4. JPEG Compression](#4-jpeg-compression)
- [5. Clean Model Testing](#5-clean-model-testing)
- [6. Combining Multiple Distortions](#6-combining-multiple-distortions)

---

## 1. Gaussian Blur

**Description**:  
Apply Gaussian blur with kernel size 5 and sigma 2.0 to input images.

**Training Command**:
```bash
python train.py --train_distortion gaussianblur:5,2.0
```

**Testing Command**:
```bash
python test.py --model_suffix _distorted_gaussianblur_5_2.0 --test_distortion gaussianblur:5,2.0
```

---

## 2. Gaussian Noise

**Description**:  
Add random Gaussian noise with mean 0 and standard deviation 0.1 to input images.

**Training Command**:
```bash
python train.py --train_distortion gaussiannoise:0,0.1
```

**Testing Command**:
```bash
python test.py --model_suffix _distorted_gaussiannoise_0_0.1 --test_distortion gaussiannoise:0,0.1
```

---

## 3. Aliasing

**Description**:  
Downscale the image by a factor of 4 and then upscale back using nearest neighbor interpolation, introducing aliasing artifacts.

**Training Command**:
```bash
python train.py --train_distortion aliasing:4
```

**Testing Command**:
```bash
python test.py --model_suffix _distorted_aliasing_4 --test_distortion aliasing:4
```

---

## 4. JPEG Compression

**Description**:  
Apply JPEG compression with quality factor 20, introducing compression artifacts.

**Training Command**:
```bash
python train.py --train_distortion jpegcompression:20
```

**Testing Command**:
```bash
python test.py --model_suffix _distorted_jpegcompression_20 --test_distortion jpegcompression:20
```

---

## 5. Clean Model Testing

You can also test models **without any distortion** using the following command:

```bash
python test.py --model_suffix _clean
```

---

## 6. Combining Multiple Distortions

You can combine multiple distortions using `/` as a separator.  
For example, to apply **Gaussian Blur** followed by **Gaussian Noise** during training:

```bash
python train.py --train_distortion gaussianblur:5,2.0/gaussiannoise:0,0.1
```

---
