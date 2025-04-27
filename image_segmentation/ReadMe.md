# Image Segmentation with Distortions (U-Net + Pascal VOC 2012)

This project implements an image segmentation model based on U-Net, trained and tested on the PASCAL VOC 2012 dataset.  
Various data distortions can be applied during training and testing to study their effects.

---

## Supported Distortions

| Distortion Type | Format Example | Description |
|:---|:---|:---|
| Gaussian Blur | `gaussianblur:5,2.0` | Apply Gaussian blur with kernel size 5 and sigma 2.0 |
| Gaussian Noise | `gaussiannoise:0,0.1` | Add random Gaussian noise with mean 0, std 0.1 |
| Aliasing | `aliasing:4` | Downscale by factor 4, upscale back using nearest neighbor interpolation |
| JPEG Compression | `jpegcompression:20` | Apply JPEG compression with quality factor 20 |

---

## Training and Testing Examples

Below are the commands for different distortion settings:

---

### 1. Clean Train → Clean Test

- **Training**:

```bash
python train.py
```

- **Testing**:

```bash
python test.py
```

---

### 2. Clean Train → Distorted Test (e.g., Gaussian Noise)

- **Training**:

```bash
python train.py
```

- **Testing**:

```bash
python test.py --test_distortion gaussiannoise:0,0.1
```

---

### 3. Distorted Train → Clean Test (e.g., Aliasing)

- **Training**:

```bash
python train.py --train_distortion aliasing:4
```

- **Testing**:

```bash
python test.py --model_suffix _distorted_aliasing_4
```

---

### 4. Distorted Train → Distorted Test (e.g., JPEG Compression)

- **Training**:

```bash
python train.py --train_distortion jpegcompression:20
```

- **Testing**:

```bash
python test.py --model_suffix _distorted_jpegcompression_20 --test_distortion jpegcompression:20
```

---

## Notes

- Models are saved under `/saved_models/`, e.g., `best_model_clean.pth`, `best_model_distorted_gaussianblur_5_2.0.pth`.
- Test results (mean IoU and per-class IoUs) are saved under `/results/`.
- Pascal VOC 2012 dataset will be automatically downloaded into `/Data/` if not found.
- Multiple distortions can be combined with `/` separator:

```bash
python train.py --train_distortion gaussianblur:5,2.0/gaussiannoise:0,0.1
```

---

## Environment Requirements

Install required libraries:

```bash
pip install -r requirements.txt
```

---

