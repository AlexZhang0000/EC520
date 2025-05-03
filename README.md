# EC520 Image Processing Project

## 📌 Overview

* Course: EC520 – Multimedia Systems and Communication
* University: Boston University
* Semester: Spring 2025
* Author: Xushuai Zhang
* Description: This project includes three main image processing tasks: Image Classification, Image Segmentation, and Object Detection. Each task is implemented in a separate module using PyTorch and follows a clear structure for training, testing, and evaluation.

---

## 📁 Project Structure

```
EC520-main/
├── image_recognition/         # Image classification
├── image_segmentation/        # Image segmentation
├── image_objectiveDection/    # Object detection
└── README.md                  # Top-level documentation
```

Each module contains:

* `train.py`: Model training script
* `test.py`: Model evaluation script
* `model.py`: Network architecture definition
* `dataloader.py`: Dataset loading and preprocessing
* `results.sh`: Shell script for reproducible experiment commands
* `plot.py`: Visualization and plotting utilities
* `table.py`: Generates tabular summaries of evaluation metrics
* `requirements.txt`: Python package dependencies
* `ReadMe.md`: Module-specific documentation and instructions

---

## 🔧 Installation

* Python 3.8+ is required
* Install dependencies per module:

```bash
cd [MODULE_NAME]
pip install -r requirements.txt
```

> Tip: Use a virtual environment (`venv` or `conda`) for isolation.

---

## 🧐 Module Descriptions

### 1. image\_recognition/

* **Task**: Classify images into predefined categories (e.g., CIFAR-10)
* **Key Files**:

  * `train.py`: Train the classification model
  * `test.py`: Evaluate model accuracy
  * `model.py`: Model architecture
  * `dataloader.py`: Dataset loader
  * `results.sh`: Run standard training/testing pipelines
  * `plot.py`: Plot accuracy/loss curves
  * `table.py`: Generate CSV tables of performance metrics
* **Output**: Accuracy metrics, plots, CSV tables saved in `./results/`

---

### 2. image\_segmentation/

* **Task**: Perform pixel-wise semantic segmentation using U-Net
* **Key Features**:

  * Image augmentation
  * IoU (Intersection over Union) evaluation
* **Key Files**:

  * `train.py`, `test.py`, `model.py`, `dataloader.py`
  * `results.sh`, `plot.py`, `table.py` (as above)
* **Output**: Predicted segmentation masks and evaluation summaries

---

### 3. image\_objectiveDection/

* **Task**: Detect objects in images and classify them
* **Key Features**:

  * Bounding box prediction
  * Robustness against distortions (Gaussian noise, blur, JPEG, aliasing)
* **Key Files**:

  * `train.py`, `test.py`, `model.py`, `dataloader.py`, `utils.py`
  * `results.sh`, `plot.py`, `table.py`
* **Output**: Annotated bounding boxes and metrics saved to `./results/`

---

## 🚀 Quick Start

Example for running classification module:

```bash
cd image_recognition
bash results.sh        # Run training and testing pipeline
python plot.py         # Plot training curves
python table.py        # Output final metrics in table form
```

> Detailed commands and hyperparameters are provided in each module's `ReadMe.md`.

---

## 📊 Results

* Each module saves results in its `results/` folder
* Evaluation metrics include:

  * Accuracy (Classification)
  * IoU Score (Segmentation)
  * Detection Accuracy / mAP (Object Detection)
* Visualization plots and metric tables provided

---

## ⚙️ Notes

* Using GPU is highly recommended for training
* Batch size and number of workers can be configured in `config.py`
* Multi-GPU training supported using PyTorch’s `DataParallel`
* Optimized for environments like BU SCC and Lambda Labs

---

## 📬 Contact

* **Name**: Xushuai Zhang
* **Email**: \[Please insert your contact email here]
* Feel free to reach out for collaboration or questions.
