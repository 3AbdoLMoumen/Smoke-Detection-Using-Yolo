# Wildfire and Smoke Detection Model Training with YOLO26

## Overview

This repository documents the training procedure for a YOLO26-based object detection model designed to identify smoke and wildfire regions in long-distance aerial and ground-based imagery. The workflow encompasses dataset preparation, class-imbalance mitigation via targeted oversampling, model training, and quantitative evaluation of detection performance.

The resulting model is intended for integration into downstream smoke and fire monitoring pipelines, including video-based growth analysis systems.

## Dataset

Training data is drawn from the Long-Distance Wildfire & Smoke Detection Dataset, which contains two annotated classes:

| Class ID | Class Name |
|---|---|
| 0 | smoke |
| 1 | wildfire |

### Citation

```bibtex
@dataset{simuletic_wildfire_smoke_2026,
  author    = {Simuletic Team},
  title     = {Long-Range Wildfire \& Smoke Detection Dataset},
  year      = {2026},
  publisher = {Kaggle},
  url       = {https://simuletic.com/datasets}
}
```

## Data Preparation

### Train/Validation Split

Images and corresponding YOLO-format label files were partitioned into training and validation subsets using an 80/20 split, with a fixed random seed (42) to ensure reproducibility. File association between images and labels was performed via symbolic linking to avoid data duplication on disk.

| Split | Images |
|---|---|
| Train | 191 |
| Validation | 48 |

All images were confirmed to have a corresponding label file (zero missing labels).

### Class Imbalance and Oversampling

Initial inspection of the training split revealed a substantial class imbalance between the two annotated categories:

| Class | Instances (pre-oversampling) |
|---|---|
| smoke (0) | 204 |
| wildfire (1) | 58 |

To mitigate this imbalance, training images containing at least one wildfire annotation were identified (38 images) and oversampled by a factor of three via symbolic-link duplication of both the image and its corresponding label file. This procedure added 76 duplicated wildfire-containing images to the training set without altering the underlying data.

Following oversampling, class instance counts in the training split were as follows:

| Class | Instances (post-oversampling) |
|---|---|
| smoke (0) | 286 |
| wildfire (1) | 174 |

### Dataset Configuration

A `data.yaml` configuration file was generated to define dataset paths and class names for use with the Ultralytics training pipeline:

```yaml
path: /kaggle/working/dataset
train: images/train
val: images/val
names:
  0: smoke
  1: wildfire
```

## Model and Training Configuration

The model was initialized from pretrained YOLO26-small (`yolo26s.pt`) weights and fine-tuned using the Ultralytics training API. Key training parameters are summarized below:

| Parameter | Value |
|---|---|
| Base model | YOLO26s (9.9M parameters) |
| Epochs | 150 (max), early stopping at 148 |
| Image size | 960 × 960 |
| Devices | 2 × NVIDIA Tesla T4 (multi-GPU, DDP) |
| Classification loss weight (`cls`) | 2.0 |
| Early stopping patience | 40 epochs |
| Scale augmentation | 0.5 |
| Copy-paste augmentation | 0.3 |
| Mixup augmentation | 0.15 |
| Optimizer | AdamW (auto-selected), lr0 = 0.001667, momentum = 0.9 |

The elevated classification loss weight was applied to further emphasize correct class discrimination given the underlying class imbalance, complementing the oversampling strategy described above.

Training completed in approximately 0.415 hours (148 epochs), with early stopping triggered after 40 epochs without improvement. The best-performing checkpoint was obtained at epoch 108 and saved as `best.pt`.

## Results

### Overall Validation Performance (Best Checkpoint)

| Metric | Value |
|---|---|
| Precision | 0.507 |
| Recall | 0.547 |
| mAP50 | 0.514 |
| mAP50-95 | 0.208 |

### Per-Class Performance (Best Checkpoint)

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| smoke | 48 | 51 | 0.755 | 0.863 | 0.852 | 0.358 |
| wildfire | 9 | 13 | 0.259 | 0.231 | 0.175 | 0.059 |

### Post-Training Validation Pass (conf = 0.1)

A subsequent validation pass was conducted at a lowered confidence threshold (0.1) to characterize performance under more permissive detection conditions:

| Metric | Value |
|---|---|
| Precision | 0.5196 |
| Recall | 0.5468 |
| mAP50 | 0.5019 |
| mAP50-95 | 0.2020 |

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| smoke | 0.7796 | 0.8627 | 0.8506 | 0.3520 |
| wildfire | 0.2596 | 0.2308 | 0.1532 | 0.0521 |

### Interpretation

Detection performance for the smoke class is substantially stronger than for the wildfire class across all metrics, reflecting both the smaller number of wildfire instances available for training and validation and the inherent visual difficulty of distinguishing flame regions at long range. Despite oversampling, the wildfire class remains data-limited (13 validation instances), and reported metrics for this class should be interpreted with corresponding caution regarding statistical reliability.

## Repository Structure

```
.
├── data.yaml                          # Dataset configuration
├── dataset/                           # Prepared train/val image and label symlinks
├── runs/detect/train/                 # Training outputs (weights, logs, plots)
│   └── weights/
│       ├── best.pt                    # Best checkpoint (epoch 108)
│       └── last.pt                    # Final checkpoint
└── archive.zip                        # Compressed archive of training run outputs
```

## Requirements

```bash
pip install ultralytics pandas pyyaml
```

## Usage

### Training

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")
results = model.train(
    data="data.yaml",
    epochs=150,
    imgsz=960,
    device=[0, 1],
    cls=2.0,
    patience=40,
    scale=0.5,
    copy_paste=0.3,
    mixup=0.15,
)
```

### Validation

```python
metrics = model.val(data="data.yaml", split="val", conf=0.1)
```

## Limitations

- The wildfire class is represented by a comparatively small number of instances (58 pre-oversampling, 174 post-oversampling in training; 13 in validation), limiting the statistical robustness of reported wildfire detection metrics.
- Oversampling was performed via duplication of existing wildfire-containing images rather than synthetic data generation; this addresses class frequency but does not introduce additional visual diversity for the underrepresented class.
- Validation set size (48 images, 64 total instances) is modest, and reported metrics should be considered indicative rather than definitive of generalization performance on unseen data.

## License

No license has been specified for this repository. Users should consult the dataset publisher and repository owner regarding permitted use, modification, and distribution of the code and data.
