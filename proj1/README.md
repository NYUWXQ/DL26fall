# Custom Sub-5M Parameter Residual Network for CIFAR-10 (Deep Learning Project 1)

This repository contains a **100% from-scratch PyTorch implementation** of a modified Residual Network (`CustomResNet`) designed for the Fall 2026 Deep Learning Project 1 Kaggle competition on CIFAR-10 under the strict **5,000,000 trainable parameter ceiling**.

## 1. Repository & Dataset Structure

Place your competition image folders (`train_images/` and `test_images/`) and CSV metadata files (`train_labels.csv`, `test.csv`, `sample_submission.csv`) in the repository root (or pass their paths via CLI flags):

```text
cifar10-resnet-sub5m/
├── README.md                       # Project overview, architecture table, and reproduction instructions
├── requirements.txt                # Python package dependencies (torch, torchvision, Pillow, pandas, etc.)
├── .gitignore                      # Git ignore rules for image folders, datasets, and checkpoints
├── sample_submission.csv           # Official competition submission template (id,label)
├── train_labels.csv                # (Place here) 50,000 training labels: id,filename,label
├── test.csv                        # (Place here) 2,000 test image filenames: id,filename
├── train_images/                   # (Place here) Folder of 50,000 PNGs: train_00000.png .. train_49999.png
├── test_images/                    # (Place here) Folder of 2,000 PNGs:  test_0000.png .. test_1999.png
├── configs/
│   └── model_configs.py            # Parameterized ResNetSpec definitions (N, B_i, C_i, F_i, K_i, P)
├── src/
│   ├── __init__.py
│   ├── model.py                    # From-scratch ResidualUnit & CustomResNet + parameter audit
│   ├── dataset.py                  # CIFARCSVDataset (train_images/ + CSVs), Cutout, & torchvision fallback
│   ├── trainer.py                  # Nesterov SGD + Warmup/Cosine LR trainer, evaluator, & TTA
│   └── visualize.py                # Automated report plotting (loss/acc curves, per-class bar chart)
├── train.py                        # CLI entrypoint for training, validation, ablations, & submission export
├── predict_kaggle.py               # Standalone CLI entrypoint to generate `submission.csv` (`id,label`)
└── notebooks/
    └── project1_walkthrough.ipynb  # Self-contained Jupyter Notebook for Colab / TA verification
```

### How the CSV Files & Image Folders Connect
- **`train_labels.csv` (`id,filename,label`) + `train_images/`**: Maps each training image filename (`train_00000.png` through `train_49999.png`) to its ground-truth class integer (`0`–`9`). `src/dataset.py` (`CIFARCSVDataset` + `build_csv_folder_loaders`) performs a stratified train/validation split (`90%` training with RandomCrop + RandomHorizontalFlip + Cutout, `10%` validation with clean normalization). If `train_images/` is not present locally, the loader automatically falls back to `torchvision.datasets.CIFAR10`.
- **`test.csv` (`id,filename`) + `test_images/`**: Lists the 2,000 unlabeled competition evaluation images (`test_0000.png` through `test_1999.png`) in exact submission order (`id` `0` to `1999`).
- **`sample_submission.csv` / `submission.csv` (`id,label`)**: Contains the final predicted class indices (`0`–`9`) paired with each `id` from `test.csv`.

## 2. Architectural Design (`<= 5,000,000` Trainable Parameters)

Every residual unit implements the exact project formulation from scratch:
$$\text{Output} = \text{ReLU}\big(S(x) + F(x)\big)$$
where $F(x)$ applies `Conv2d(F_i x F_i) -> BatchNorm2d -> ReLU -> Conv2d(F_i x F_i) -> BatchNorm2d`, and $S(x)$ is either the identity mapping or a `Conv2d(K_i x K_i, stride) -> BatchNorm2d` projection shortcut.

| Architecture ID | Stages ($N$) | Channels ($C_i$) | Blocks ($B_i$) | Filters ($F_i$) | Skip ($K_i$) | Pool ($P$) | Trainable Parameters | % of 5M Budget |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_4stage` | 4 | `[64, 128, 192, 256]` | `[2, 2, 2, 2]` | `[3, 3, 3, 3]` | `[1, 1, 1, 1]` | 4 | **4,184,522** | 83.69% |
| `wide_3stage` | 3 | `[64, 144, 264]` | `[3, 3, 3]` | `[3, 3, 3]` | `[1, 1, 1]` | 8 | **4,773,930** | 95.48% |
| `skip3x3_4stage` | 4 | `[64, 128, 192, 256]` | `[2, 2, 2, 2]` | `[3, 3, 3, 3]` | `[1, 3, 3, 3]` | 4 | **4,839,882** | 96.80% |
| **`champion_sub5m`** | **4** | **`[64, 128, 208, 264]`** | **`[3, 3, 2, 2]`** | **`[3, 3, 3, 3]`** | **`[1, 1, 1, 1]`** | **4** | **4,917,450** | **98.35%** |

## 3. Quickstart & Reproduction Instructions

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Champion Model (`4,917,450` Parameters) on `train_images/` + `train_labels.csv`
```bash
python3 train.py \
    --arch champion_sub5m \
    --train-img-dir ./train_images \
    --train-csv ./train_labels.csv \
    --test-img-dir ./test_images \
    --test-csv ./test.csv \
    --epochs 100 \
    --batch-size 128
```
This script automatically:
1. Audits and prints the exact trainable parameter count (`4,917,450`) and `torchsummary.summary(model, (3, 32, 32))`.
2. Loads `train_images/` and `train_labels.csv` using `CIFARCSVDataset` (with a 90/10 stratified train/val split, or falls back to `torchvision.datasets.CIFAR10` if `train_images/` is absent) and saves the best checkpoint to `./outputs/champion_sub5m_best.pth`.
3. Prints the **Final Test/Validation Accuracy** (both standard and with horizontal-flip Test-Time Augmentation) and saves `champion_sub5m_training_curves.png` and `champion_sub5m_per_class_accuracy.png`.
4. If `test_images/` and `test.csv` are present, automatically runs TTA inference and outputs `submission.csv` (`id,label`).

### Run the 4-Model Architectural Ablation Study
```bash
python3 train.py --run-ablation --ablation-epochs 25
```

### Generate Kaggle Competition `submission.csv` (`id,label`) Standalone
```bash
python3 predict_kaggle.py \
    --checkpoint ./outputs/champion_sub5m_best.pth \
    --test-dir ./test_images \
    --test-csv ./test.csv \
    --sample-submission ./sample_submission.csv \
    --output submission.csv
```
