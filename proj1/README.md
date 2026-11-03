# Custom Sub-5M Parameter Residual Network for CIFAR-10 — Try 5 (`ultra_sub5m` + AutoAugment + Mixup)

This repository contains the **Try 5 upgraded 100% from-scratch PyTorch implementation** of our modified Residual Network (`CustomResNet`) for the Fall 2026 Deep Learning Project 1 Kaggle competition under the strict **5,000,000 trainable parameter ceiling**.

## 1. What Is Upgraded in Try 5 (`ultra_sub5m`)

1. **Maximized Sub-5M Architecture (`ultra_sub5m` — `4,977,834` Parameters, 99.56% of Budget)**:
   - Reallocates channel widths to `C = [64, 144, 208, 280]` with `B = [2, 2, 2, 2]` blocks per stage, widening both intermediate (`144`) and late-stage (`280`) feature representations while remaining strictly under `5,000,000` parameters.
2. **Official CIFAR-10 AutoAugment (`T.AutoAugmentPolicy.CIFAR10`)**:
   - Combines learned CIFAR-10 shear, rotation, solarization, and color-contrast sub-policies with reflection-padded `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, and `RandomErasing` (Cutout).
3. **Mixup Regularization (`alpha = 0.3` on 50% of Batches)**:
   - Blends image pairs $\tilde{x} = \lambda x_i + (1 - \lambda) x_j$ and target labels during training to prevent the late-epoch overfitting observed in 300-epoch unregularized runs.
4. **100% Training Data Utilization (`val_split = 0.0`, All 50,000 Images)**:
   - Trains on all 50,000 labeled competition images (`train_labels.csv` + `train_images/`) for **220 epochs** with an 8-epoch linear warmup and cosine annealing learning rate schedule.

## 2. Repository Structure

```text
cifar10-resnet-ultra-sub5m/
├── README.md                       # Try 5 overview, architecture table, and reproduction instructions
├── requirements.txt                # Python package dependencies (torch, torchvision, Pillow, pandas, etc.)
├── .gitignore                      # Git ignore rules for image folders, datasets, and checkpoints
├── sample_submission.csv           # Official competition submission template (id,label)
├── test.csv                        # 2,000 test image filenames: id,filename (test_0000.png .. test_1999.png)
├── configs/
│   └── model_configs.py            # ResNetSpec registry including `ultra_sub5m` (4,977,834 params)
├── src/
│   ├── __init__.py
│   ├── model.py                    # From-scratch ResidualUnit & CustomResNet + parameter audit
│   ├── dataset.py                  # CIFARCSVDataset + CIFAR-10 AutoAugment + 100% data loader
│   ├── trainer.py                  # Nesterov SGD + Mixup regularization + Warmup/Cosine LR + TTA
│   └── visualize.py                # Automated report plotting (loss/acc curves, per-class bar chart)
├── train.py                        # CLI entrypoint defaulting to `--arch ultra_sub5m --epochs 220`
├── predict_kaggle.py               # Standalone CLI entrypoint to generate `submission.csv` (`id,label`)
└── notebooks/
    └── project1_walkthrough.ipynb  # Self-contained Try 5 Notebook for Kaggle Notebooks & Colab
```

## 3. Architectural Comparison Table (`<= 5,000,000` Trainable Parameters)

| Architecture ID | Stages ($N$) | Channels ($C_i$) | Blocks ($B_i$) | Filters ($F_i$) | Skip ($K_i$) | Pool ($P$) | Trainable Parameters | % of 5M Budget |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_4stage` | 4 | `[64, 128, 192, 256]` | `[2, 2, 2, 2]` | `[3, 3, 3, 3]` | `[1, 1, 1, 1]` | 4 | **4,184,522** | 83.69% |
| `wide_3stage` | 3 | `[64, 144, 264]` | `[3, 3, 3]` | `[3, 3, 3]` | `[1, 1, 1]` | 8 | **4,773,930** | 95.48% |
| `skip3x3_4stage` | 4 | `[64, 128, 192, 256]` | `[2, 2, 2, 2]` | `[3, 3, 3, 3]` | `[1, 3, 3, 3]` | 4 | **4,839,882** | 96.80% |
| `champion_sub5m` | 4 | `[64, 128, 208, 264]` | `[3, 3, 2, 2]` | `[3, 3, 3, 3]` | `[1, 1, 1, 1]` | 4 | **4,917,450** | 98.35% |
| **`ultra_sub5m` (Try 5)** | **4** | **`[64, 144, 208, 280]`** | **`[2, 2, 2, 2]`** | **`[3, 3, 3, 3]`** | **`[1, 1, 1, 1]`** | **4** | **4,977,834** | **99.56%** |

## 4. Quickstart & Reproduction Instructions

### Train Try 5 (`ultra_sub5m`, 220 Epochs, AutoAugment + Mixup, 100% Data)
```bash
python3 train.py \
    --arch ultra_sub5m \
    --train-img-dir ./train_images \
    --train-csv ./train_labels.csv \
    --test-img-dir ./test_images \
    --test-csv ./test.csv \
    --epochs 220 \
    --batch-size 128
```

### Generate Kaggle Competition `submission.csv` (`id,label`) Standalone
```bash
python3 predict_kaggle.py \
    --checkpoint ./outputs/ultra_sub5m_best.pth \
    --test-dir ./test_images \
    --test-csv ./test.csv \
    --sample-submission ./sample_submission.csv \
    --output submission.csv
```
