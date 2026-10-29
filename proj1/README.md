# Project 1: CIFAR-10 ResNet

[Kaggle competition](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl)

Wide ResNets trained from scratch on CIFAR-10, with a 5-million-parameter limit.
Training uses 45,000 images and a fixed, stratified 5,000-image validation split.
The selected configuration is then retrained on all 50,000 training images.

## Setup

Use Python 3.11 and an NVIDIA GPU with CUDA 12.4 support. Run these commands from
the repository root:

```bash
python -m pip install -r proj1/requirements.txt
python -c "from torchvision.datasets import CIFAR10; CIFAR10('proj1/data', train=True, download=True)"
python proj1/check.py
```

Data goes in `proj1/data/`; checkpoints and logs go in `proj1/runs/`.
Both directories are ignored by Git.

## Training

Train the selected configuration on one GPU:

```bash
python -u proj1/train.py --name wrn16_randaugment_mix --device cuda:0 \
  --depth 16 --width 84 --epochs 300 --randaugment 6 --cutout 0 --mixup 0.2 --cutmix 1
```

Training uses SGD with Nesterov momentum, a warmup/cosine learning-rate schedule,
BF16, and an exponential moving average of the weights. Each run saves its
configuration, source files, split indices, metrics, and checkpoints.
`best.pt` contains the raw or EMA weights with the highest validation accuracy;
`last.pt` includes the optimizer and RNG states for resuming.
Append `--resume` to the same command to continue an interrupted run.

For a two-GPU comparison, `python -u proj1/run.py` runs WRN-28 with Cutout and
WRN-16 with RandAugment + Mixup/CutMix on `cuda:0` and `cuda:1`. It selects the
best plain or horizontal-flip validation score, saves `runs/selection.json`,
then retrains the winner on all 50,000 images. Run one copy of this launcher
at a time.

To train the selected configuration on all images directly, use a new run name:

```bash
python -u proj1/train.py --name wrn16_randaugment_mix_full --device cuda:0 \
  --depth 16 --width 84 --epochs 300 --randaugment 6 --cutout 0 --mixup 0.2 --cutmix 1 --full-data
```

## Validation results

| Configuration | Parameters | Epochs | Plain | Horizontal flip |
|---|---:|---:|---:|---:|
| WRN-28, width 58, Cutout | 4,805,542 | 300 | 96.76% | 96.98% |
| WRN-16, width 84, RandAugment + Mixup/CutMix | 4,728,330 | 300 | 97.10% | **97.48%** |
| WRN-28, width 58, RandAugment + Mixup/CutMix | 4,805,542 | 300 | 97.26% | 97.24% |
| WRN-16, width 84, RandAugment + Mixup/CutMix | 4,728,330 | 600 | 97.26% | 97.36% |

All scores use the same 5,000-image holdout. The selected WRN-16 uses raw weights
and horizontal-flip inference. Its full-data run completed 300 epochs;
97.48% is the holdout model's validation score.

The original Cutout run's nominal length of 16 produced a maximum 15-pixel side.
The current implementation uses a 16-pixel side. Cutout is disabled in the
selected configuration.

## Inference and CSV generation

`refine.py` compares plain, horizontal-flip, and ten-view inference, with and
without BatchNorm recalibration on the training split. Use a holdout checkpoint:

```bash
python proj1/refine.py proj1/runs/wrn16_randaugment_mix/best.pt \
  --out proj1/runs/refine_mix_best --device cuda:0
```

Export the full-data model with the selected weight type and inference settings:

```bash
python proj1/export.py proj1/runs/wrn16_randaugment_mix_full/last.pt \
  --recipe proj1/runs/refine_mix_best --out proj1/runs/exported/wrn16_full.pt
```

Store Kaggle credentials at `~/.kaggle/kaggle.json`, then download the competition
images and manifests:

```bash
python proj1/download_competition.py
```

The downloader verifies images and retries rate-limited requests.
It expects `test.csv`, `sample_submission.csv`, and `test_images/` under
`proj1/data/competition/`.

```bash
python proj1/predict.py proj1/runs/exported/wrn16_full.pt \
  --out proj1/runs/submissions/full_wrn16/submission.csv
```

The output has columns `id,label`, labels 0–9, and the sample submission's row
order. The script also saves class probabilities and checkpoint/CSV hashes.
This command writes local files; Kaggle submission is a separate manual step.

## References

- [Wide Residual Networks](https://arxiv.org/abs/1605.07146),
  [author code](https://github.com/szagoruyko/wide-residual-networks)
- [RandAugment](https://arxiv.org/abs/1909.13719), via torchvision
- [CutMix](https://arxiv.org/abs/1905.04899),
  [author code](https://github.com/clovaai/CutMix-PyTorch)
