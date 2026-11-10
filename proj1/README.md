# Project 1: CIFAR-10 ResNet

[Kaggle competition](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl)

Residual CNNs trained from scratch on CIFAR-10, with a 5-million-parameter limit.

## Model selected for the report

The selected **PyramidNet-110 (alpha 96)** has **4,804,512 parameters** and scored
**0.913 public accuracy** in submission **58019390**. It trained on 40,000 images
for 1,800 epochs. Its final raw weights reach **98.17% on the held-out 10,000-image
fold 1**, using FP32 horizontal-flip inference. The separate 50k retrain scored
0.912; it is not the selected checkpoint.

The executed [report_results.ipynb](report_results.ipynb) recomputes the parameter
count and saved validation accuracy and prints the Kaggle public score from its
receipt. Its evaluation source and inference settings are explicit in the output.

The report is [REPORT.pdf](REPORT.pdf). Its results stop at submission 58045145.
[Selected-model metadata](Figures/pyramid_result.json) identifies the exact
checkpoint, evaluation settings, and Kaggle receipt.

After the setup commands below, reproduce the selected training recipe from the
repository root:

```bash
python proj1/train.py --name report_pyramid_reproduction --architecture pyramid \
  --depth 110 --pyramid-alpha 96 --cv-fold 1 --cv-weights raw \
  --epochs 1800 --batch-size 128 --eval-batch-size 500 \
  --lr 0.1 --warmup-epochs 60 --min-lr 0.0001 --weight-decay 0.0001 \
  --autoaugment --ricap 0.3 --random-erasing 0.5 --cutout 0 --compile-model

python proj1/predict.py proj1/runs/report_pyramid_reproduction/final.pt \
  --data proj1/data/competition --tta flip --precision fp32 \
  --out proj1/runs/report_pyramid_reproduction/predictions/submission.csv
```

The saved submitted weights are `runs/opus_direct_20261104_pyramid_f1/final.pt`.
Their SHA256 is
`12a84327e589c1f705c7aa0fc9413c0293f841571b355014bccb7c850fda2513`.
Training data and weights are local artifacts and are not bundled in Git.
For the executed report notebook and plots, install `matplotlib` and `jupyter`
with `python -m pip install matplotlib jupyter`. Regenerate the selected-model
figure with `python proj1/plot_report.py`.

## Earlier experiments and setup

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

For a basic WRN training run on one GPU:

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

To train that WRN configuration on all images directly, use a new run name:

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

All scores use the same 5,000-image holdout. In this initial WRN comparison,
WRN-16 was selected with raw weights and horizontal-flip inference. Its
full-data run completed 300 epochs;
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

## Architecture search and error analysis

The model accepts `--stage-widths W1 W2 W3`, `--stage-blocks N1 N2 N3`, and
`--groups G`. Groups apply to the second spatial convolution in each residual
block; the first convolution mixes channels. The 5M parameter limit also applies
to these configurations.

Install the optional search dependency and run the search checks:

```bash
python -m pip install -r proj1/requirements-search.txt
python proj1/check_search.py
python proj1/search.py --name structure_search --devices 0 1 --trials 12 --epochs 80 --finalists 2 --final-epochs 300
```

The first seven trials compare asymmetric and grouped architectures with the
same training settings. Optuna then searches architecture, learning rate,
weight decay, dropout, RandAugment strength, Mixup, and CutMix. A median pruner
can stop weaker trials after epoch 40. The two highest-scoring completed trials
are retrained from scratch for 300 epochs. The SQLite study, commands, logs,
and result summary are saved under `runs/<search-name>/`.

To inspect the original validation errors:

```bash
python proj1/analyze_validation.py proj1/runs/wrn16_randaugment_mix/best.pt --out proj1/runs/validation_errors
```

This uses the fixed 5,000-image split from the CIFAR-10 training set and writes
per-class accuracy, a confusion matrix, error records, and an image grid.

## Shake-Shake with AutoAugment

The CIFAR implementation is ported from [Xavier Gastaldi's author code](https://github.com/xgastaldi/shake-shake).
It uses independent per-image forward and backward branch weights, with equal
branch weights at evaluation. The two 26-layer configurations contain 2,923,162
parameters (width 32) and 4,561,354 parameters (width 40).

```bash
python proj1/check_shake.py
python proj1/train.py --name shake26_32_aa --architecture shake --depth 26 --width 32 \
  --epochs 1800 --lr 0.2 --weight-decay 0.0001 --autoaugment --cutout 16
```

AutoAugment uses torchvision's CIFAR-10 policy. Training retains the existing
45,000/5,000 split, SGD/Nesterov, cosine schedule with warmup, BF16 and raw/EMA
validation selection. These training settings are recorded in each run.

`shake_experiment.py --name shake_aa_round1 --after distill_20261030` queues both
widths on the two GPUs after the specified experiment's final selection completes.
It compares them against the existing models and distilled students, then prepares
a full-data model and local CSV if validation accuracy improves.

## Teacher distillation

### SE-ResNet and multiple teachers

`python proj1/check_se.py` checks the SE bottleneck network, RICAP labels, and
probability averaging for multiple teachers. The CIFAR SE-ResNet-164 has
2,488,378 parameters, three stages of 18 bottlenecks, SE reduction 4, and pooled
projection shortcuts. Its structure follows the [Spring 2025 project code](https://github.com/rigvedrs/Optimised-ResNet-for-CIFAR-10).

Reproduce the selected model with fresh run names:

```bash
se_options=(--architecture se --depth 164 --width 16 --epochs 1800 \
  --autoaugment --ricap 0.3 --random-erasing 0.5 --cutout 0 \
  --lr 0.1 --weight-decay 0.0001 --warmup-epochs 60 --min-lr 0.0001)
python proj1/train.py --name se164_ricap --device cuda:0 "${se_options[@]}"
python proj1/train.py --name se164_ricap_full --device cuda:0 --full-data "${se_options[@]}"
python proj1/refine.py proj1/runs/se164_ricap/best.pt \
  --out proj1/runs/se164_ricap_recipe --device cuda:0 --precision fp32
python proj1/export.py proj1/runs/se164_ricap_full/last.pt \
  --recipe proj1/runs/se164_ricap_recipe --out proj1/runs/se164_ricap_export.pt --device cuda:0
python proj1/predict.py proj1/runs/se164_ricap_export.pt \
  --out proj1/runs/se164_ricap_prediction/submission.csv --device cuda:0
```

The export records inference precision. Prediction uses that setting by default;
older checkpoints retain BF16 inference. To compare both routes using the
existing WRN-28 and Shake-Shake-40 teacher checkpoints:

```bash
python proj1/se_experiment.py --name se_ricap_20261031
```

This runs two local experiments on GPUs 0 and 1: 1,800 epochs of SE-ResNet-164
with AutoAugment, RICAP beta 0.3 and RandomErasing; and 600 epochs of the
frontloaded WRN distilled from the existing WRN-28 teacher and Shake-Shake-40.
The latter averages the teachers' probabilities at temperature 2. Both retain
the fixed 45,000/5,000 training split. A validation improvement over the existing
97.92% model triggers full-data retraining and local CSV preparation.
The run does not submit or upload anything.

`train.py --teacher-only` permits a larger network for teacher training.
Train the teacher on the same 45,000-image split used by the student:

```bash
python proj1/train.py --name teacher_wrn28 --teacher-only --depth 28 --width 160 \
  --epochs 300 --eval-batch-size 128 --randaugment 6 --cutout 0 --mixup 0.2 --cutmix 1
python proj1/train.py --name student_kd --depth 16 --width 84 --epochs 300 \
  --randaugment 6 --cutout 0 --mixup 0.2 --cutmix 1 \
  --teacher-checkpoint proj1/runs/teacher_wrn28/best.pt --distill-weight 0.5 --temperature 4
```

The student combines cross-entropy with temperature-scaled KL divergence on the
same augmented inputs. Teacher weights are frozen and their checkpoint hash is
recorded. Validation training rejects teachers that used the validation images.
Full-data student training can reuse the selected 45,000-image teacher.
Export and competition prediction reject teacher-only checkpoints and models
above 5 million parameters.

`python proj1/distill.py --name distill_round1` runs 36.5M- and 93.3M-parameter
teachers on two GPUs, selects the better teacher on validation, and trains both
the frontloaded and WRN-16 students. If a student improves the current 97.54%
validation reference, it is retrained on all 50,000 images and exported to a
local CSV. Commands, teacher selection, student results, and final selection are
saved under `runs/<name>/`.

The current stronger distillation recipe reuses the completed 93.3M teacher:

```bash
python proj1/train.py --name distill_20261030_frontloaded_t2_e600 --device cuda:1 \
  --epochs 600 --stage-widths 136 192 232 --stage-blocks 3 3 2 \
  --lr 0.0816831165758183 --weight-decay 0.00023238214295871968 \
  --randaugment 9 --cutout 0 --mixup 0 --cutmix 1 \
  --teacher-checkpoint proj1/runs/distill_20261030_teacher256/best.pt \
  --distill-weight 0.75 --temperature 2
```

## LeJEPA experiment

This is a CIFAR adaptation of the [LeJEPA minimal example](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)
by Randall Balestriero and Yann LeCun. The existing frontloaded residual backbone
uses four 32x32 views, an auxiliary projection MLP, view consistency, and SIGReg.
The projector is used during self-supervised training and discarded afterward.
The final classifier retains 4,681,354 parameters.

The prepared configuration is `configs/lejepa_frontloaded.json`: 300 self-supervised
epochs, batch size 128, AdamW, and SIGReg weight 0.02. The 45,000-image training
split is used throughout pretraining; validation images are excluded. Pretraining
selects its final epoch. Supervised fine-tuning then uses the existing validation
procedure and a fresh classification head.

CPU preparation checks:

```bash
python proj1/check_lejepa.py
python proj1/pretrain_lejepa.py --name lejepa_frontloaded --check-only
```

Run pretraining, fine-tuning, comparison with the completed candidates, and full-data
training plus local CSV generation if the candidate improves validation accuracy:

```bash
python proj1/lejepa_experiment.py --name lejepa_20261030 --device 0
```

To run the two training stages individually:

```bash
python proj1/pretrain_lejepa.py --name lejepa_frontloaded --device cuda:0
python proj1/train.py --name lejepa_frontloaded_finetune --device cuda:0 \
  --stage-widths 136 192 232 --stage-blocks 3 3 2 --epochs 300 \
  --lr 0.0816831165758183 --weight-decay 0.00023238214295871968 \
  --randaugment 9 --cutout 0 --mixup 0 --cutmix 1 \
  --init-checkpoint proj1/runs/lejepa_frontloaded/backbone.pt
```

Compare fine-tuned accuracy with the existing same-architecture supervised run
`structure_search_20261029_trial11_final`. The prepared schedule adds 300 self-supervised
epochs before 300 supervised epochs; record both stages' compute time. Both stages save
resumable `last.pt` files. A full-data fine-tune may reuse the selected training-only
backbone; export checks that its initialization hash matches the validation run.
The experiment driver records its current stage under `runs/<name>/status.json`.

For the current batch, `complete_batch.py --name final_batch_20261030` waits for
`lejepa_20261030` and the stronger 600-epoch distillation run, compares both with
the completed candidates, and prepares the winner on GPU 1. It reuses an existing
full-data export when that model wins. Final checkpoint and CSV paths are saved
under `runs/final_batch_20261030/delivery.json`.

## References

- [Wide Residual Networks](https://arxiv.org/abs/1605.07146),
  [author code](https://github.com/szagoruyko/wide-residual-networks)
- [RandAugment](https://arxiv.org/abs/1909.13719), via torchvision
- [CutMix](https://arxiv.org/abs/1905.04899),
  [author code](https://github.com/clovaai/CutMix-PyTorch)
- [PyTorch knowledge distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- [LeJEPA](https://arxiv.org/abs/2511.08544),
  [author code](https://github.com/galilai-group/lejepa), CC BY-NC 4.0
