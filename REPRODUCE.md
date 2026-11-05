# Reproduce this project / 复现指南

Two pipelines live here: `src/` is ours (pre-activation wide ResNet), `proj1/` is the teammates'
SE-ResNet/RICAP pipeline, cited in the report and used for the width control experiment. Everything
below runs on CPU except training.

本仓库有两条管线：`src/` 是我们的（预激活宽残差网络），`proj1/` 是队友的 SE-ResNet/RICAP 管线。
下面除训练外都能在 CPU 上跑。

## 0. Environment

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision numpy pandas scikit-learn matplotlib jupyter
```

## 1. Verify the parameter budget (seconds, CPU)

```bash
.venv/bin/python src/models.py
```

Expected: one line per architecture with `params=...  ok=True/False`. `ok=False` means the built-in
guard refuses it, e.g. `preact16x6 params=6,242,026 ok=False`.

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'proj1')
from model import build_model
for w in (16,22):
    print(w, sum(p.numel() for p in build_model({'architecture':'se','depth':164,'width':w}).parameters()))"
```

Expected: `16 2488378` and `22 4684888`. Width 23 raises `ValueError: Model exceeds 5M parameters`.

## 2. Train (GPU; the numbers in the report come from 600-epoch runs)

```bash
# our wide-ResNet pipeline
.venv/bin/python src/train.py --model preact22x4 --epochs 300 --batch-size 512 --lr 0.25 \
    --aug strong --mix both --out runs/wrn22x4_s0 --workers 8

# the SE pipeline used for the width experiment (recipe identical to the 0.894/0.899 rows)
cd proj1 && python -u train.py --name se_w16_s0 --device cuda:0 --architecture se --depth 164 \
    --width 16 --autoaugment --ricap 0.3 --random-erasing 0.5 --cutout 0 \
    --lr 0.1 --weight-decay 0.0001 --warmup-epochs 60 --min-lr 0.0001 \
    --batch-size 128 --workers 8 --full-data --epochs 600
```

Wall-clock on one A100: ~59 s/epoch at width 16 and ~58 s/epoch at width 22, i.e. ~10 h for 600
epochs. `--resume` continues from `runs/<name>/last.pt`, which is written every epoch.

**Determinism.** `proj1/train.py` pins `torch.manual_seed(42)` and exposes no seed switch, so two
"seeds" of that pipeline are really two runs that differ through non-deterministic kernels and
different nodes. The report measures and reports that run-to-run band (~0.5 leaderboard points) on
purpose; don't describe those pairs as a seed sweep. `src/train.py` does take `--seed`.

## 3. Reproduce the leaderboard numbers

The exact CSVs that produced every leaderboard row are committed under `submissions/`, and the
mapping is the table in the root `README.md` (configuration → parameters → public LB → CSV → log).
To re-submit one:

```bash
kaggle competitions submit -c cifar-10-res-net-architecture-challenge-dl \
    -f submissions/se_w22_s0_single.csv -m "reproduction check"
```

Scores are returns from the competition's leaderboard, which has daily quota; they are not
recomputable offline. Local accuracy on the labelled CIFAR-10 test split is a different measurement
and lives in the notebook.

## 4. Executed evidence (required by the assignment)

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/results.ipynb --inplace
```

It loads a checkpoint, prints **4,326,986** parameters, measures **0.9750** on the 10,000 labelled
CIFAR-10 test images (0.9732 plain) and plots the confusion matrix and training curves. The
checkpoint is not committed (19 MB); see `docs/HPC.md` for how it was produced, or point the first
cell at any compatible `final.pt`.

## 5. Build the report

See `report/latex/README.md`: copy `aaai2027.sty`/`.bst` from the AAAI 2027 author kit next to
`main.tex`, then `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Output: 6 pages.

## 6. Cluster

`slurm/*.sbatch` are the templates; `docs/HPC.md` records the account, partitions and the two
failure modes we hit (a partition that requires two GPUs, and a job that needs `--resume` because the
8-hour session window ends first).
