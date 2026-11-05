> **Superseded in part.** The `LB = (1-p)·a + p/10` label-noise fit below was **withdrawn**: it
> conflicts with the pixel-level audit (the shipped training images are exactly CIFAR-10; the test
> images match neither CIFAR-10 test nor CIFAR-10.1 at any dihedral transform). It also audits a
> 4,000-image sample, whereas the final audit hashed all 50,000 pairs. The current position is in
> `docs/REPORT_FACTS.md`; keep this file only as a record of the analysis path.
> **部分失效**：下面的标签噪声拟合已撤回；当前结论以 `docs/REPORT_FACTS.md` 为准。

# Leaderboard saturation and the IID→OOD gap

Status: **measured** (numbers from `analysis/lb_noise.py`, `analysis/refine.py`,
and the team-lead's controlled experiment). No number here is modeled or
extrapolated unless the source command is named.

## 1. The 6.9-point val→LB drop is not label noise

An earlier draft of this document derived an "irreducible label-noise rate"
p ≈ 7.9% from `LB = (1-p)·a + p/10`, fit to the teammate's single data point
(97.48% holdout → 0.906 LB). **That derivation is withdrawn.** It assumed the
entire 6.9-point gap came from wrong LB labels. The team lead's measurements
contradict the premise:

| measurement | result | source |
|---|---|---|
| competition `train_images/` vs official CIFAR-10 train, 4000 sampled | **4000/4000 pixel-exact** | team lead |
| competition `test_images/` vs official CIFAR-10 test | **0/2000** | team lead |
| competition `test_images/` vs CIFAR-10.1 (Recht et al.) | **0/2000** | team lead |
| competition `test_images/` vs official test under 8 dihedral transforms | 18/2000 (horizontal flip only) | team lead |

So the training data **is** stock CIFAR-10 and the test pool is **another
2,000-image collection of the same 10 classes** — a re-collected set, not a
transform of anything we hold. `analysis/lb_noise.py` still reports the
arithmetic for completeness, but it is explicitly labeled as an unverified
model, not a finding.

## 2. Photometric domain shift is real but costs ≤ 0.2 points

Applying the measured test-set statistics to the 2,500-image labeled
validation split (epoch-40 checkpoint):

| transform | val acc | Δ |
|---|---|---|
| baseline | 0.8548 | — |
| x·1.05 + 8 (brightness +4.4/255, contrast +5%) | 0.8568 | +0.0020 |
| x·1.04 (brightness only) | 0.8540 | −0.0008 |
| gamma 0.92 | 0.8552 | +0.0004 |
| quantize step 4 | 0.8560 | +0.0012 |
| gaussian blur σ≈1 *(negative control)* | 0.2784 | −0.5764 |

The negative control shows the experiment would have detected a real shift.
Gradient energy (test 15.34 > train 14.38) confirms the shipped test images are
not blurred. **H2 (photometric domain shift) is rejected.**

## 3. The measurable quantity: IID → OOD gap

Since the competition test pool is re-collected rather than transformed, the
operationally meaningful number is how far a checkpoint drops when the
evaluation distribution changes. CIFAR-10.1 is used as a **diagnostic proxy
only** — it is not the competition test set (0/2000 overlap), it was never used
for training, and it was never used for model selection.

Measured (`src/refine.py`, epoch-62 checkpoint, raw weights, same
preprocessing, seed-0 pinned split):

| split | n | plain | flip |
|---|---|---|---|
| IID (standard CIFAR-10 test) | 60,000 | 0.8808 | 0.8905 |
| OOD (CIFAR-10.1) | 2,000 | 0.7565 | 0.7775 |
| **gap** | | **−12.43** | **−11.30** |

Two observations that shape the strategy:

1. **TTA pays more on OOD than on IID.** Flip-vs-plain is +0.97 points on IID
   but **+2.10 points on OOD**. This is direct evidence for the "ensembling
   and TTA help more on a re-collected pool" claim, rather than an assumption.
2. **The 12.4-point gap is a ceiling on the re-collected effect, not the
   competition value.** CIFAR-10.1 and the competition test set are different
   collections (0/2000 overlap), so the competition drop is somewhere between
   the teammate's measured 6.9 points and this proxy's 12.4 — **do not
   subtract them.** The only way to know the real number is a submission.

## 4. What this implies for the submission

- The gap is a **data-difficulty** effect (re-collected test set), in the
  3–11 point band reported by Recht et al. for benchmark re-collection. Our
  6.9-point measured drop sits inside that band.
- The levers that move accuracy on a harder OOD pool are **true accuracy of
  the single model**, **ensembling**, and **TTA** — not domain matching, which
  Section 2 shows costs ≤ 0.2 points.
- The current best *supervised, submission-safe* configuration is
  `raw.flip` (no BN recalibration, so no transductive disclosure). On the
  epoch-62 checkpoint that is 0.8688 on the pinned val split.
- A real LB submission from the final checkpoint will measure the val→LB
  mapping directly and replace every number in this section with a data point.

## 5. Reproducibility

```bash
python analysis/lb_noise.py            # Section 1 arithmetic + sensitivity
python analysis/domain_probe.py        # train/test distribution probe (numpy+scipy)
```

CIFAR-10.1 lives in `.cache/cifar101/` and is referenced **only** by
diagnostic scripts. No training or selection path reads it.
