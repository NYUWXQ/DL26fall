# Local validation results

All classifiers below use the same stratified 45,000/5,000 split of the permitted
CIFAR-10 training data. Plain and horizontal-flip accuracy are measured on the
same 5,000 validation images. Model and inference selection reuse this validation
set; these are development results.

The selected SE-ResNet-164 has 2,488,378 parameters and completed 1,800 full-data
epochs on all 50,000 permitted images. Its final raw weights use FP32 inference
with horizontal flips. The held-out source model reaches 98.10% with this recipe.

- Model: `runs/se_ricap_20261031/winner_full_fp32.pt`
- CSV: `runs/se_ricap_20261031/submission_fp32/submission.csv`
- Selection and artifact paths: `runs/se_ricap_20261031/delivery.json`

The export audit reproduced 98.10% on the held-out source model, verified all
5,000 PNG tensors against the original validation images, and matched the
exported weights exactly to the completed full-data raw weights. The 2,000-row CSV
matches the sample order, schema, saved probabilities, and recorded hashes.
Audits: `runs/se_ricap_20261031/inference_audit_fp32/result.json` and
`runs/se_ricap_20261031/completion_audit.json`.

The table retains training-time BF16 scores except for the explicitly marked
FP32 row. The final SE export uses FP32; its 5,000 validation predictions and
2,000-row final CSV match exactly across both local GPUs.

| Run | Parameters | Epochs | Plain | Flip |
| --- | ---: | ---: | ---: | ---: |
| `structure_search_20261029_trial11_final` | 4,681,354 | 300 | 97.26% | 97.54% |
| `wrn16_randaugment_mix` | 4,728,330 | 300 | 97.10% | 97.48% |
| `distill_20261030_frontloaded` | 4,681,354 | 300 | 97.54% | 97.62% |
| `distill_20261030_frontloaded_t2_e600` | 4,681,354 | 600 | 97.92% | 97.78% |
| `se_ricap_20261031_multiteacher` | 4,681,354 | 600 | 97.98% | 97.90% |
| `se_ricap_20261031_se164_ricap` (BF16) | 2,488,378 | 1,800 | 98.02% | 98.08% |
| Same SE checkpoint, FP32 inference | 2,488,378 | 1,800 | **98.02%** | **98.10%** |
| `distill_20261030_wrn16` | 4,728,330 | 300 | 97.44% | 97.42% |
| `shake_aa_20261030_w32` | 2,923,162 | 1,800 | 97.12% | 97.18% |
| `shake_aa_20261030_w40` | 4,561,354 | 1,800 | 97.54% | 97.48% |
| `lejepa_20261030_finetune` | 4,681,354 | 300 SSL + 300 supervised | 96.96% | 97.32% |

The two-teacher variant averages temperature-scaled probabilities from the
93,338,074-parameter WRN and the 4,561,354-parameter Shake-Shake-40. Both teachers
were trained from scratch on the same 45,000 images and exclude the 5,000-image
validation set. The student retains 600 epochs, distillation weight 0.75,
temperature 2, and the frontloaded WRN configuration. Its best checkpoint is
raw weights at epoch 537: 4,899/5,000 correct with plain inference, three more
than the previous 600-epoch student. Training completed all 211,200 steps in
9,885 seconds. Full-data retraining completed 234,600 steps in 10,624 seconds;
validation evaluation was disabled throughout.
Audits: `runs/se_ricap_20261031/teacher_provenance_audit.json`
and `runs/se_ricap_20261031/multiteacher_completion_audit.json`.

SE-ResNet-164/RICAP completed all 1,800 epochs and 633,600 optimizer steps in
34,066 seconds. It uses 2,488,378 parameters, AutoAugment, RICAP beta 0.3,
RandomErasing probability 0.5, SGD/Nesterov, and a 60-epoch warmup followed by
cosine decay to a minimum learning rate of 0.0001. Its best checkpoint is raw
weights at epoch 1,735: 4,901/5,000 correct with plain inference and 4,904/5,000
with horizontal-flip inference. This improves the previous 97.92% recipe by
eight validation images and the two-teacher variant by five. The batch selected
this model. Its full-data retraining completed all 1,800 epochs and 703,800
steps in 34,531 seconds, with validation evaluation disabled. Audit:
`runs/se_ricap_20261031/se_completion_audit.json`.

BF16 CSV replay changed one validation prediction, scoring 98.06% versus the
98.08% recorded during selection. FP32 replay scores 98.10% (4,905/5,000) on
both GPUs. Under the same FP32 precision, the previous 600-epoch student scores
97.84%; paired predictions show 56 corrected errors, 43 introduced errors, and
52 shared errors. This comparison changes the architecture, augmentation and
training length together. Records: `runs/se_ricap_20261031/precision_audit.json`
and `runs/se_ricap_20261031/paired_validation_fp32.npz`.

The 300-epoch distilled frontloaded classifier improves validation accuracy by 0.08
percentage points (four images). Its teacher is the from-scratch WRN-28 with
93,338,074 parameters: 98.18% plain and 98.28% flip accuracy. The other teacher,
with 36,479,194 parameters, reaches 97.90% plain and 98.04% flip. Both teachers
use only the 45,000 training images; they are excluded from the final model.

Paired predictions show 47 corrected errors and 43 newly introduced errors,
with 76 images misclassified by both models. The two models disagree on 97
images. The largest class-level reduction is automobile errors (10 to 5);
cat and dog errors total 50 after distillation versus 53 before. Paired records:
`runs/distill_inference_audit_20261030/paired_comparison.json`.

The 300-epoch student has completed full-data training on all 50,000 permitted
images. Its local artifacts are:

- Checkpoint: `runs/distill_20261030/winner_full.pt`
- CSV: `runs/distill_20261030/submission/submission.csv`
- Training and selection records: `runs/distill_20261030/final_selection.json`

The export audit passed: all 5,000 validation images survived PNG serialization
with identical normalized tensors, the CSV prediction path reproduced 97.62%,
and the full-data export matched the selected raw training weights exactly.
The local 2,000-row CSV matches its saved probabilities. Audit record:
`runs/distill_inference_audit_20261030/result.json`.

LeJEPA has completed 300 self-supervised epochs on the same 45,000 images
(7,500 seconds), followed by 300 supervised epochs with a fresh classification
head (1,941 seconds). It reaches 97.32% with flip inference. The backbone hash
matches the fine-tuning initialization, and both stages use the same fixed split.
The final classifier has 4,681,354 parameters; the pretraining projector is
discarded. Audit:
`runs/lejepa_20261030/pretraining_audit.json`.

The stronger distillation recipe has completed on the same student and split:
weight 0.75, temperature 2, and 600 epochs, compared with weight 0.5,
temperature 4, and 300 epochs above. This tests the combined recipe. It reaches
97.92% with plain inference: 4,896/5,000 correct, 15 more than the previous
student. Paired predictions show 45 corrected errors, 30 newly introduced errors,
and 74 errors shared by both models. Training took 9,069 seconds.

The actual CSV prediction path reproduced 97.92% after matching validation's
cuDNN benchmarking and evaluation batch size. Its previous settings changed
one BF16 prediction. Validation replay and numerical comparisons are saved in
`runs/kd600_validation_audit_20261030/`.

The batch selected `distill_20261030_frontloaded_t2_e600`. Its 600-epoch
full-data training took 9,694 seconds and completed 234,600 optimizer steps.
Full-data training disables validation evaluation; the saved split file retains
the original validation indices as a reference. Model selection uses the
held-out runs above.

The completed-routes audit verifies all eight held-out student runs and the
two teachers plus LeJEPA pretraining: identical split arrays, complete epoch
records, student parameter limits, and matching teacher/initialization hashes.
Record: `runs/final_batch_20261030/completed_routes_audit.json`.
