# Report handoff — rules for editing `report/latex/main.tex`

The report is at **6 pages** (cap 8). It lives in Overleaf, project
`https://www.overleaf.com/project/6aea2770096db15da572bf6b`; the same source is mirrored here in
`report/latex/`. 报告当前 6 页（上限 8），Overleaf 与本目录是同一份源；**在 Overleaf 里改，然后告诉我，
我把改动同步进仓库**，避免两边分叉。

## 1. Build settings — do not change / 不要改的构建设置

| Setting | Value | Why |
|---|---|---|
| `\usepackage{aaai2027}` | **no `[submission]` option** | `[submission]` is the anonymous-review mode: it blanks `\showauthors` and hides the author block. The assignment requires all names on page 1. |
| `\nocopyright` | keep it | camera-ready mode otherwise prints "Copyright (c) 2027, AAAI", which is false for a course report. |
| `/TemplateVersion (2027.2)` | keep | matches the camera-ready template we build against. |
| `pdflatex` + `bibtex` + `pdflatex` ×2 | | figures and citations both need the extra passes. |

## 2. Numbers are frozen at their sources / 数字只能来自日志

Every leaderboard value in Table 2 traces to a real return under `submissions/log_*.txt`, and every
parameter count traces to `python src/models.py`. If you add a row:

1. submit the candidate, read the score back from the Kaggle CLI output;
2. add the row **and the log filename** to `report/latex/README.md`'s results table in the repo root
   README (same table, same order);
3. only then add it to `main.tex`.

Never type a score that is not in a log. **不要写日志里没有的分数。**

## 3. The compliance section is a claim, not boilerplate / 合规段是论证

`Section 5` argues that the literal reading of "the total number of trainable parameters" makes a
twelve-member ensemble (~52–57M) non-compliant, so our deliverable is a **single network**, and the
two-run SE average (4,976,756 summed) is a **labelled boundary case**, not a single model. If you add
an ensemble result, keep it above/below the rule correctly and keep the label. Don't reintroduce a
headline that is an ensemble.

## 4. Claims we withdrew — do not bring them back / 已撤回的结论不要再写

- "~7% irreducible label noise" from the `LB = (1-p)·a + p/10` fit — withdrawn, it conflicts with the
  pixel-level audit. See `docs/REPORT_FACTS.md` §7.
- "domain shift explains the validation→leaderboard gap" — a controlled experiment moved accuracy by
  only +0.2 points.
- any linear extrapolation from local accuracy to leaderboard score.

## 5. What we now know, in one line each / 结论口径

1. Test images are an independently collected, harder pool (hash audit: 0/2000 match at any dihedral
   transform).
2. Changing the architecture family moved the score; spending the remaining parameters (w16 → w22) and
   tripling training (600 → 1,800 ep) are both inside the 0.5-point run-to-run band.
3. Flip TTA helps, ten-view TTA hurts, and ensembling saturates: twelve members (≈55M) buy 0.903 while
   two compliant 4.98M members buy 0.902.
4. Local accuracy and CIFAR-10.1 accuracy do not predict the leaderboard ordering.

## 6. Checklist before uploading to Brightspace / 交稿前检查

- [ ] PDF is ≤ 8 pages including figures, tables and citations (currently 6).
- [ ] Author names and the public repository URL appear on page 1.
- [ ] `https://github.com/NYUWXQ/DL26fall` loads in a private browser window (i.e. the repository is
      actually public).
- [ ] No new Overfull `\hbox` warning bigger than ~20pt; `Underfull \vbox` is cosmetic and expected.
- [ ] Every number you added appears in `submissions/log_*.txt` and in the repository README table.
- [ ] The PDF you upload is the one built from this source (not an older local draft).
