# Final report source (AAAI 2027 camera-ready style)

`main.tex` + `project1.bib` + `Figures/` are the source of the submitted report, mirrored
from the Overleaf project so the codebase and the paper cannot drift apart.

**Author block.** The document is built in camera-ready mode (`\usepackage{aaai2027}`, no
`[submission]` option), which is what prints the author names and affiliation; the anonymous
review mode would replace them with an anonymised note. `\nocopyright` suppresses the default
"Copyright (c) 2027, AAAI" line, because this is a course report and not an AAAI publication.

**Build.** The AAAI 2027 style files (`aaai2027.sty`, `aaai2027.bst`) are not redistributed
here, because they ship with the AAAI author kit under its own terms. Drop them next to
`main.tex`, then:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The submitted PDF was produced by the Overleaf project with exactly this source. It is 6 pages;
the assignment's cap is 8 pages including figures, tables and citations.

**Where the numbers come from.** Every leaderboard value in Table 2 traces to a line under
`submissions/log_*.txt` (the Kaggle CLI output written by `scripts/submit_candidates.sh`).
Parameter counts come from `python src/models.py`, which builds each model and sums its
trainable parameters. The two figures are produced by executing `notebooks/results.ipynb`.
`docs/REPORT_FACTS.md` records each claim with its provenance, including the corrections we
made to earlier drafts, so a reader can check the report against the raw logs.

**Note on the ensemble rows.** Table 2 lists ensembles whose *summed* trainable parameters
exceed the five-million cap. They are labelled ablations, not deliverables. The one exception is
the two-seed SE average whose summed parameters are 4,976,756; it is labelled as a boundary
reading, with both members named, and is never presented as a single model.
