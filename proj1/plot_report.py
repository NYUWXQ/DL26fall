"""Plot the submitted PyramidNet's saved training and validation measurements."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    figures = Path(__file__).resolve().parent / "Figures"
    history = np.loadtxt(figures / "pyramid_training.csv", delimiter=",", skiprows=1)
    counts = np.loadtxt(figures / "pyramid_confusion.csv", delimiter=",", dtype=int)
    assert np.array_equal(history[:, 0], np.arange(1, 1801))
    assert counts.shape == (10, 10) and counts.sum() == 10000
    assert counts.trace() == 9817 and np.all(counts.sum(axis=1) == 1000)
    classes = ["airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7, 3.25), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[2, 1])
    loss = fig.add_subplot(grid[0, 0])
    rate = fig.add_subplot(grid[1, 0], sharex=loss)
    matrix = fig.add_subplot(grid[:, 1])
    loss.plot(history[:, 0], history[:, 1], color="#256497", linewidth=0.65)
    loss.set(title="(a) Submitted PyramidNet: training", ylabel="RICAP training loss")
    loss.tick_params(labelbottom=False)
    rate.plot(history[:, 0], history[:, 2], color="#a04b29", linewidth=1)
    rate.set(xlabel="Epoch", ylabel="Learning rate", xlim=(1, 1800), xticks=[1, 600, 1200, 1800])
    for ax in (loss, rate):
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    matrix.imshow(counts / counts.sum(axis=1, keepdims=True) * 100, cmap="Blues", vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            if counts[i, j]:
                matrix.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=6.5,
                            color="white" if i == j else "black")
    matrix.set(xticks=range(10), yticks=range(10), xticklabels=classes, yticklabels=classes,
               xlabel="Predicted class", ylabel="True class",
               title="(b) Fold 1 validation: 98.17%")
    plt.setp(matrix.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    matrix.tick_params(length=0, labelsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"pyramid_diagnostics.{suffix}", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
