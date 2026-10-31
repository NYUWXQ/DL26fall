"""Verify the label-noise hypothesis: LB = (1-p)·a + p/10.

Pure numpy, no torch.  Takes (holdout accuracy, leaderboard score) pairs and
least-squares fits the two-parameter noise model, then reports the implied
irreducible noise rate p and the accuracy required to reach a target LB.

用法: python analysis/lb_noise.py
"""
from __future__ import annotations

import numpy as np

# REAL data points only (no self-referential anchors).
#   teammate proj1/ WRN-16-84: 5k holdout 97.48%, public LB 0.906  (Chief, verified)
#   competition leaderboard ceiling: 0.909 (2000 test images; ~6/2000 off best)
# The ceiling is NOT an (a, LB) pair -- it is an LB target we solve *toward*.
TEAMMATE_A = 0.9748
TEAMMATE_LB = 0.906
CEILING_LB = 0.909


def implied_p(a: float, lb: float) -> float:
    """lb = (1-p)*a + p/10  =>  p = (lb - a) / (0.1 - a)"""
    return (lb - a) / (0.1 - a)


def main() -> None:
    p = implied_p(TEAMMATE_A, TEAMMATE_LB)
    print(f"[noise] single real point: a={TEAMMATE_A:.4f}, LB={TEAMMATE_LB:.4f}")
    print(f"[noise] implied noise rate p = {p:.4f}  ({p*100:.2f}%)")
    print(f"[noise] interpretation: of every 100 test labels, ~{p*100:.1f} are wrong,")
    print(f"[noise]   so a perfect classifier scores (1-p)*1 + p/10 = {(1-p)*1+p/10:.4f} = {TEAMMATE_LB:.3f}")

    print()
    print("[noise] accuracy required for target LBs (solving a = (LB - p)/(1-p)):")
    for target in (0.906, 0.909, 0.9095, 0.9100):
        need = (target - p) / (1 - p)
        print(f"   LB {target:.4f}  ->  need true acc {need:.4f}  "
              f"(+{need-TEAMMATE_A:+.4f} vs teammate {TEAMMATE_A:.2%})")

    print()
    print("[noise] sensitivity -- how p moves if teammate LB measurement is off:")
    for dl in (-0.002, -0.001, 0.0, 0.001, 0.002):
        pp = implied_p(TEAMMATE_A, TEAMMATE_LB + dl)
        print(f"   LB {TEAMMATE_LB+dl:+.3f} -> p={pp:.4f} ({pp*100:.2f}%)")

    print()
    print("[noise] sensitivity -- how p moves if teammate holdout accuracy is off:")
    for da in (-0.005, -0.002, 0.0, 0.002, 0.005):
        pp = implied_p(TEAMMATE_A + da, TEAMMATE_LB)
        print(f"   a {TEAMMATE_A+da:+.3f} -> p={pp:.4f} ({pp*100:.2f}%)")

    print()
    print(f"[noise] ceiling check: LB {CEILING_LB} with p={p:.4f} implies true acc {(CEILING_LB - p)/(1-p):.4f}")


if __name__ == "__main__":
    main()
