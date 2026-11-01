"""Evaluate a set of checkpoints on IID / OOD references and on the competition test set.

Purpose: once the cluster runs land locally we need one table that answers
"which of these should we submit, and does ensembling help?" without burning
Kaggle submissions on every combination.

评估一组 checkpoint 在 IID / OOD 参照集与竞赛测试集上的表现。
目的：集群产物落到本地后，用一张表回答「该提交哪个、集成是否有用」，
避免每试一种组合就消耗一次 Kaggle 提交额度。

Reference sets / 参照集:
  iid   official CIFAR-10 test (10k, never trained on, valid even for full-data runs)
  ood   CIFAR-10.1 (2k, a re-collected harder set; diagnostic only, never for training)

Usage / 用法:
  .venv/bin/python scripts/eval_checkpoints.py runs/full_s0/final.pt runs/kd_s10/final.pt
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data import CifarNumpy, load_test  # noqa: E402
from models import build_model, n_params  # noqa: E402


@torch.no_grad()
def probs_for(model, x: np.ndarray, mode: str = "flip", bs: int = 500) -> np.ndarray:
    """Average softmax over TTA views. / 对 TTA 各视角取 softmax 平均。"""
    ds = CifarNumpy(x, np.zeros(len(x), dtype=np.int64), train=False)
    out = []
    for i in range(0, len(x), bs):
        xb = torch.stack([ds[j][0] for j in range(i, min(i + bs, len(x)))])
        views = [xb]
        if mode in ("flip", "translate_flip"):
            views.append(xb.flip(-1))
        if mode == "translate_flip":
            pad = F.pad(xb, (2, 2, 2, 2), mode="reflect")
            views += [pad[:, :, dy:dy + 32, dx:dx + 32] for dy, dx in ((0, 4), (4, 0), (4, 4))]
        out.append(torch.stack([model(v).float().softmax(1) for v in views]).mean(0))
    return torch.cat(out).numpy()


def load_ckpt(path: str, device: torch.device):
    """Load the EMA weights (what we submit) plus metadata. / 载入 EMA 权重与元信息。"""
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    model = build_model(args.get("model", "preact16x5.25")).to(device)
    model.load_state_dict(ck.get("ema", ck["model"]))
    model.eval()
    return model, {"path": path, "epochs": ck.get("epoch"), "args_model": args.get("model"),
                   "teacher": args.get("teacher", ""), "params": n_params(model)}


def load_refs(root: str):
    """Load IID (official CIFAR-10 test) and OOD (CIFAR-10.1) references. / 载入参照集。"""
    refs = {}
    try:
        iid_x = np.load(os.path.join(root, ".cache", "iid_test_x.npy"))
        iid_y = np.load(os.path.join(root, ".cache", "iid_test_y.npy"))
        refs["iid"] = (iid_x, iid_y)
    except FileNotFoundError:
        with open(os.path.join(root, ".cache", "kcifar", "test_batch"), "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        refs["iid"] = (d[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8),
                       np.asarray(d[b"labels"], dtype=np.int64))
    ood_p = os.path.join(root, ".cache", "cifar101", "data.npy")
    if os.path.exists(ood_p):
        refs["ood"] = (np.load(ood_p).transpose(0, 3, 1, 2),
                       np.load(os.path.join(root, ".cache", "cifar101", "labels.npy")))
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--tta", default="flip", choices=["plain", "flip", "translate_flip"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--probs-dir", default="analysis/ckpt_probs", help="cache probs here / 概率缓存目录")
    args = ap.parse_args()

    device = torch.device(args.device if torch.backends.mps.is_available() or args.device == "cpu" else "cpu")
    refs = load_refs(args.root)
    test_x, ids = load_test(args.root)
    os.makedirs(os.path.join(args.root, args.probs_dir), exist_ok=True)

    rows = []
    test_probs = {}
    for path in args.ckpts:
        model, meta = load_ckpt(path, device)
        rec = dict(meta)
        for name, (rx, ry) in refs.items():
            p = probs_for(model, rx, args.tta)
            rec[name] = float((p.argmax(1) == ry).mean())
        pt = probs_for(model, test_x, args.tta)
        tag = os.path.basename(os.path.dirname(path)) + "_" + os.path.basename(path).replace(".pt", "")
        np.save(os.path.join(args.root, args.probs_dir, tag + ".npy"), pt)
        test_probs[tag] = pt
        rec["test_mean_conf"] = float(pt.max(1).mean())
        rows.append(rec)
        print(f"[eval] {path:46s} iid={rec.get('iid', float('nan')):.4f} "
              f"ood={rec.get('ood', float('nan')):.4f} conf={rec['test_mean_conf']:.4f}", flush=True)

    print("\n| checkpoint | epochs | teacher | iid(10k) | ood(2k) | test conf |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {os.path.basename(os.path.dirname(r['path']))} | {r['epochs']} | "
              f"{os.path.basename(r['teacher']) or '-'} | {r.get('iid', float('nan')):.4f} | "
              f"{r.get('ood', float('nan')):.4f} | {r['test_mean_conf']:.4f} |")

    # Pairwise agreement on the competition test set: a cheap diversity measure.
    # 竞赛测试集上的两两一致率：廉价的多样性度量。
    tags = list(test_probs)
    if len(tags) > 1:
        print("\n[pairwise agreement on competition test]")
        preds = {t: test_probs[t].argmax(1) for t in tags}
        print("| | " + " | ".join(tags) + " |")
        print("|---|" + "---|" * len(tags))
        for a in tags:
            cells = ["—" if a == b else f"{(preds[a] == preds[b]).mean():.3f}" for b in tags]
            print(f"| {a} | " + " | ".join(cells) + " |")

    # Ensemble of everything, and of the top-2 by OOD, as candidate submissions.
    # 全量集成与 OOD 前二的集成，作为候选提交。
    if len(tags) > 1:
        ens = np.mean([test_probs[t] for t in tags], axis=0)
        np.save(os.path.join(args.root, args.probs_dir, "ENSEMBLE_all.npy"), ens)
        print(f"\n[ensemble] all {len(tags)} checkpoints averaged -> {args.probs_dir}/ENSEMBLE_all.npy")


if __name__ == "__main__":
    main()
