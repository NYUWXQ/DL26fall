"""Cluster-side sweep: per-checkpoint IID accuracy, greedy weight soup, test-set probabilities.

Runs where the checkpoints live (no 38 MB transfers). For every checkpoint it
reports accuracy on the official CIFAR-10 test set (10k, never trained on, so it
is a clean selection metric even for full-data runs), then builds a greedy
weight-average soup over models of the same architecture, and exports the
competition-test probabilities for each model and each soup as a small .npz.

在 checkpoint 所在处运行（无需传 38 MB 权重）。对每个 checkpoint 报告官方 CIFAR-10
测试集（1 万张，从未参与训练，因此即使 full-data 训练也是干净的选型指标）上的准确率，
然后对同结构模型做贪心权重平均（model soup），并把每个模型与 soup 在竞赛测试集上的
概率导出为一个小 .npz。

Usage (cluster) / 集群用法:
  python scripts/burst_eval_all.py --ckpts "runs/full_s*/final.pt" "runs/kd_s*/final.pt" \
      --test-dir test_images --out analysis/eval_all.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from models import build_model  # noqa: E402
from burst_predict import flip_probs, load_test_tensors  # noqa: E402


def load_iid(root: str):
    """Official CIFAR-10 test split from the local cache. / 本地缓存里的官方 CIFAR-10 测试集。"""
    x = np.load(os.path.join(root, ".cache", "iid_test_x.npy"))
    y = np.load(os.path.join(root, ".cache", "iid_test_y.npy"))
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    t = torch.from_numpy(x.astype(np.float32) / 255.0)
    return (t - mean) / std, y


def load_state(path: str, device):
    """Return (state_dict, model_name) for one checkpoint. / 返回权重与结构名。"""
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    return ck.get("ema", ck["model"]), args.get("model", "preact16x5.25")


def tag_of(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--test-dir", default="test_images")
    ap.add_argument("--out", default="analysis/eval_all.npz")
    ap.add_argument("--soup-groups", nargs="*", default=["full_s", "kd_s"],
                    help="directory-name prefixes to soup separately / 分别做 soup 的目录名前缀")
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.ckpts:
        paths.extend(sorted(glob.glob(pattern)))
    paths = [p for p in paths if os.path.isfile(p)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sweep] {len(paths)} checkpoints on {device}", flush=True)

    iid_x, iid_y = load_iid(args.root)
    test_x = load_test_tensors(os.path.join(args.root, args.test_dir), n=2000)

    payload, individual = {}, {}
    for path in paths:
        tag = tag_of(path)
        sd, name = load_state(path, device)
        model = build_model(name).to(device)
        model.load_state_dict(sd)
        model.eval()
        iid_acc = float((flip_probs(model, iid_x, device).argmax(1) == iid_y).mean())
        payload[tag] = flip_probs(model, test_x, device).astype(np.float32)
        individual[tag] = {"path": path, "iid": iid_acc, "state": sd, "model": name,
                           "conf": float(payload[tag].max(1).mean())}
        print(f"[sweep] {tag:10s} iid={iid_acc:.4f} conf={individual[tag]['conf']:.4f}", flush=True)
        del model

    # Greedy soup per group: add a model only if IID accuracy of the average improves.
    # 每组贪心 soup：只有当平均权重的 IID 准确率提升时才纳入该模型。
    for prefix in args.soup_groups:
        members = [t for t in individual if t.startswith(prefix)]
        if len(members) < 2:
            continue
        order = sorted(members, key=lambda t: -individual[t]["iid"])
        soup_sd, chosen, best = None, [], -1.0
        for tag in order:
            cand = individual[tag]["state"]
            trial = ({k: v.float() for k, v in cand.items()} if soup_sd is None else
                     {k: (soup_sd[k] * len(chosen) + v.float()) / (len(chosen) + 1) for k, v in cand.items()})
            model = build_model(individual[tag]["model"]).to(device)
            model.load_state_dict(trial)
            model.eval()
            acc = float((flip_probs(model, iid_x, device).argmax(1) == iid_y).mean())
            if acc > best + 1e-6:
                best, soup_sd = acc, trial
                chosen.append(tag)
                print(f"[soup] {prefix}: +{tag} -> iid={acc:.4f}", flush=True)
            del model
        if soup_sd is not None:
            model = build_model(individual[chosen[0]]["model"]).to(device)
            model.load_state_dict(soup_sd)
            model.eval()
            payload[f"soup_{prefix}"] = flip_probs(model, test_x, device).astype(np.float32)
            individual[f"soup_{prefix}"] = {"path": "+".join(chosen), "iid": best,
                                            "conf": float(payload[f"soup_{prefix}"].max(1).mean())}
            print(f"[soup] {prefix}: final iid={best:.4f} members={chosen}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"[sweep] wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")
    print("\n| tag | iid(10k) | test conf |")
    print("|---|---|---|")
    for tag, rec in sorted(individual.items(), key=lambda kv: -kv[1]["iid"]):
        print(f"| {tag} | {rec['iid']:.4f} | {rec['conf']:.4f} |")


if __name__ == "__main__":
    main()
