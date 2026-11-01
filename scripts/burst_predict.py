"""Run inference for many checkpoints on the cluster and export probability matrices.

Why: transferring 12 x 38 MB checkpoints through the OnDemand file browser is
fragile, while the prediction we actually need is 2000 x 10 floats per model
(~80 KB). We therefore compute the predictions where the checkpoints live and
bring back only the probabilities; any ensemble can then be built locally.

为什么这么做：通过 OnDemand 文件浏览器传 12×38 MB 的 checkpoint 很脆弱，而我们真正需要
的只是每个模型 2000×10 的概率（约 80 KB）。所以在 checkpoint 所在的地方算完推理，
只回传概率矩阵，任何集成都能在本地拼出来。

Usage (on the cluster) / 在集群上用:
  python scripts/burst_predict.py --ckpts "runs/full_s*/final.pt" "runs/kd_s*/final.pt" \
      --test-dir test_images --out analysis/probs.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data import MEAN, STD  # noqa: E402
from models import build_model  # noqa: E402


def load_test_tensors(test_dir: str, n: int = 2000) -> torch.Tensor:
    """Load test_0000..test_NNNN PNGs into a normalized (N,3,32,32) tensor.
    把 test_0000.. 的 PNG 读成归一化的 (N,3,32,32) 张量。"""
    from PIL import Image
    xs = []
    for i in range(n):
        path = os.path.join(test_dir, f"test_{i:04d}.png")
        if not os.path.exists(path):
            break
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        xs.append(torch.from_numpy(arr).permute(2, 0, 1))
    x = torch.stack(xs)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def flip_probs(model, x: torch.Tensor, device, bs: int = 500) -> np.ndarray:
    """Horizontal-flip TTA probabilities. / 水平翻转 TTA 的概率。"""
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        p = (model(xb).float().softmax(1) + model(xb.flip(-1)).float().softmax(1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="globs; shell handles expansion if quoted / 支持 glob")
    ap.add_argument("--test-dir", default="test_images")
    ap.add_argument("--out", default="analysis/probs.npz")
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.ckpts:
        paths.extend(sorted(glob.glob(pattern)))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        raise SystemExit(f"no checkpoints matched: {args.ckpts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = load_test_tensors(args.test_dir)
    print(f"[predict] {len(paths)} checkpoints, test images {tuple(x.shape)}, device {device}", flush=True)

    payload = {"n": np.int64(len(x))}
    for path in paths:
        tag = os.path.basename(os.path.dirname(path))  # e.g. full_s0 / kd_s10
        ck = torch.load(path, map_location=device, weights_only=False)
        cargs = ck.get("args", {})
        model = build_model(cargs.get("model", "preact16x5.25")).to(device)
        model.load_state_dict(ck.get(args.weights, ck["model"]))
        model.eval()
        prob = flip_probs(model, x, device)
        payload[tag] = prob.astype(np.float32)
        conf = float(prob.max(1).mean())
        print(f"[predict] {tag:10s} epochs={ck.get('epoch')} teacher={os.path.basename(cargs.get('teacher', '') or '-')} "
              f"mean_conf={conf:.4f}", flush=True)
        del model

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    size = os.path.getsize(args.out) / 1024
    print(f"[predict] wrote {args.out} ({size:.0f} KB) with keys {[k for k in payload if k != 'n']}")


if __name__ == "__main__":
    main()
