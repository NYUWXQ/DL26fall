"""Save per-class errors and confident mistakes on the training-set holdout."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from model import build_model
from refine import probabilities
from train import ROOT, MEAN, STD, split_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--tta', choices=['plain', 'flip', 'translate'], default='flip')
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state.get('kind') == 'lejepa_backbone':
        raise ValueError('Fine-tune the self-supervised backbone before validation')
    if state['config']['full_data']:
        raise ValueError('Use a checkpoint trained on the 45,000-image split')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = build_model(state['config']['model']).to(device).eval()
    model.load_state_dict(state['model'])
    data = CIFAR10(ROOT / 'data', train=True, transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]))
    _, indices = split_indices(data.targets)
    loader = DataLoader(Subset(data, indices), batch_size=64, num_workers=2)
    predictions = []
    with torch.inference_mode():
        for images, _ in loader:
            predictions.append(probabilities(model, images.to(device), args.tta).cpu().numpy())
    probs = np.concatenate(predictions)
    labels = np.asarray(data.targets)[indices]
    predicted = probs.argmax(1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    assert confusion.sum() == 5000 and np.all(confusion.sum(1) == 500)
    np.savez(args.out / 'predictions.npz', indices=indices, labels=labels, probabilities=probs)
    with (args.out / 'errors.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['train_index', 'true_class', 'predicted_class', 'confidence'])
        mistakes = np.flatnonzero(predicted != labels)
        mistakes = mistakes[np.argsort(-probs[mistakes].max(1))]
        for i in mistakes:
            writer.writerow([int(indices[i]), data.classes[labels[i]], data.classes[predicted[i]], float(probs[i].max())])
    summary = {
        'checkpoint': str(args.checkpoint), 'inference': args.tta, 'device': str(device),
        'accuracy': float((predicted == labels).mean()), 'errors': len(mistakes),
        'per_class_accuracy': dict(zip(data.classes, (confusion.diagonal() / 500).tolist())),
        'confusion': confusion.tolist(), 'rows': 'true class', 'columns': 'predicted class',
    }
    (args.out / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    canvas = Image.new('RGB', (6 * 160, 6 * 150), 'white')
    draw = ImageDraw.Draw(canvas)
    for position, i in enumerate(mistakes[:36]):
        x, y = position % 6 * 160, position // 6 * 150
        tile = Image.fromarray(data.data[indices[i]]).resize((96, 96), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, y))
        draw.text((x, y + 98), f'{data.classes[labels[i]]} -> {data.classes[predicted[i]]}', fill='black')
        draw.text((x, y + 115), f'index {indices[i]}  p={probs[i].max():.3f}', fill='black')
    canvas.save(args.out / 'confident_errors.png')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
