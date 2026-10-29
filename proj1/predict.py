"""Write competition predictions in sample-submission order."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from model import WideResNet
from refine import probabilities
from train import ROOT, MEAN, STD


class CompetitionImages(Dataset):
    def __init__(self, root):
        with (root/'sample_submission.csv').open() as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ['id', 'label']:
                raise ValueError('Unexpected submission schema')
            self.ids = [row['id'] for row in reader]
        with (root/'test.csv').open() as f:
            rows = list(csv.DictReader(f))
        files = {row['id']: row['filename'] for row in rows}
        if not self.ids or len(set(self.ids)) != len(self.ids) or len(files) != len(rows) or set(files) != set(self.ids):
            raise ValueError('Duplicate or mismatched test/sample IDs')
        self.paths = []
        image_root = (root/'test_images').resolve()
        for identifier in self.ids:
            path = (image_root/files[identifier]).resolve()
            if not path.is_relative_to(image_root) or not path.is_file():
                raise ValueError(f'Missing or invalid image for ID {identifier}')
            self.paths.append(path)
        self.transform = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            if image.size != (32, 32) or image.mode != 'RGB':
                raise ValueError(f'Unexpected image format: {self.paths[index].name}')
            return self.transform(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT/'data/competition')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tta', choices=['plain', 'flip', 'translate'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.out.name != 'submission.csv':
        raise ValueError('Competition requires the filename submission.csv')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = WideResNet(**checkpoint['config']['model']).to(device, memory_format=torch.channels_last).eval()
    model.load_state_dict(checkpoint['model'])
    count = sum(p.numel() for p in model.parameters())
    assert count == checkpoint['config']['parameters'] and count <= 5_000_000
    mode = args.tta or checkpoint.get('inference') or checkpoint.get('selection', {}).get('inference')
    if mode is None:
        raise ValueError('Specify --tta using the held-out validation result')
    dataset = CompetitionImages(args.data)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    predictions = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for images in loader:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            predictions.append(probabilities(model, images, mode).cpu())
    probs = torch.cat(predictions).numpy()
    assert probs.shape == (len(dataset), 10) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(1), 1, atol=1e-5)
    labels = probs.argmax(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(zip(dataset.ids, labels.tolist()))
    with args.out.open() as f:
        rows = list(csv.DictReader(f))
    assert [r['id'] for r in rows] == dataset.ids
    assert all(0 <= int(r['label']) <= 9 for r in rows)
    np.save(args.out.with_suffix('.probabilities.npy'), probs)
    receipt = {'checkpoint': str(args.checkpoint.resolve()), 'parameters': count, 'inference': mode,
               'rows': len(rows), 'label_counts': np.bincount(labels, minlength=10).tolist(),
               'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               'submission_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
