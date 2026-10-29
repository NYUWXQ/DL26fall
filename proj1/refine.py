"""Compare test-time augmentation and BatchNorm recalibration on validation."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from model import WideResNet
from train import ROOT, MEAN, STD, atomic_save, split_indices


@torch.inference_mode()
def probabilities(model, images, mode):
    if mode not in ('plain', 'flip', 'translate'):
        raise ValueError(mode)
    views = [images]
    if mode == 'translate':
        padded = F.pad(images, (2, 2, 2, 2), mode='reflect')
        views += [padded[:, :, y:y+32, x:x+32] for y, x in ((0,0), (0,4), (4,0), (4,4))]
    if mode != 'plain':
        views += [view.flip(-1) for view in list(views)]
    return sum(model(view).float().softmax(1) for view in views) / len(views)


@torch.inference_mode()
def score(model, loader, device, mode):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = probabilities(model, images, mode).argmax(1).cpu()
        correct += (prediction == labels).sum().item()
        total += len(labels)
    return {'accuracy': correct/total, 'correct': correct, 'total': total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['config']['full_data']:
        raise ValueError('A model trained on all 50k images has no held-out validation split')
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_save(checkpoint, args.out / 'source.pt')
    model = WideResNet(**checkpoint['config']['model']).to(device, memory_format=torch.channels_last)
    data = CIFAR10(ROOT/'data', train=True, transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]))
    train, val = split_indices(data.targets)
    train_loader = DataLoader(Subset(data, train), batch_size=256, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(data, val), batch_size=500, num_workers=2, pin_memory=True)
    weights = [('model', checkpoint['model'])]
    if 'ema' in checkpoint:
        weights.append(('ema', checkpoint['ema']))
    results, best = [], -1.
    for name, state in weights:
        model.load_state_dict(state)
        for calibrated in (False, True):
            if calibrated:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    update_bn(train_loader, model, device)
            for mode in ('plain', 'flip', 'translate'):
                record = {'weights': name, 'calibrate_bn': calibrated, 'inference': mode,
                          **score(model, val_loader, device, mode)}
                results.append(record)
                print(json.dumps(record), flush=True)
                if record['accuracy'] > best:
                    best = record['accuracy']
                    selected = record
                    atomic_save({'model': model.state_dict(), 'config': checkpoint['config'],
                                 'epoch': checkpoint['epoch'], 'selection': record}, args.out/'best.pt')
    (args.out/'result.json').write_text(json.dumps({'selected': selected, 'comparisons': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
