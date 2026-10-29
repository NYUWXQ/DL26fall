"""Train a Wide ResNet on CIFAR-10 with a fixed 5,000-image validation split."""
import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from model import WideResNet

ROOT = Path(__file__).resolve().parent
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def split_indices(targets):
    rng = np.random.default_rng(2026)
    train, val = [], []
    targets = np.asarray(targets)
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(targets == label))
        val.extend(indices[:500].tolist())
        train.extend(indices[500:].tolist())
    return np.asarray(train), np.asarray(val)


def mix_batch(images, labels, mixup, cutmix):
    if mixup == 0 and cutmix == 0:
        return images, labels, labels, 1.0
    order = torch.randperm(len(labels), device=images.device)
    if cutmix > 0 and (mixup == 0 or random.random() < 0.5):
        lam = np.random.beta(cutmix, cutmix)
        height, width = images.shape[-2:]
        side = math.sqrt(1 - lam)
        cx, cy = np.random.randint(width), np.random.randint(height)
        left, right = max(0, cx - int(width * side) // 2), min(width, cx + int(width * side) // 2)
        top, bottom = max(0, cy - int(height * side) // 2), min(height, cy + int(height * side) // 2)
        images[:, :, top:bottom, left:right] = images[order, :, top:bottom, left:right]
        lam = 1 - (right - left) * (bottom - top) / (height * width)
    else:
        lam = np.random.beta(mixup, mixup)
        images = images * lam + images[order] * (1 - lam)
    return images, labels, labels[order], float(lam)


@torch.inference_mode()
def evaluate(model, loader, device, flip=False):
    model.eval()
    correct, loss, total = 0, 0., 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images).float()
            if flip:
                probs = (logits.softmax(1) + model(images.flip(-1)).float().softmax(1)) / 2
                logits = probs.clamp_min(1e-10).log()
        correct += (logits.argmax(1) == labels).sum().item()
        loss += F.cross_entropy(logits, labels, reduction='sum').item()
        total += len(labels)
    return {'accuracy': correct / total, 'loss': loss / total, 'correct': correct, 'total': total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--width', type=int, default=58)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--randaugment', type=int, default=0)
    parser.add_argument('--cutout', type=int, default=16)
    parser.add_argument('--mixup', type=float, default=0.)
    parser.add_argument('--cutmix', type=float, default=0.)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--full-data', action='store_true')
    parser.add_argument('--max-batches', type=int, default=0, help='Short smoke test only')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        parser.error('Invalid training configuration')
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=args.resume)
    model_config = dict(depth=args.depth, width=args.width, dropout=args.dropout)
    model = WideResNet(**model_config).to(device, memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9,
                                weight_decay=args.weight_decay, nesterov=True)
    normalize = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    augment = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if args.randaugment:
        augment.append(T.RandAugment(num_ops=2, magnitude=args.randaugment))
    augment.append(normalize)
    train_data = CIFAR10(ROOT / 'data', train=True, transform=T.Compose(augment))
    val_data = CIFAR10(ROOT / 'data', train=True, transform=normalize)
    train_indices, val_indices = split_indices(train_data.targets)
    if args.full_data:
        train_indices = np.arange(len(train_data))
    split_hash = hashlib.sha256(train_indices.tobytes() + val_indices.tobytes()).hexdigest()
    loader_args = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=args.batch_size,
                              shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=500, shuffle=False, **loader_args)
    config = {**vars(args), 'model': model_config, 'parameters': sum(p.numel() for p in model.parameters()),
              'split_sha256': split_hash, 'train_images': len(train_indices), 'validation_images': 0 if args.full_data else len(val_indices),
              'normalization_mean': MEAN, 'normalization_std': STD,
              'torch_version': str(torch.__version__), 'training_data': 'CIFAR-10 official train split',
              'seed': 42, 'selection': 'maximum raw or EMA validation accuracy' if not args.full_data else 'final EMA'}
    start_epoch, best, steps, elapsed = 0, 0., 0, 0.
    if args.resume:
        state = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        for key in ('model', 'epochs', 'batch_size', 'lr', 'weight_decay', 'randaugment', 'cutout', 'mixup', 'cutmix', 'split_sha256', 'full_data', 'max_batches'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume configuration changed: {key}')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, best, steps, elapsed = state['epoch'], state['best'], state['steps'], state['elapsed_seconds']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
    else:
        (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        np.savez(out / 'split.npz', train=train_indices, val=val_indices)
        for source in ('train.py', 'model.py'):
            shutil.copy2(ROOT / source, out / source)
    print(json.dumps(config), flush=True)
    clock_start = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.monotonic()
        model.train()
        loss_sum, count = 0., 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            progress = epoch + batch_idx / len(train_loader)
            warmup = min(5, args.epochs / 10)
            lr = args.lr * min(1., (progress + 1 / len(train_loader)) / warmup)
            lr *= .5 * (1 + math.cos(math.pi * max(0., progress - warmup) / (args.epochs - warmup)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            if args.cutout:
                centers = torch.randint(0, 32, (len(images), 2, 1, 1), device=device)
                row = torch.arange(32, device=device).view(1, 32, 1)
                col = torch.arange(32, device=device).view(1, 1, 32)
                row_mask = (row >= centers[:, 0] - args.cutout // 2) & (row < centers[:, 0] + (args.cutout + 1) // 2)
                col_mask = (col >= centers[:, 1] - args.cutout // 2) & (col < centers[:, 1] + (args.cutout + 1) // 2)
                mask = row_mask & col_mask
                images = images.masked_fill(mask.unsqueeze(1), 0)
            images, ya, yb, lam = mix_batch(images, labels, args.mixup, args.cutmix)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images)
                loss = lam * F.cross_entropy(logits, ya) + (1 - lam) * F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch+1}, batch {batch_idx}')
            loss.backward()
            optimizer.step()
            steps += 1
            decay = min(.999, (1 + steps) / (10 + steps))
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)
                for dst, src in zip(ema.buffers(), model.buffers()):
                    dst.copy_(src)
            loss_sum += loss.detach().item() * len(labels)
            count += len(labels)
            if args.max_batches and batch_idx + 1 >= args.max_batches:
                break
        metrics = {'epoch': epoch+1, 'train_loss': loss_sum/count, 'lr': lr}
        if not args.full_data:
            raw_result = evaluate(model, val_loader, device)
            ema_result = evaluate(ema, val_loader, device)
            metrics.update(val_accuracy=raw_result['accuracy'], ema_accuracy=ema_result['accuracy'],
                           val_loss=raw_result['loss'], ema_loss=ema_result['loss'])
            chosen, score = (ema, ema_result['accuracy']) if ema_result['accuracy'] >= raw_result['accuracy'] else (model, raw_result['accuracy'])
            if score > best:
                best = score
                atomic_save({'model': chosen.state_dict(), 'config': config, 'epoch': epoch+1,
                             'val_accuracy': best, 'weights': 'ema' if chosen is ema else 'raw'}, out / 'best.pt')
        metrics.update(best_accuracy=best, epoch_seconds=time.monotonic()-epoch_start,
                       elapsed_seconds=elapsed+time.monotonic()-clock_start)
        state = {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch+1, 'config': config, 'best': best, 'steps': steps,
                 'elapsed_seconds': metrics['elapsed_seconds'], 'python_rng': random.getstate(),
                 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state(device)}
        atomic_save(state, out / 'last.pt')
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
    if args.full_data:
        atomic_save({'model': ema.state_dict(), 'config': config, 'epoch': args.epochs, 'weights': 'ema'}, out / 'final.pt')
        result = {'status': 'complete', 'epochs': args.epochs, 'elapsed_seconds': elapsed+time.monotonic()-clock_start}
    else:
        checkpoint = torch.load(out / 'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        result = {'status': 'complete', 'best_epoch': checkpoint['epoch'], 'weights': checkpoint['weights'],
                  'plain': evaluate(model, val_loader, device), 'flip': evaluate(model, val_loader, device, flip=True),
                  'elapsed_seconds': elapsed+time.monotonic()-clock_start}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
