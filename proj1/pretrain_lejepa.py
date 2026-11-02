"""Pretrain a residual backbone with LeJEPA on the fixed 45,000-image split."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from lejepa import LeJEPAEncoder, MultiView, SIGReg, lejepa_loss
from train import ROOT, atomic_save, split_indices

UPSTREAM_COMMIT = 'c293d291ca87cd4fddee9d3fffe4e914c7272052'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/lejepa_frontloaded.json')
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Check configuration and parameter counts on CPU')
    args = parser.parse_args()
    specification = json.loads(args.config.read_text())
    settings = specification['pretrain']
    if settings['epochs'] < 1 or settings['batch_size'] < 2 or not 0 < settings['sigreg_weight'] < 1:
        parser.error('Invalid epochs, batch size, or SIGReg weight')
    if settings['lr'] <= 0 or settings['weight_decay'] < 0 or args.workers < 0:
        parser.error('Invalid optimizer or worker settings')
    if specification['model'].get('teacher'):
        parser.error('Use a student backbone under the 5M parameter limit')
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    encoder = LeJEPAEncoder(specification['model'], settings['hidden'], settings['projection_dim'])
    augmentation = MultiView(settings['views'])
    regularizer = SIGReg(directions=settings['directions'])
    parameter_counts = {
        'classifier_parameters': sum(p.numel() for p in encoder.backbone.parameters()),
        'temporary_projector_parameters': sum(p.numel() for p in encoder.projector.parameters()),
        'ssl_trainable_parameters': sum(p.numel() for p in encoder.parameters() if p.requires_grad),
    }
    if args.check_only:
        print(json.dumps({'status': 'configuration_checked', **parameter_counts, 'pretrain': settings}), flush=True)
        return
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    out = ROOT / 'runs' / args.name
    out.mkdir(parents=True, exist_ok=args.resume)
    data = CIFAR10(ROOT / 'data', train=True, transform=augmentation, download=False)
    train, validation = split_indices(data.targets)
    split_hash = hashlib.sha256(train.tobytes() + validation.tobytes()).hexdigest()
    config = {**settings, 'model': specification['model'], 'name': args.name,
              'parameters': parameter_counts['classifier_parameters'], **parameter_counts,
              'training_data': 'CIFAR-10 official train split', 'train_images': len(train),
              'full_data': False, 'max_batches': 0, 'split_sha256': split_hash, 'seed': 42,
              'selection': 'final pretraining epoch; labels used only for the fixed split',
              'upstream_commit': UPSTREAM_COMMIT,
              'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    loader = DataLoader(Subset(data, train), batch_size=settings['batch_size'], shuffle=True,
                        drop_last=True, num_workers=args.workers, pin_memory=device.type == 'cuda',
                        persistent_workers=args.workers > 0)
    encoder.to(device, memory_format=torch.channels_last)
    regularizer.to(device)
    optimizer = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad),
                                  lr=settings['lr'], weight_decay=settings['weight_decay'])
    start, elapsed = 0, 0.0
    if args.resume:
        state = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        for key in (*settings.keys(), 'model', 'split_sha256'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Pretraining resume configuration changed: {key}')
        encoder.load_state_dict(state['encoder'])
        optimizer.load_state_dict(state['optimizer'])
        start, elapsed = state['epoch'], state['elapsed_seconds']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
    else:
        (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        np.savez(out / 'split.npz', train=train, val=validation)
        for name in ('pretrain_lejepa.py', 'lejepa.py', 'train.py', 'model.py', 'shakeshake.py'):
            shutil.copy2(ROOT / name, out / name)
        shutil.copy2(ROOT / 'licenses/lejepa.txt', out / 'LEJEPA_LICENSE.txt')
    print(json.dumps(config), flush=True)
    started = time.monotonic()
    for epoch in range(start, settings['epochs']):
        encoder.train()
        totals = np.zeros(4)
        examples = 0
        for batch, (views, _) in enumerate(loader):
            progress = epoch + batch / len(loader)
            warmup = min(1.0, settings['epochs'] / 10)
            lr = settings['lr'] * min(1.0, (progress + 1 / len(loader)) / warmup)
            lr *= 0.5 * (1 + math.cos(math.pi * max(0.0, progress - warmup) / (settings['epochs'] - warmup)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            views = views.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                projections = encoder(views)
                loss, invariance, isotropy = lejepa_loss(projections, regularizer, settings['sigreg_weight'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite LeJEPA loss at epoch {epoch + 1}, batch {batch}')
            loss.backward()
            optimizer.step()
            deviation = projections.detach().float().std(dim=1).mean()
            totals += np.asarray([loss.item(), invariance.item(), isotropy.item(), deviation.item()]) * len(views)
            examples += len(views)
        metrics = dict(zip(('loss', 'invariance', 'sigreg', 'projection_std'), (totals / examples).tolist()))
        metrics.update(epoch=epoch + 1, lr=lr, elapsed_seconds=elapsed + time.monotonic() - started)
        atomic_save({'encoder': encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
                     'epoch': epoch + 1, 'elapsed_seconds': metrics['elapsed_seconds'],
                     'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                     'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}, out / 'last.pt')
        with (out / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
    atomic_save({'kind': 'lejepa_backbone', 'status': 'complete', 'model': encoder.backbone.state_dict(),
                 'config': config, 'epoch': settings['epochs']}, out / 'backbone.pt')
    result = {'status': 'complete', 'epochs': settings['epochs'], **parameter_counts,
              'elapsed_seconds': elapsed + time.monotonic() - started, 'backbone': str(out / 'backbone.pt')}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
