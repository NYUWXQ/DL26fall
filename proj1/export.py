"""Export full-data weights with the inference settings selected on validation."""
import argparse
import json
from pathlib import Path

import torch
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from model import build_model
from refine import probabilities
from train import ROOT, MEAN, STD, atomic_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='Completed full-data run last.pt')
    parser.add_argument('--recipe', type=Path, required=True, help='Completed refine.py output directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    source = torch.load(args.recipe/'source.pt', map_location='cpu', weights_only=False)
    recipe = json.loads((args.recipe/'result.json').read_text())['selected']
    precision = recipe.get('precision', 'bf16')
    if precision == 'fp32':
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    config = state['config']
    if config.get('teacher_only') or config['parameters'] > 5_000_000:
        raise ValueError('Competition export requires a student model with at most 5M parameters')
    if not config['full_data'] or state['epoch'] != config['epochs'] or config['max_batches']:
        raise ValueError('A completed full-data training checkpoint is required')
    if source['config']['full_data']:
        raise ValueError('Inference recipe must be selected on held-out images')
    for key in ('model', 'epochs', 'batch_size', 'lr', 'weight_decay', 'randaugment', 'cutout', 'mixup', 'cutmix'):
        if config[key] != source['config'][key]:
            raise ValueError(f'Validation and full-data training differ: {key}')
    for key, default in (('teacher_sha256', None), ('extra_teacher_sha256', []), ('distill_weight', 0.5),
                         ('temperature', 4.0), ('autoaugment', False), ('initialization_sha256', None),
                         ('ricap', 0.), ('random_erasing', 0.), ('warmup_epochs', 5.), ('min_lr', 0.)):
        if config.get(key, default) != source['config'].get(key, default):
            raise ValueError(f'Validation and full-data distillation differ: {key}')
    kind = 'ema' if recipe['weights'] == 'ema' else source.get('weights', 'raw')
    device = torch.device(args.device)
    model = build_model(config['model']).to(device, memory_format=torch.channels_last)
    model.load_state_dict(state['ema' if kind == 'ema' else 'model'])
    data = CIFAR10(ROOT/'data', train=True, transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]))
    if recipe['calibrate_bn']:
        loader = DataLoader(data, batch_size=256, num_workers=2, pin_memory=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
            update_bn(loader, model, device)
    model.eval()
    with torch.inference_mode():
        prediction = probabilities(model, torch.stack([data[0][0], data[1][0]]).to(device), recipe['inference'])
        assert prediction.shape == (2, 10) and torch.isfinite(prediction).all()
        assert torch.allclose(prediction.sum(1), torch.ones(2, device=device), atol=1e-5)
    model.cpu()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save({'model': model.state_dict(), 'config': config, 'weights': kind,
                 'classes': data.classes, 'inference': recipe['inference'],
                 'inference_precision': precision,
                 'calibrate_bn': recipe['calibrate_bn'], 'validation_recipe': recipe,
                 'validation_source': str(args.recipe), 'epoch': state['epoch']}, args.out)
    print(json.dumps({'output': str(args.out), 'weights': kind, 'inference': recipe['inference'],
                      'parameters': config['parameters'], 'train_images': config['train_images']}))


if __name__ == '__main__':
    main()
