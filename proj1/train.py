"""Train on CIFAR-10 with a fixed holdout or stratified five-fold evaluation."""
import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from model import build_model
from optimizations import sam_step
from modern_kd import ea_loss, rld_loss

ROOT = Path(__file__).resolve().parent
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def cosine_lr(progress, epochs, peak, warmup_epochs, minimum, batches):
    warmup = min(warmup_epochs, epochs / 10)
    ramp = min(1., (progress + 1 / batches) / warmup) if warmup else 1.
    value = peak * ramp
    value *= .5 * (1 + math.cos(math.pi * max(0., progress - warmup) / (epochs - warmup)))
    return max(minimum, value)


def split_indices(targets, audit_holdout=False, cv_fold=None):
    if cv_fold is not None and (cv_fold not in range(5) or audit_holdout):
        raise ValueError('Five-fold evaluation needs fold 0..4 and no audit holdout')
    rng = np.random.default_rng(2026)
    train, val = [], []
    targets = np.asarray(targets)
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(targets == label))
        if cv_fold is not None:
            parts = np.array_split(indices, 5)
            val.extend(parts[cv_fold].tolist())
            train.extend(np.concatenate([part for i, part in enumerate(parts) if i != cv_fold]).tolist())
            continue
        val.extend(indices[:500].tolist())
        train.extend(indices[1000 if audit_holdout else 500:].tolist())
    return np.asarray(train), np.asarray(val)


class AugMixViews:
    def __init__(self):
        self.augment = T.AugMix(severity=3, all_ops=False)
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __call__(self, image):
        return (self.normalize(image), self.normalize(self.augment(image)),
                self.normalize(self.augment(image)))


class TwoViews:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)


def two_view_js(logits):
    first, second = logits.float().chunk(2)
    probabilities = (first.softmax(1), second.softmax(1))
    mixture = ((probabilities[0] + probabilities[1]) / 2).clamp_min(1e-7).log()
    return sum(F.kl_div(mixture, value, reduction='batchmean') for value in probabilities) / 2


def two_view_loss(logits, targets, weights, strength):
    first, second = logits.float().chunk(2)
    supervised = sum(weight * (F.cross_entropy(first, target) + F.cross_entropy(second, target)) / 2
                     for target, weight in zip(targets, weights))
    return supervised + strength * two_view_js(logits)


def augmix_loss(logits, labels):
    clean, first, second = logits.float().chunk(3)
    distributions = [value.softmax(1) for value in (clean, first, second)]
    mixture = torch.stack(distributions).mean(0).clamp_min(1e-7).log()
    consistency = sum(F.kl_div(mixture, value, reduction='batchmean') for value in distributions) / 3
    return F.cross_entropy(clean, labels) + 12 * consistency


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


def ricap_batch(images, labels, beta):
    height, width = images.shape[-2:]
    left, top = np.round(np.random.beta(beta, beta, 2) * [width, height]).astype(int)
    patches, targets, weights = [], [], []
    for w, h in ((left, top), (width - left, top), (left, height - top), (width - left, height - top)):
        order = torch.randperm(len(labels), device=images.device)
        x, y = np.random.randint(width - w + 1), np.random.randint(height - h + 1)
        patches.append(images[order, :, y:y+h, x:x+w])
        targets.append(labels[order])
        weights.append(w * h / (width * height))
    mixed = torch.cat([torch.cat(patches[:2], dim=3), torch.cat(patches[2:], dim=3)], dim=2)
    return mixed, targets, weights


def distillation_loss(student, teacher, temperature):
    teachers = teacher if isinstance(teacher, (list, tuple)) else [teacher]
    target = torch.stack([F.softmax(logits.detach().float() / temperature, dim=1)
                          for logits in teachers]).mean(0)
    return F.kl_div(F.log_softmax(student.float() / temperature, dim=1),
                    target, reduction='batchmean') * temperature**2


def validate_teacher(config, student_config):
    if config['training_data'] != student_config['training_data'] or config.get('max_batches', 0):
        raise ValueError('Teacher must be trained on the allowed CIFAR-10 data')
    if not student_config['full_data']:
        if config['full_data'] or config['split_sha256'] != student_config['split_sha256']:
            raise ValueError('Teacher must exclude the student validation images')
    if student_config.get('cv_fold') is not None and config.get('cv_fold') != student_config['cv_fold']:
        raise ValueError('Cross-validation teacher must be trained on the matching fold')


def initialize_backbone(model, source, config):
    if source.get('kind') != 'lejepa_backbone' or source.get('status') != 'complete':
        raise ValueError('Use a completed LeJEPA backbone checkpoint')
    source_config = source['config']
    if source['epoch'] != source_config['epochs']:
        raise ValueError('Pretraining has not reached its configured final epoch')
    if source_config['model'] != config['model']:
        raise ValueError('Pretraining and classifier architectures differ')
    if source_config['training_data'] != config['training_data'] or source_config.get('max_batches', 0):
        raise ValueError('Pretraining must use the allowed CIFAR-10 training data')
    if not config['full_data'] and (source_config['full_data'] or source_config['split_sha256'] != config['split_sha256']):
        raise ValueError('Pretraining must exclude the classifier validation images')
    weights = {key: value for key, value in source['model'].items() if not key.startswith('fc.')}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
        raise ValueError('Pretraining checkpoint does not match the backbone')


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
    parser.add_argument('--architecture', choices=['wrn', 'shake', 'se', 'star', 'pyramid'], default='wrn')
    parser.add_argument('--pyramid-alpha', type=int, default=96)
    parser.add_argument('--shakedrop-death-rate', type=float, help='Pyramid final-block drop probability; default 0.5')
    parser.add_argument('--star-variant', choices=['s1', 's2'], default='s1')
    parser.add_argument('--optimizer', choices=['sgd', 'adamw'], default='sgd')
    parser.add_argument('--distill-method', choices=['kl', 'rld', 'ea'], default='kl')
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--width', type=int, default=58)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--stage-widths', type=int, nargs=3)
    parser.add_argument('--stage-blocks', type=int, nargs=3)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--teacher-only', action='store_true', help='Allow an oversized training-only teacher')
    parser.add_argument('--teacher-checkpoint')
    parser.add_argument('--extra-teacher-checkpoint', action='append', default=[])
    parser.add_argument('--init-checkpoint', help='Locally trained LeJEPA backbone')
    parser.add_argument('--init-supervised', help='Completed local supervised run; excludes held-out images')
    parser.add_argument('--sam-rho', type=float, default=0., help='SAM perturbation radius; zero uses SGD')
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--swa-start', type=float, default=0.5, help='Fraction of training before SWA begins')
    parser.add_argument('--swa-lr', type=float, default=0.005)
    parser.add_argument('--distill-weight', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--warmup-epochs', type=float, default=5.)
    parser.add_argument('--min-lr', type=float, default=0.)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--randaugment', type=int, default=0)
    parser.add_argument('--autoaugment', action='store_true')
    parser.add_argument('--augmix', action='store_true', help='Three views with JSD consistency')
    parser.add_argument('--two-views', action='store_true', help='Two independent views with shared RICAP layout')
    parser.add_argument('--two-view-js', type=float, default=0., help='Final two-view JSD weight after warmup')
    parser.add_argument('--vicreg-weight', type=float, help='Three-view auxiliary training; zero is the view-matched control')
    parser.add_argument('--audit-holdout', action='store_true', help='Reserve another 5k from training for final evaluation')
    parser.add_argument('--cv-fold', type=int, choices=range(5))
    parser.add_argument('--cv-weights', choices=['raw', 'ema'], default='raw')
    parser.add_argument('--cutout', type=int, default=16)
    parser.add_argument('--mixup', type=float, default=0.)
    parser.add_argument('--cutmix', type=float, default=0.)
    parser.add_argument('--ricap', type=float, default=0., help='RICAP beta; replaces Mixup/CutMix')
    parser.add_argument('--random-erasing', type=float, default=0., help='Probability of square erasing, area 0.125–0.2')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--compile-model', action='store_true')
    parser.add_argument('--full-data', action='store_true')
    parser.add_argument('--max-batches', type=int, default=0, help='Short smoke test only')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        parser.error('Invalid training configuration')
    if args.eval_batch_size < 1 or not 0 <= args.distill_weight <= 1 or args.temperature <= 0:
        parser.error('Invalid evaluation batch size or distillation settings')
    if args.teacher_only and (args.teacher_checkpoint or args.extra_teacher_checkpoint):
        parser.error('Choose teacher training or student distillation')
    if args.extra_teacher_checkpoint and not args.teacher_checkpoint:
        parser.error('Extra teachers require --teacher-checkpoint')
    if args.distill_method != 'kl' and (not args.teacher_checkpoint or args.extra_teacher_checkpoint
                                       or args.ricap or args.mixup or args.cutmix):
        parser.error('RLD/EA use one teacher and unmixed training images')
    if args.optimizer != 'sgd' and (args.sam_rho or args.swa):
        parser.error('SAM/SWA currently use SGD')
    if args.autoaugment and args.randaugment:
        parser.error('Choose AutoAugment or RandAugment')
    if not math.isfinite(args.two_view_js) or args.two_view_js < 0 or (args.two_view_js and not args.two_views):
        parser.error('Nonnegative two-view JSD weight requires --two-views')
    if args.two_views and (args.augmix or args.teacher_checkpoint):
        parser.error('Two-view training uses supervised loss and its own consistency term')
    if args.vicreg_weight is not None:
        if not math.isfinite(args.vicreg_weight) or args.vicreg_weight < 0:
            parser.error('VICReg weight must be finite and nonnegative')
        if (args.batch_size < 2 or args.cv_fold is None or args.optimizer != 'sgd'
                or any((args.two_views, args.augmix, args.teacher_checkpoint, args.sam_rho, args.swa))):
            parser.error('VICReg pilot requires fold training, SGD, batch>=2 and its own three-view loss')
    if args.augmix and any((args.autoaugment, args.randaugment, args.ricap, args.mixup,
                           args.cutmix, args.cutout, args.random_erasing, args.teacher_checkpoint)):
        parser.error('AugMix uses its own three-view augmentation and supervised loss; set --cutout 0')
    if args.audit_holdout and args.full_data:
        parser.error('Audit holdout and full-data training are separate runs')
    if args.cv_fold is not None and any((args.full_data, args.audit_holdout, args.init_checkpoint, args.swa)):
        parser.error('Fold training requires its own held-out split and fixed final raw/EMA weights')
    if args.init_supervised and args.init_checkpoint:
        parser.error('Choose one initialization checkpoint')
    if args.sam_rho < 0 or not math.isfinite(args.sam_rho):
        parser.error('SAM radius must be finite and nonnegative')
    if args.swa and (not 0 <= args.swa_start < 1 or not 0 < args.swa_lr <= args.lr):
        parser.error('Invalid SWA start or learning rate')
    if args.architecture != 'wrn' and (args.stage_widths or args.stage_blocks or args.groups != 1 or args.dropout or args.teacher_only):
        parser.error('SE and Shake-Shake use depth and width; WRN-specific options are unsupported')
    if args.ricap < 0 or (args.ricap and (args.mixup or args.cutmix)):
        parser.error('RICAP requires a positive beta and replaces Mixup/CutMix')
    if (not 0 <= args.random_erasing <= 1 or not math.isfinite(args.warmup_epochs)
            or args.warmup_epochs < 0 or not 0 <= args.min_lr <= args.lr):
        parser.error('Invalid erasing probability or learning-rate schedule')
    if args.shakedrop_death_rate is not None and (args.architecture != 'pyramid'
            or not math.isfinite(args.shakedrop_death_rate) or not 0 <= args.shakedrop_death_rate <= 1):
        parser.error('ShakeDrop death rate requires Pyramid and a value between zero and one')
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
    if args.stage_widths is not None or args.stage_blocks is not None or args.groups != 1:
        model_config.update(stage_widths=args.stage_widths, stage_blocks=args.stage_blocks, groups=args.groups)
    if args.teacher_only:
        model_config['teacher'] = True
    if args.architecture != 'wrn':
        model_config = {'architecture': args.architecture, 'depth': args.depth, 'width': args.width}
    if args.architecture == 'star':
        model_config = {'architecture': 'star', 'variant': args.star_variant}
    if args.architecture == 'pyramid':
        model_config = {'architecture': 'pyramid', 'depth': args.depth, 'alpha': args.pyramid_alpha}
        if args.shakedrop_death_rate is not None:
            model_config['death_rate'] = args.shakedrop_death_rate
    model = build_model(model_config).to(device, memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if args.optimizer == 'adamw':
        decay = [p for p in model.parameters() if p.ndim > 1]
        no_decay = [p for p in model.parameters() if p.ndim <= 1]
        optimizer = torch.optim.AdamW([{'params': decay}, {'params': no_decay, 'weight_decay': 0.}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9,
                                    weight_decay=args.weight_decay, nesterov=True)
    head = None
    if args.vicreg_weight is not None:
        from vicreg import ThreeViews, auxiliary_loss, projector
        head = projector(model.fc.in_features).to(device)
        optimizer.add_param_group({'params': head.parameters()})
    normalize = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    augment = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if args.randaugment:
        augment.append(T.RandAugment(num_ops=2, magnitude=args.randaugment))
    if args.autoaugment:
        augment.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    augment.append(AugMixViews() if args.augmix else normalize)
    if args.random_erasing:
        augment.append(T.RandomErasing(p=args.random_erasing, scale=(0.125, 0.2), ratio=(0.99, 1.0)))
    transform = T.Compose(augment)
    view_transform = ThreeViews(transform) if head is not None else TwoViews(transform) if args.two_views else transform
    train_data = CIFAR10(ROOT / 'data', train=True, transform=view_transform)
    val_data = CIFAR10(ROOT / 'data', train=True, transform=normalize)
    train_indices, val_indices = split_indices(train_data.targets, args.audit_holdout, args.cv_fold)
    if args.full_data:
        train_indices = np.arange(len(train_data))
    split_hash = hashlib.sha256(train_indices.tobytes() + val_indices.tobytes()).hexdigest()
    loader_args = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=args.batch_size,
                              shuffle=True, **loader_args)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=args.eval_batch_size, shuffle=False, **loader_args)
    config = {**vars(args), 'model': model_config, 'parameters': sum(p.numel() for p in model.parameters()),
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'split_sha256': split_hash, 'train_images': len(train_indices), 'validation_images': 0 if args.full_data else len(val_indices),
              'normalization_mean': MEAN, 'normalization_std': STD,
              'torch_version': str(torch.__version__), 'training_data': 'CIFAR-10 official train split',
              'seed': 42, 'selection': 'maximum raw or EMA validation accuracy' if not args.full_data else 'final EMA'}
    if args.cv_fold is not None:
        config['selection'] = 'fixed final epoch: ' + args.cv_weights
    if head is not None:
        config['projector_parameters'] = sum(p.numel() for p in head.parameters())
        config['vicreg_coefficients'] = [25, 25, 1]
    if args.init_checkpoint:
        initialization_path = Path(args.init_checkpoint)
        source = torch.load(initialization_path, map_location='cpu', weights_only=False)
        initialize_backbone(model, source, config)
        ema.load_state_dict(model.state_dict())
        config['initialization_sha256'] = hashlib.sha256(initialization_path.read_bytes()).hexdigest()
        del source
    if args.init_supervised:
        source_path = Path(args.init_supervised).resolve()
        if not source_path.is_relative_to((ROOT / 'runs').resolve()):
            raise ValueError('Supervised initialization must come from a local project run')
        source_result = json.loads((source_path.parent / 'result.json').read_text())
        if source_result['status'] != 'complete':
            raise ValueError('Supervised initialization run is incomplete')
        source = torch.load(source_path, map_location='cpu', weights_only=False)
        validate_teacher(source['config'], config)
        if args.cv_fold is not None and source['epoch'] != source['config']['epochs']:
            raise ValueError('Fold continuation requires the completed fixed-epoch source')
        if source['config']['model'] != config['model']:
            raise ValueError('Supervised initialization architecture differs')
        model.load_state_dict(source['model'])
        ema.load_state_dict(model.state_dict())
        config['supervised_initialization_sha256'] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        config['source_training_epochs'] = source['config']['epochs']
        del source
    swa = AveragedModel(model) if args.swa else None
    teachers = []
    teacher_paths = [args.teacher_checkpoint, *args.extra_teacher_checkpoint] if args.teacher_checkpoint else []
    for index, path in enumerate(teacher_paths):
        teacher_path = Path(path)
        source = torch.load(teacher_path, map_location='cpu', weights_only=False)
        if source.get('kind') == 'lejepa_backbone':
            raise ValueError('Fine-tune the self-supervised backbone before using it as a classifier teacher')
        validate_teacher(source['config'], config)
        if args.cv_fold is not None and source['epoch'] != source['config']['epochs']:
            raise ValueError('Cross-validation requires the completed fixed-epoch teacher')
        digest = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
        if index == 0:
            config['teacher_sha256'] = digest
            config['teacher_parameters'] = source['config']['parameters']
        else:
            config.setdefault('extra_teacher_sha256', []).append(digest)
            config.setdefault('extra_teacher_parameters', []).append(source['config']['parameters'])
        if source.get('kind') == 'cross_fold_teacher':
            if not args.full_data or args.distill_method != 'kl':
                raise ValueError('Cross-fold teacher groups require full-data KL training')
            from crossfold_teacher import load_crossfold_teacher
            teacher, member_hashes = load_crossfold_teacher(source, ROOT, device, args.temperature)
            config.setdefault('crossfold_teacher_members', {})[str(index)] = member_hashes
        else:
            teacher = build_model(source['config']['model']).to(device, memory_format=torch.channels_last)
            teacher.load_state_dict(source['model'])
        teacher.eval().requires_grad_(False)
        teachers.append(teacher)
        del source
    start_epoch, best, steps, elapsed = 0, 0., 0, 0.
    if args.resume:
        state = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        for key in ('model', 'epochs', 'batch_size', 'lr', 'weight_decay', 'randaugment', 'cutout', 'mixup', 'cutmix', 'split_sha256', 'full_data', 'max_batches'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume configuration changed: {key}')
        for key, default in (('teacher_sha256', None), ('extra_teacher_sha256', []), ('distill_weight', 0.5),
                             ('temperature', 4.0), ('autoaugment', False), ('initialization_sha256', None),
                             ('ricap', 0.), ('random_erasing', 0.), ('warmup_epochs', 5.), ('min_lr', 0.),
                             ('augmix', False), ('audit_holdout', False), ('sam_rho', 0.),
                             ('two_views', False), ('two_view_js', 0.),
                             ('vicreg_weight', None),
                             ('swa', False), ('swa_start', 0.5), ('swa_lr', 0.005),
                             ('supervised_initialization_sha256', None),
                             ('optimizer', 'sgd'), ('distill_method', 'kl'),
                             ('cv_fold', None), ('cv_weights', 'raw')):
            if state['config'].get(key, default) != config.get(key, default):
                raise ValueError(f'Resume configuration changed: {key}')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        if head is not None:
            head.load_state_dict(state['projector'])
        if swa is not None:
            swa.load_state_dict(state['swa_state'])
        start_epoch, best, steps, elapsed = state['epoch'], state['best'], state['steps'], state['elapsed_seconds']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
    else:
        (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        np.savez(out / 'split.npz', train=train_indices, val=val_indices)
        if args.audit_holdout:
            audit = np.setdiff1d(np.arange(len(train_data)), np.concatenate([train_indices, val_indices]))
            np.save(out / 'audit_indices.npy', audit)
        for source in ('train.py', 'model.py', 'shakeshake.py', 'seresnet.py', 'optimizations.py',
                       'starnet.py', 'modern_kd.py', 'modern_sources.json', 'pyramidnet.py', 'crossfold_teacher.py'):
            shutil.copy2(ROOT / source, out / source)
        if head is not None:
            shutil.copy2(ROOT / 'vicreg.py', out / 'vicreg.py')
    print(json.dumps(config), flush=True)
    clock_start = time.monotonic()
    if args.compile_model:
        model.compile(mode='default' if head is not None else 'reduce-overhead')
    if head is not None:
        feature_forward = torch.compile(model.forward_features) if args.compile_model else model.forward_features
        if args.compile_model:
            head.compile()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.monotonic()
        model.train()
        if head is not None:
            head.train()
        loss_sum, count = 0., 0
        ce_sum, auxiliary_sum = 0., 0.
        for batch_idx, (images, labels) in enumerate(train_loader):
            progress = epoch + batch_idx / len(train_loader)
            lr = cosine_lr(progress, args.epochs, args.lr, args.warmup_epochs, args.min_lr, len(train_loader))
            if args.swa and epoch >= int(args.epochs * args.swa_start):
                lr = args.swa_lr
            for group in optimizer.param_groups:
                group['lr'] = lr
            if head is not None:
                auxiliary_views = [value.to(device, non_blocking=True, memory_format=torch.channels_last)
                                   for value in images[1:]]
                images = images[0]
            elif args.two_views:
                # Packing channels makes RICAP share permutations and crop boxes across views.
                images = torch.cat(images, dim=1)
            elif args.augmix:
                images = torch.cat(images, dim=0)
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
            if args.ricap:
                images, targets, weights = ricap_batch(images, labels, args.ricap)
            else:
                images, ya, yb, lam = mix_batch(images, labels, args.mixup, args.cutmix)
                targets, weights = [ya, yb], [lam, 1 - lam]
            if args.two_views:
                images = torch.cat(images.chunk(2, dim=1), dim=0).contiguous(memory_format=torch.channels_last)
            components = {}
            def closure():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images)
                    if args.two_views:
                        strength = args.two_view_js * (min(1., progress / args.warmup_epochs)
                                                      if args.warmup_epochs else 1.)
                        loss = two_view_loss(logits, targets, weights, strength)
                    else:
                        loss = (augmix_loss(logits, labels) if args.augmix else
                                sum(weight * F.cross_entropy(logits, target) for target, weight in zip(targets, weights)))
                    if teachers:
                        with torch.no_grad():
                            teacher_logits = [teacher(images) for teacher in teachers]
                        if args.distill_method == 'rld':
                            loss = loss + min((epoch + 1) / 20., 1.) * rld_loss(
                                logits, teacher_logits[0], labels, args.temperature)
                        else:
                            kd = (ea_loss(logits, teacher_logits[0], args.temperature)
                                  if args.distill_method == 'ea' else
                                  distillation_loss(logits, teacher_logits, args.temperature))
                            loss = (1 - args.distill_weight) * loss + args.distill_weight * kd
                    if head is not None:
                        auxiliary = auxiliary_loss(feature_forward, head, auxiliary_views)
                        components.update(ce=loss.detach(), auxiliary=auxiliary.detach())
                        loss = loss + args.vicreg_weight * auxiliary
                return loss
            if args.sam_rho:
                loss = sam_step(model, optimizer, closure, args.sam_rho)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = closure()
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
            if head is not None:
                ce_sum += components['ce'].item() * len(labels)
                auxiliary_sum += components['auxiliary'].item() * len(labels)
            count += len(labels)
            if args.max_batches and batch_idx + 1 >= args.max_batches:
                break
        metrics = {'epoch': epoch+1, 'train_loss': loss_sum/count, 'lr': lr}
        if head is not None:
            metrics.update(ce_loss=ce_sum/count, vicreg_loss=auxiliary_sum/count,
                           processed_views=3*count)
        if swa is not None and epoch >= int(args.epochs * args.swa_start):
            swa.update_parameters(model)
        if not args.full_data and args.cv_fold is None:
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
        if swa is not None:
            state['swa_state'] = swa.state_dict()
        if head is not None:
            state['projector'] = head.state_dict()
        atomic_save(state, out / 'last.pt')
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
    if swa is not None:
        calibration = DataLoader(Subset(val_data, train_indices), batch_size=256,
                                 num_workers=args.workers, pin_memory=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            update_bn(calibration, swa, device=device)
        state['swa'] = swa.module.state_dict()
        state['swa_state'] = swa.state_dict()
        atomic_save(state, out / 'last.pt')
        atomic_save({'model': swa.module.state_dict(), 'config': config,
                     'epoch': args.epochs, 'weights': 'swa',
                     'averaged_epochs': int(swa.n_averaged)}, out / 'swa.pt')
    if args.cv_fold is not None:
        chosen = ema if args.cv_weights == 'ema' else model
        atomic_save({'model': chosen.state_dict(), 'config': config, 'epoch': args.epochs,
                     'weights': args.cv_weights}, out / 'final.pt')
        result = {'status': 'complete', 'epochs': args.epochs, 'weights': args.cv_weights,
                  'outer_validation_evaluated': False,
                  'elapsed_seconds': elapsed+time.monotonic()-clock_start}
    elif args.full_data:
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
