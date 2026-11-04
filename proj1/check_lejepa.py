"""CPU checks for LeJEPA losses, backbone transfer, and data-boundary guards."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from lejepa import LeJEPAEncoder, MultiView, SIGReg, lejepa_loss
from model import build_model
from train import ROOT, initialize_backbone
import pretrain_lejepa


def main():
    torch.set_num_threads(2)
    regularizer = SIGReg(directions=64)
    gaussian = torch.randn(2, 2048, 16)
    collapsed = torch.zeros_like(gaussian)
    assert regularizer(gaussian) < regularizer(collapsed) / 5
    repeated = torch.randn(1, 16, 8).expand(4, -1, -1)
    _, invariance, _ = lejepa_loss(repeated, regularizer, 0.02)
    assert invariance == 0
    image = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    views = MultiView(4)(image)
    assert views.shape == (4, 3, 32, 32) and torch.isfinite(views).all()
    model_config = {'depth': 10, 'width': 8, 'dropout': 0.0}
    encoder = LeJEPAEncoder(model_config, hidden=32, projection_dim=8)
    optimizer = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad), lr=0.001)
    projections = encoder(torch.randn(4, 2, 3, 32, 32))
    assert projections.shape == (2, 4, 8)
    loss, _, _ = lejepa_loss(projections, SIGReg(directions=16), 0.02)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters() if p.requires_grad)
    assert encoder.backbone.fc.weight.grad is None
    optimizer.step()
    config = {'model': model_config, 'epochs': 1, 'full_data': False, 'max_batches': 0,
              'training_data': 'CIFAR-10 official train split', 'split_sha256': 'test-split'}
    source = {'kind': 'lejepa_backbone', 'status': 'complete', 'config': config,
              'epoch': 1, 'model': encoder.backbone.state_dict()}
    target = build_model(model_config)
    initial_head = target.fc.weight.detach().clone()
    initialize_backbone(target, source, config)
    assert torch.equal(target.fc.weight, initial_head) and target.fc.weight.requires_grad
    assert all(torch.equal(value, source['model'][key]) for key, value in target.state_dict().items() if not key.startswith('fc.'))
    target.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        old_forward = target.fc(torch.relu(target.bn(target.blocks(target.stem(images)))).mean((2, 3)))
        assert torch.equal(target(images), old_forward)
    target.train()
    torch.nn.functional.cross_entropy(target(images), torch.tensor([0, 1])).backward()
    assert target.fc.weight.grad is not None and torch.isfinite(target.fc.weight.grad).all()
    assert target.stem.weight.grad is not None and torch.isfinite(target.stem.weight.grad).all()
    for change in ({'full_data': True}, {'split_sha256': 'different'}, {'max_batches': 1}):
        bad = copy.deepcopy(source)
        bad['config'].update(change)
        try:
            initialize_backbone(target, bad, config)
            raise AssertionError('Invalid pretraining source accepted')
        except ValueError:
            pass
    for change in ({'epoch': 0}, {'status': 'running'}):
        try:
            initialize_backbone(target, source | change, config)
            raise AssertionError('Incomplete pretraining accepted')
        except ValueError:
            pass
    with TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / 'backbone.pt'
        torch.save(source, checkpoint)
        restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        initialize_backbone(target, restored, config)
        result = subprocess.run([sys.executable, str(ROOT / 'predict.py'), str(checkpoint),
                                 '--out', str(root / 'submission.csv'), '--device', 'cpu'],
                                text=True, capture_output=True)
        assert result.returncode != 0 and 'Fine-tune' in result.stderr
        assert not (root / 'submission.csv').exists()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'licenses').mkdir()
        for name in ('pretrain_lejepa.py', 'lejepa.py', 'train.py', 'model.py', 'shakeshake.py', 'licenses/lejepa.txt'):
            (root / name).write_bytes((ROOT / name).read_bytes())
        settings = {'epochs': 2, 'batch_size': 4, 'views': 2, 'hidden': 16,
                    'projection_dim': 4, 'sigreg_weight': 0.02, 'directions': 4,
                    'lr': 0.001, 'weight_decay': 0.05}
        config_path = root / 'config.json'
        config_path.write_text(json.dumps({'model': model_config, 'pretrain': settings}))
        class SyntheticImages:
            def __init__(self, path, train, transform, download):
                assert train and not download
                self.transform = transform
                self.targets = list(range(10))
                self.images = np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)

            def __len__(self):
                return 10

            def __getitem__(self, index):
                return self.transform(Image.fromarray(self.images[index])), self.targets[index]

        class Interrupted(Exception):
            pass

        save = pretrain_lejepa.atomic_save
        def interrupt_after_save(value, path):
            save(value, path)
            if path.name == 'last.pt' and value['epoch'] == 1:
                raise Interrupted()

        argv = ['pretrain_lejepa.py', '--name', 'check', '--config', str(config_path), '--device', 'cpu', '--workers', '0']
        with patch.object(pretrain_lejepa, 'ROOT', root), \
             patch.object(pretrain_lejepa, 'CIFAR10', SyntheticImages), \
             patch.object(pretrain_lejepa, 'split_indices', return_value=(np.arange(8), np.arange(8, 10))), \
             patch.object(pretrain_lejepa.subprocess, 'check_output', return_value='cpu-self-check\n'), \
             redirect_stdout(io.StringIO()):
            with patch.object(sys, 'argv', argv), patch.object(pretrain_lejepa, 'atomic_save', interrupt_after_save):
                try:
                    pretrain_lejepa.main()
                    raise AssertionError('Expected simulated interruption')
                except Interrupted:
                    pass
            assert not (root / 'runs/check/backbone.pt').exists()
            with patch.object(sys, 'argv', argv + ['--resume']):
                pretrain_lejepa.main()
        restored = torch.load(root / 'runs/check/last.pt', map_location='cpu', weights_only=False)
        assert restored['epoch'] == 2
        assert all(state['step'].item() == 4 for state in restored['optimizer']['state'].values())
        backbone = torch.load(root / 'runs/check/backbone.pt', map_location='cpu', weights_only=False)
        assert backbone['status'] == 'complete' and backbone['epoch'] == 2
        assert all(not key.startswith('projector.') for key in backbone['model'])
    print('PASS: SIGReg, multiview gradients, fresh classification head, split guards, checkpoint roundtrip, and prediction guard')
    print('PASS: CPU synthetic-data interruption and resume; training-only projector excluded from backbone')


if __name__ == '__main__':
    main()
