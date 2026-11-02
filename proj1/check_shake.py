"""Check Shake-Shake mixing, checkpoints, and CPU prediction."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image
import torch

from model import WideResNet, build_model
from shakeshake import Shake
from train import ROOT


def main():
    torch.set_num_threads(2)
    left, right = torch.ones(2, 3, 4, 4, requires_grad=True), torch.zeros(2, 3, 4, 4, requires_grad=True)
    with patch('shakeshake.torch.rand', side_effect=[torch.full((2, 1, 1, 1), 0.2), torch.full((2, 1, 1, 1), 0.8)]):
        output = Shake.apply(left, right, True)
        assert torch.allclose(output, torch.full_like(output, 0.2))
        output.sum().backward()
    assert torch.allclose(left.grad, torch.full_like(left, 0.8))
    assert torch.allclose(right.grad, torch.full_like(right, 0.2))
    assert torch.equal(Shake.apply(left, right, False), (left + right) / 2)
    for width in (32, 40):
        config = {'architecture': 'shake', 'depth': 26, 'width': width}
        model = build_model(config)
        count = sum(parameter.numel() for parameter in model.parameters())
        assert count <= 5_000_000
        images = torch.randn(2, 3, 32, 32)
        model(images).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            assert torch.equal(model(images), model(images))
        copy = build_model(config)
        copy.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(build_model({'depth': 16, 'width': 84}), WideResNet)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'test_images').mkdir()
        for name in ('a.png', 'b.png'):
            Image.new('RGB', (32, 32), 'blue').save(root / 'test_images' / name)
        (root / 'sample_submission.csv').write_text('id,label\n2,0\n1,0\n')
        (root / 'test.csv').write_text('id,filename\n1,a.png\n2,b.png\n')
        checkpoint = root / 'model.pt'
        torch.save({'config': {'model': config, 'parameters': count}, 'model': model.state_dict(), 'inference': 'flip'}, checkpoint)
        output = root / 'submission.csv'
        subprocess.run([sys.executable, str(ROOT / 'predict.py'), str(checkpoint), '--data', str(root),
                        '--out', str(output), '--device', 'cpu'], check=True, capture_output=True)
        with output.open() as handle:
            rows = list(csv.DictReader(handle))
        assert [row['id'] for row in rows] == ['2', '1']
        assert all(0 <= int(row['label']) < 10 for row in rows)
        assert json.loads(output.with_suffix('.json').read_text())['parameters'] == count
    print('PASS: independent forward/backward mixing, gradients, budget, checkpoint loading, and CSV prediction')


if __name__ == '__main__':
    main()
