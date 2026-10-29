"""Check models, augmentation, checkpointing, and prediction inputs."""
import numpy as np
import torch
from model import WideResNet
from train import atomic_save, mix_batch, split_indices
from pathlib import Path
from tempfile import TemporaryDirectory
from run import select_result
from refine import probabilities
from predict import CompetitionImages
from PIL import Image


def main():
    torch.set_num_threads(2)
    for depth, width in ((28, 58), (16, 84)):
        model = WideResNet(depth, width)
        assert sum(p.numel() for p in model.parameters()) <= 5_000_000
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (2, 10)
        logits.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    try:
        WideResNet(28, 64)
        raise AssertionError('Oversized model was accepted')
    except ValueError:
        pass
    targets = np.repeat(np.arange(10), 5000)
    train, val = split_indices(targets)
    assert len(train) == 45000 and len(val) == 5000
    assert len(set(train) & set(val)) == 0
    assert len(set(train) | set(val)) == 50000
    assert np.all(np.bincount(targets[val]) == 500)
    images = torch.stack([torch.full((3, 32, 32), float(i)) for i in range(16)])
    labels = torch.arange(16)
    for mixup, cutmix in ((0, 0), (.2, 0), (0, 1), (.2, 1)):
        mixed, ya, yb, lam = mix_batch(images.clone(), labels, mixup, cutmix)
        assert mixed.shape == images.shape and 0 <= lam <= 1
        expected = ya.float() * lam + yb.float() * (1 - lam)
        assert torch.allclose(mixed.mean((1, 2, 3)), expected, atol=1e-5)
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'checkpoint.pt'
        atomic_save({'x': torch.ones(2)}, path)
        assert torch.equal(torch.load(path, weights_only=True)['x'], torch.ones(2))
        assert not path.with_suffix('.pt.tmp').exists()
    def result(plain, flip):
        return {'status': 'complete', 'plain': {'accuracy': plain, 'total': 5000},
                'flip': {'accuracy': flip, 'total': 5000}}
    selected = select_result({'a': result(.95, .96), 'b': result(.97, .965)})
    assert selected == {'name': 'b', 'inference': 'plain', 'validation_accuracy': .97}
    assert select_result({'a': result(.95, .96)})['inference'] == 'flip'
    constant = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 10)).eval()
    flat_images = torch.ones(2, 3, 32, 32)
    expected = constant(flat_images).softmax(1)
    for mode in ('plain', 'flip', 'translate'):
        assert torch.allclose(probabilities(constant, flat_images, mode), expected)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'test_images').mkdir()
        for name in ('a.png', 'b.png'):
            Image.new('RGB', (32, 32)).save(root / 'test_images' / name)
        (root/'sample_submission.csv').write_text('id,label\n2,0\n1,0\n')
        (root/'test.csv').write_text('id,filename\n1,a.png\n2,b.png\n')
        data = CompetitionImages(root)
        assert data.ids == ['2', '1'] and data.paths[0].name == 'b.png'
        assert data[0].shape == (3, 32, 32)
        (root/'test.csv').write_text('id,filename\n1,a.png\n1,b.png\n')
        try:
            CompetitionImages(root)
            raise AssertionError('Duplicate IDs accepted')
        except ValueError:
            pass
        (root/'test.csv').write_text('id,filename\n1,../../outside.png\n2,b.png\n')
        try:
            CompetitionImages(root)
            raise AssertionError('Image path traversal accepted')
        except ValueError:
            pass
    print('PASS: model budget, gradients, split, mixing, checkpoints, selection, TTA, submission IDs and paths')


if __name__ == '__main__':
    main()
