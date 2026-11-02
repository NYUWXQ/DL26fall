"""Check SE bottlenecks, RICAP area labels, and multi-teacher distillation."""
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from model import build_model
from train import distillation_loss, ricap_batch


def main():
    torch.set_num_threads(2)
    config = {'architecture': 'se', 'depth': 164, 'width': 16}
    model = build_model(config)
    parameters = sum(p.numel() for p in model.parameters())
    assert 2_000_000 < parameters <= 5_000_000
    images = torch.randn(2, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored = build_model(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(images), restored(images))

    labels = torch.arange(8)
    images = labels.float().view(8, 1, 1, 1).expand(8, 3, 32, 32)
    for split in ([0.25, 0.75], [0., 1.], [1., 0.]):
        with patch('train.np.random.beta', return_value=np.array(split)):
            mixed, targets, weights = ricap_batch(images, labels, 0.3)
        expected = sum(weight * target.float() for target, weight in zip(targets, weights))
        assert mixed.shape == images.shape and sum(weights) == 1
        assert torch.allclose(mixed.mean((1, 2, 3)), expected)

    student = torch.randn(4, 10, requires_grad=True)
    teachers = [torch.randn(4, 10, requires_grad=True) for _ in range(2)]
    temperature = 2.
    target = sum((teacher.detach() / temperature).softmax(1) for teacher in teachers) / 2
    expected = F.kl_div((student / temperature).log_softmax(1), target, reduction='batchmean') * temperature**2
    loss = distillation_loss(student, teachers, temperature)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert all(teacher.grad is None for teacher in teachers)
    print(f'PASS: {parameters:,} parameters, residual gradients, checkpoint, RICAP edge patches, teacher probabilities')


if __name__ == '__main__':
    main()
