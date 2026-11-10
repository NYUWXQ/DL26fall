"""Weight perturbation for SAM; averaging uses torch.optim.swa_utils."""
import torch


def sam_step(model, optimizer, closure, rho):
    optimizer.zero_grad(set_to_none=True)
    loss = closure()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite SAM loss')
    loss.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    gradients = [p.grad for p in parameters]
    norm = torch.stack(torch._foreach_norm(gradients)).norm()
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite SAM gradient norm')
    original = [p.detach().clone() for p in parameters]
    batchnorm = [(m, m.track_running_stats) for m in model.modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    try:
        with torch.no_grad():
            torch._foreach_add_(parameters, torch._foreach_mul(gradients, rho / (norm + 1e-12)))
        # The perturbed pass uses batch statistics without counting this batch twice.
        for module, _ in batchnorm:
            module.track_running_stats = False
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss = closure()
        if not torch.isfinite(perturbed_loss):
            raise RuntimeError('Nonfinite perturbed SAM loss')
        perturbed_loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not torch.stack(torch._foreach_norm(gradients)).isfinite().all():
            raise RuntimeError('Nonfinite perturbed SAM gradients')
    finally:
        with torch.no_grad():
            torch._foreach_copy_(parameters, original)
        for module, tracking in batchnorm:
            module.track_running_stats = tracking
    optimizer.step()
    return loss.detach()
