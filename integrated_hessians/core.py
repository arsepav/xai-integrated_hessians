import torch
from torch import nn

def _integral_approx(fun, steps):
    """
    Simple Riemann approximation of the integral from 0 to 1.
    fun: function mapping alpha to Tensor
    steps: number of steps
    """
    alphas = torch.linspace(0.0, 1.0, steps).to(fun(0).device)
    vals = [fun(alpha) for alpha in alphas]
    return torch.stack(vals, dim=0).mean(dim=0)

class IntegratedHessians:
    def __init__(self, model: nn.Module, baseline: torch.Tensor = None, steps: int = 50):
        self.model = model.eval()
        self.baseline = baseline
        self.steps = steps

    def _make_baseline(self, x: torch.Tensor):
        if self.baseline is not None:
            return self.baseline.repeat(x.size(0), 1)
        return torch.zeros_like(x)

    def pairwise_interactions(self, x: torch.Tensor) -> torch.Tensor:
        bsz, d = x.size()
        base = self._make_baseline(x)
        device = x.device

        def f_at(alpha_beta):
            pt = base + alpha_beta * (x - base)
            pt.requires_grad_(True)
            out = self.model(pt)[0] if isinstance(self.model(pt), tuple) else self.model(pt)
            if out.ndim > 1:
                out = out[:, 1]
            grads = torch.autograd.grad(out.sum(), pt, create_graph=True)[0]
            H = []
            for i in range(d):
                grad_i = grads[:, i]
                H_i = torch.autograd.grad(grad_i.sum(), pt, retain_graph=True)[0]
                H.append(H_i)
            H = torch.stack(H, dim=1)
            return H * (x - base).view(bsz, 1, d)

        H_int = _integral_approx(lambda ab: _integral_approx(lambda a: ab * f_at(a * ab), self.steps), self.steps)
        diffs = (x - base).view(bsz, d, 1) * (x - base).view(bsz, 1, d)
        return H_int * diffs