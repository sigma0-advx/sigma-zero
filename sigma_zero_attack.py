import torch
from adv_lib.utils.losses import difference_of_logits
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor, nn

def sigma_zero(model: nn.Module,
               inputs: Tensor,
               labels: Tensor,
               steps: int = 100,
               lr: float = 1.0,
               sigma: float = 1e-3,
               threshold: float = 0.3,
               verbose: bool = False,
               epsilon_budget=None,
               grad_norm=torch.inf,
               t = 0.01
               ):
    clamp = lambda tensor: tensor.data.add_(inputs.data).clamp_(min=0, max=1).sub_(inputs.data)
    l0_approximation = lambda tensor, sigma: tensor.square().div(tensor.square().add(sigma)).sum(dim=1)
    batch_view = lambda tensor: tensor.view(tensor.shape[0], *[1] * (inputs.ndim - 1))
    normalize = lambda tensor: (
            tensor.flatten(1) / tensor.flatten(1).norm(p=grad_norm, dim=1, keepdim=True).clamp_(min=1e-12)).view(
        tensor.shape)

    device = next(model.parameters()).device
    batch_size, max_size = inputs.shape[0], torch.prod(torch.tensor(inputs.shape[1:]))

    delta = torch.zeros_like(inputs, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 10)
    best_delta = delta.clone()
    best_l0 = torch.full((batch_size,), max_size, device=device)
    th = torch.ones(size=inputs.shape, device=device) * threshold

    for i in range(steps):
        optimizer.zero_grad()

        adv_inputs = inputs + delta  

        # compute loss
        logits = model(adv_inputs)
        dl_loss = difference_of_logits(logits, labels).clip(0) 
        l0_approx = l0_approximation(delta.flatten(1), sigma)
        l0_approx_normalized = l0_approx / delta.data.flatten(1).shape[1]
        # keep best solutions
        predicted_classes = (logits).argmax(1)
        true_l0 = delta.data.flatten(1).ne(0).sum(dim=1)
        is_not_adv = predicted_classes == labels
        is_smaller = true_l0 < best_l0
        is_both = ~is_not_adv & is_smaller
        best_l0 = torch.where(is_both, true_l0.detach(), best_l0)
        best_delta = torch.where(batch_view(is_both), delta.data.clone().detach(), best_delta)
        # update step
        adv_loss = (is_not_adv + dl_loss + l0_approx_normalized).mean()

        if verbose and i % 100 == 0:
            print(th.flatten(1).mean(dim=1), th.flatten(1).mean(dim=1).shape)
            print(is_not_adv)
            print(
                f"iter: {i}, dl loss: {dl_loss.mean().item():.4f}, l0 normalized loss: {l0_approx_normalized.mean().item():.4f}, current median norm: {delta.data.flatten(1).ne(0).sum(dim=1).median()}")

        adv_loss.backward()

        delta.grad.data = normalize(delta.grad.data)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # enforce box constraints
            clamp(delta.data)
            # dynamic thresholding step
            (th)[is_not_adv, :, :, :] -= t * scheduler.get_last_lr()[0]
            (th)[~is_not_adv, :, :, :] += t * scheduler.get_last_lr()[0]
            th.clamp_(0, 1)
            # filter components
            delta.data[delta.data.abs() < th] = 0

    return (inputs + best_delta)
