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
    query_mask = torch.full((batch_size,), True, device=device)
    best_l0 = torch.full((batch_size,), max_size, device=device)
    is_adv_below_eps = torch.full((batch_size,), False, device=device)
    th = torch.ones(size=inputs.shape, device=device) * threshold
    
    for i in range(steps):
        optimizer.zero_grad()

        active_delta = delta[query_mask].clone().detach().requires_grad_(True)
        active_inputs = inputs[query_mask]
        active_labels = labels[query_mask]  
        
        adv_inputs = active_inputs + active_delta

        # compute loss
        logits = model(adv_inputs)
        dl_loss = difference_of_logits(logits, active_labels).clip(0)
        l0_approx = l0_approximation(active_delta.flatten(1), sigma)
        l0_approx_normalized = l0_approx / active_delta.data.flatten(1).shape[1]

        # keep best solutions
        predicted_classes = (logits).argmax(1)
        true_l0 = active_delta.data.flatten(1).ne(0).sum(dim=1)
        is_not_adv = predicted_classes == active_labels
        is_smaller = true_l0 < best_l0[query_mask]
        is_both = ~is_not_adv & is_smaller
        best_l0[query_mask] = torch.where(is_both, true_l0.detach(), best_l0[query_mask])
        best_delta[query_mask] = torch.where(batch_view(is_both), active_delta.data.clone().detach(), best_delta[query_mask]) 
        is_adv_below_eps = best_l0 <= epsilon_budget if epsilon_budget is not None else is_adv_below_eps 
    
        # update step
        adv_loss = (is_not_adv + dl_loss + l0_approx_normalized).mean()

        if verbose and i % 100 == 0:
            print(th.flatten(1).mean(dim=1), th.flatten(1).mean(dim=1).shape)
            print(is_not_adv)
            print(
                f"iter: {i}, dl loss: {dl_loss.mean().item():.4f}, l0 normalized loss: {l0_approx_normalized.mean().item():.4f}, current median norm: {delta.data.flatten(1).ne(0).sum(dim=1).median()}")

        adv_loss.backward()

        if delta.grad is None:
            delta.grad = torch.zeros_like(delta, device=device)
        # Copy gradients from active_delta.grad to delta.grad at the masked positions
        delta.grad[query_mask] += active_delta.grad
 
        delta.grad.data = normalize(delta.grad.data)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # enforce box constraints
            clamp(delta.data)
            # dynamic thresholding step
            th_active = th[query_mask]
        
            th_active[is_not_adv, :, :, :] -= t * scheduler.get_last_lr()[0]
            th_active[~is_not_adv, :, :, :] += t * scheduler.get_last_lr()[0]
            th[query_mask] = th_active
            th.clamp_(0, 1)
            # filter components
            delta.data[delta.data.abs() < th] = 0
            # update active set
            query_mask[is_adv_below_eps] = False
            if not any(query_mask):
                break

    return (inputs + best_delta)
