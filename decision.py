import torch
from torch.autograd import Function
"""
Taken from Cheng et al. (2024) - https://github.com/Yu-AngCheng/RTify
Adapted for negative evidence
"""

import torch

class SimpleThresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trajectory, dsdt_trajectory, x, b, dt, tnd=0.2):
        mask = trajectory >= 0
        decision_time = mask.float().argmax(dim=1).int()
        decision_time[mask.sum(dim=1) == 0] = torch.tensor(trajectory.shape[1] - 1, dtype=torch.int)

        evidences = torch.gather(x, dim=1, index=decision_time.unsqueeze(-1).to(dtype=torch.int64)).squeeze(-1)
        times = decision_time * dt + tnd
        signs = torch.sign(evidences)
        times = times * signs

        ctx.save_for_backward(dsdt_trajectory, decision_time, trajectory, signs)
        return times, evidences, decision_time

    @staticmethod
    def backward(ctx, grad_rt, grad_evidences, grad_decision_time):
        dsdt_trajectory, decision_times, trajectory, signs = ctx.saved_tensors
        decision_times = decision_times.squeeze()
        mask = trajectory >= 0
        idx1 = (mask.sum(dim=1) == 0).long()
        idx2 = dsdt_trajectory[torch.arange(dsdt_trajectory.size(0)), decision_times.long()] < 0
        idx = torch.logical_and(idx1.bool(), idx2.bool()).squeeze()
        grads = torch.zeros_like(dsdt_trajectory)
        batch_indices = torch.arange(decision_times.size(0)).to(decision_times.device)

        grads[batch_indices, decision_times.long()] = -1.0 / (dsdt_trajectory[
            batch_indices, decision_times.long()] + 1e-6)
        grads[batch_indices[idx], decision_times[idx].long()] = 1e-6 / (
            trajectory[batch_indices[idx], decision_times[idx].long()] + 1e-6
        )
        grads = grads * grad_rt.unsqueeze(1).expand_as(grads) * signs.unsqueeze(-1)
        return grads, None, None, None, None, None