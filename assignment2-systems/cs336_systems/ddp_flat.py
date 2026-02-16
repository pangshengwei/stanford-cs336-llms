"""
Flattened gradient DDP implementation.

This module implements DDP with a single batched all-reduce call by flattening
all parameter gradients into a single tensor before communication.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class FlattenedDDP(nn.Module):
    """
    DDP wrapper that batches all gradient all-reduces into a single communication call.

    Instead of all-reducing each parameter gradient individually, this implementation:
    1. Flattens all parameter gradients into a single contiguous tensor
    2. All-reduces the flattened tensor in one communication call
    3. Unflattens and copies the results back to individual parameter gradients

    This reduces communication overhead by:
    - Reducing the number of kernel launches
    - Better utilizing network bandwidth with larger messages
    - Reducing latency overhead (one message vs. many small messages)

    Args:
        module: The PyTorch module to wrap with DDP
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # Broadcast parameters from rank 0 to all other ranks
        # This ensures all ranks start with the same model
        if dist.is_initialized():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # Store parameters that require gradients for efficient iteration
        self._grad_params = [p for p in self.module.parameters() if p.requires_grad]

    def finish_gradient_synchronization(self):
        """
        Flatten all gradients, all-reduce them in a single call, then unflatten.

        This should be called after the backward pass is complete, but before
        the optimizer step. It ensures that all gradients have been properly
        averaged across all ranks before we update the parameters.
        """
        if not dist.is_initialized():
            return

        # Collect all gradients that are not None
        grads = []
        for param in self._grad_params:
            if param.grad is not None:
                grads.append(param.grad.data)

        if not grads:
            return

        # Flatten all gradients into a single contiguous tensor
        # This uses PyTorch's internal utility for efficient flattening
        flat_grads = torch._utils._flatten_dense_tensors(grads)

        # All-reduce the flattened gradient tensor using SUM
        # (AVG not supported with Gloo in newer PyTorch versions)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)

        # Divide by world_size to get the average
        world_size = dist.get_world_size()
        flat_grads.div_(world_size)

        # Unflatten the all-reduced gradients back to individual parameter shapes
        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)

        # Copy the all-reduced gradients back to the parameters
        for param_grad, synced_grad in zip(grads, unflat_grads):
            param_grad.copy_(synced_grad)

    def forward(self, *args, **kwargs):
        """Forward pass - just delegate to the wrapped module."""
        return self.module(*args, **kwargs)