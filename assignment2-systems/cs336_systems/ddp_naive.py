"""
Minimal implementation of Distributed Data Parallel (DDP) training.

This module implements a naive DDP approach that all-reduces individual parameter
gradients after the backward pass.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class NaiveDDP(nn.Module):
    """
    A naive DDP wrapper that all-reduces parameter gradients after backward pass.

    This implementation:
    1. Broadcasts parameters from rank 0 to all other ranks during initialization
    2. Registers backward hooks on all parameters to all-reduce gradients
    3. Averages gradients across all ranks after backward pass

    Args:
        module: The PyTorch module to wrap with DDP
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # List to store async all-reduce work handles
        self._async_work_handles = []

        # Broadcast parameters from rank 0 to all other ranks
        # This ensures all ranks start with the same model
        if dist.is_initialized():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # Register backward hooks on all parameters that require gradients
        # These hooks will be called after gradients are computed during backward pass
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))

    def _make_hook(self, param: nn.Parameter):
        """
        Create a backward hook for a parameter that will all-reduce its gradient.

        The hook is called after the gradient for this parameter is computed.
        We use all_reduce with SUM to accumulate gradients across all ranks,
        then divide by world_size to get the average.

        Args:
            param: The parameter to create a hook for

        Returns:
            A hook function that all-reduces the parameter's gradient
        """
        def hook(grad):
            if dist.is_initialized():
                # All-reduce the gradient across all ranks using SUM
                # (AVG not supported with Gloo in newer PyTorch versions)
                # We'll divide by world_size after the all-reduce completes
                work = dist.all_reduce(
                    grad.data,
                    op=dist.ReduceOp.SUM,
                    async_op=True
                )
                self._async_work_handles.append((work, param))
            return grad
        return hook

    def finish_gradient_synchronization(self):
        """
        Wait for all async all-reduce operations to complete.

        This should be called after the backward pass is complete, but before
        the optimizer step. It ensures that all gradients have been properly
        averaged across all ranks before we update the parameters.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        for work, param in self._async_work_handles:
            work.wait()
            # Divide by world_size to get the average gradient
            if world_size > 1 and param.grad is not None:
                param.grad.data.div_(world_size)

        # Clear the list for the next iteration
        self._async_work_handles.clear()

    def forward(self, *args, **kwargs):
        """Forward pass - just delegate to the wrapped module."""
        return self.module(*args, **kwargs)