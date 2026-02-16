"""
DDP implementation with overlapped computation and communication.

This module implements DDP that overlaps backward pass computation with gradient
communication by asynchronously all-reducing parameter gradients as they become ready.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class DDPWithOverlap(nn.Module):
    """
    DDP wrapper that overlaps backward computation with gradient communication.

    This implementation:
    1. Broadcasts parameters from rank 0 to all other ranks during initialization
    2. Registers post-accumulate gradient hooks to all-reduce gradients asynchronously
       as soon as they are ready during the backward pass
    3. Allows backward computation to continue while gradients are being communicated

    The key benefit is reduced training iteration time by overlapping computation
    and communication, rather than waiting for all gradients before starting communication.

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

        # Register post-accumulate gradient hooks on all parameters
        # These hooks are called after gradients are accumulated for each parameter
        # Using post_accumulate_grad_hook ensures the gradient is fully ready
        for param in self.module.parameters():
            if param.requires_grad:
                # Use the newer post_accumulate_grad_hook API which is called
                # after gradient accumulation is complete
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param: nn.Parameter):
        """
        Create a post-accumulate gradient hook for a parameter.

        The hook is called after the gradient for this parameter has been accumulated.
        It asynchronously all-reduces the gradient across all ranks, allowing the
        backward pass to continue computing other gradients while communication happens.

        Args:
            param: The parameter to create a hook for

        Returns:
            A hook function that asynchronously all-reduces the parameter's gradient
        """
        def hook(param_arg):
            # param_arg is the parameter passed by the hook
            # We use the captured param from the closure for clarity
            if dist.is_initialized() and param.grad is not None:
                # All-reduce the gradient across all ranks asynchronously
                # Using SUM operation (AVG not supported in Gloo backend in newer PyTorch)
                # We'll divide by world_size after the all-reduce completes
                # async_op=True allows the backward pass to continue while
                # the all-reduce operation is queued and executed
                work = dist.all_reduce(
                    param.grad.data,
                    op=dist.ReduceOp.SUM,
                    async_op=True
                )
                self._async_work_handles.append((work, param))

        return hook

    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous gradient communication to complete.

        This should be called after the backward pass is complete, but before
        the optimizer step. It ensures that all asynchronous all-reduce operations
        have been queued on the GPU, so the gradients are ready to be used by
        the optimizer.

        Note: For GPU operations, this ensures operations are queued, not necessarily
        completed, since CUDA operations are asynchronous. Subsequent operations
        that depend on the gradients will automatically wait as needed.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        for work, param in self._async_work_handles:
            work.wait()
            # Divide by world_size to get the average gradient
            # (we used SUM operation instead of AVG for Gloo compatibility)
            if world_size > 1:
                param.grad.data.div_(world_size)

        # Clear the list for the next iteration
        self._async_work_handles.clear()

    def forward(self, *args, **kwargs):
        """
        Forward pass - delegates to the wrapped module.

        Args:
            *args: Positional arguments to pass to the module
            **kwargs: Keyword arguments to pass to the module

        Returns:
            The output of the wrapped module's forward pass
        """
        return self.module(*args, **kwargs)