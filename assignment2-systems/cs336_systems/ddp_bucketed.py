"""
Bucketed DDP implementation with gradient communication overlap.

This module implements DDP that overlaps backward computation with gradient
communication by all-reducing buckets of parameters as they become ready.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


class BucketedDDP(nn.Module):
    """
    DDP wrapper that uses gradient bucketing for efficient communication.

    This implementation combines the benefits of:
    1. Overlap: Asynchronous all-reduce during backward pass
    2. Batching: Fewer communication calls by grouping parameters

    Parameters are grouped into buckets (default ~25MB). When all gradients
    in a bucket are ready, the bucket is all-reduced asynchronously. This
    reduces communication overhead while maintaining computation-communication overlap.

    Args:
        module: The PyTorch module to wrap with DDP
        bucket_size_mb: Maximum size of each bucket in megabytes
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb

        # Broadcast parameters from rank 0 to all other ranks
        if dist.is_initialized():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # Create buckets of parameters in reverse order
        # (gradients become ready in reverse order during backward)
        self.buckets = self._create_buckets()

        # Track which parameters have received gradients
        self.param_ready = {}

        # Store async work handles for all-reduce operations
        self._async_work_handles = []

        # Register hooks on all parameters
        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket['params']:
                if param.requires_grad:
                    # Mark parameter as not ready initially
                    self.param_ready[param] = False
                    # Register hook to track when gradient is ready
                    param.register_post_accumulate_grad_hook(
                        self._make_hook(param, bucket_idx)
                    )

    def _create_buckets(self) -> List[dict]:
        """
        Create buckets of parameters in reverse order.

        Groups parameters into buckets of approximately bucket_size_mb MB.
        Uses reverse order because gradients become ready in that order.

        Returns:
            List of bucket dictionaries, each containing:
                - 'params': list of parameters in the bucket
                - 'size_bytes': total size of the bucket in bytes
        """
        buckets = []
        current_bucket = []
        current_size = 0
        bucket_size_bytes = self.bucket_size_mb * 1024 * 1024

        # Iterate parameters in reverse order
        params_list = list(self.module.parameters())
        for param in reversed(params_list):
            if not param.requires_grad:
                continue

            param_size = param.numel() * param.element_size()

            # If adding this parameter would exceed bucket size and we have params,
            # start a new bucket (unless this is the first param in the bucket)
            if current_size + param_size > bucket_size_bytes and current_bucket:
                buckets.append({
                    'params': current_bucket,
                    'size_bytes': current_size
                })
                current_bucket = []
                current_size = 0

            current_bucket.append(param)
            current_size += param_size

        # Add the last bucket if it has parameters
        if current_bucket:
            buckets.append({
                'params': current_bucket,
                'size_bytes': current_size
            })

        return buckets

    def _make_hook(self, param: nn.Parameter, bucket_idx: int):
        """
        Create a post-accumulate gradient hook for a parameter.

        The hook marks the parameter as ready and checks if all parameters
        in its bucket are ready. If so, triggers async all-reduce for the bucket.

        Args:
            param: The parameter to create a hook for
            bucket_idx: Index of the bucket this parameter belongs to

        Returns:
            A hook function that triggers bucket all-reduce when ready
        """
        def hook(param_arg):
            if not dist.is_initialized():
                return

            # Mark this parameter as ready
            self.param_ready[param] = True

            # Check if all parameters in this bucket are ready
            bucket = self.buckets[bucket_idx]
            if all(self.param_ready.get(p, False) for p in bucket['params']):
                # All parameters in bucket are ready - all-reduce the bucket
                self._all_reduce_bucket(bucket_idx)

        return hook

    def _all_reduce_bucket(self, bucket_idx: int):
        """
        All-reduce a bucket of gradients asynchronously.

        Flattens all gradients in the bucket, performs async all-reduce,
        and stores the work handle for later synchronization.

        Args:
            bucket_idx: Index of the bucket to all-reduce
        """
        bucket = self.buckets[bucket_idx]

        # Collect gradients from all parameters in the bucket
        grads = []
        for param in bucket['params']:
            if param.grad is not None:
                grads.append(param.grad.data)

        if not grads:
            return

        # Flatten gradients into a single tensor
        flat_grads = torch._utils._flatten_dense_tensors(grads)

        # Async all-reduce using SUM (will divide by world_size later)
        work = dist.all_reduce(
            flat_grads,
            op=dist.ReduceOp.SUM,
            async_op=True
        )

        # Store work handle along with bucket info for later processing
        self._async_work_handles.append({
            'work': work,
            'flat_grads': flat_grads,
            'grads': grads,
            'bucket_idx': bucket_idx
        })

    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous bucket all-reduces to complete.

        This should be called after the backward pass, before the optimizer step.
        It waits for all bucket all-reduces, averages the gradients, and copies
        them back to the parameters.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        for handle_info in self._async_work_handles:
            # Wait for the all-reduce to complete
            handle_info['work'].wait()

            # Divide by world_size to get average
            if world_size > 1:
                handle_info['flat_grads'].div_(world_size)

            # Unflatten and copy back to parameters
            unflat_grads = torch._utils._unflatten_dense_tensors(
                handle_info['flat_grads'],
                handle_info['grads']
            )

            for param_grad, synced_grad in zip(handle_info['grads'], unflat_grads):
                param_grad.copy_(synced_grad)

        # Clear for next iteration
        self._async_work_handles.clear()

        # Reset all parameters to not ready for next iteration
        for param in self.param_ready:
            self.param_ready[param] = False

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
