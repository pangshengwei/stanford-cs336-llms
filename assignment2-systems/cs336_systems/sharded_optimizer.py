"""
Optimizer state sharding implementation.

This module implements optimizer state sharding where each rank maintains
optimizer state for only a subset of parameters (approximately 1/world_size).
After each optimizer step, parameters are synchronized across ranks via broadcast.
"""
from typing import Any, Callable, Iterable, Optional, Type

import torch
import torch.distributed as dist
import torch.optim


class ShardedOptimizer(torch.optim.Optimizer):
    """
    Optimizer wrapper that shards optimizer state across ranks.

    Each rank maintains optimizer state (e.g., momentum buffers, Adam variance)
    for only a subset of the model parameters. After each optimizer step, the
    updated parameters are broadcast from each rank to synchronize the model.

    This reduces memory usage by approximately 1/world_size for optimizer states
    while maintaining identical optimization behavior to non-sharded training.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        optimizer_cls: The optimizer class to wrap (e.g., torch.optim.AdamW)
        **kwargs: Additional keyword arguments forwarded to optimizer_cls constructor
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs: Any
    ):
        # Get distributed information
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Store wrapped optimizer info
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # Shard parameters across ranks
        param_groups_input = self._normalize_param_groups(params)
        self.all_params, self.param_to_rank, local_param_groups = self._shard_param_groups(param_groups_input)

        # Initialize parent class with local param groups
        # Use empty defaults dict - wrapped optimizer handles actual defaults
        super().__init__(local_param_groups if local_param_groups else [], {})

        # Create wrapped optimizer with only this rank's parameters
        if local_param_groups:
            self.optimizer = self.optimizer_cls(local_param_groups, **self.optimizer_kwargs)
        else:
            # This rank has no parameters
            self.optimizer = None

    def _normalize_param_groups(self, params: Any) -> list[dict[str, Any]]:
        """
        Convert params argument to normalized list of parameter group dicts.

        Args:
            params: Either an iterable of parameters or a list of param group dicts

        Returns:
            List of parameter group dictionaries
        """
        if isinstance(params, (list, tuple)):
            if len(params) > 0 and isinstance(params[0], dict):
                # Already in param groups format
                return list(params)
            else:
                # List of parameters - wrap in single group
                return [{'params': list(params)}]
        else:
            # Single iterable of parameters
            return [{'params': list(params)}]

    def _shard_param_groups(
        self,
        param_groups: list[dict[str, Any]]
    ) -> tuple[list[torch.nn.Parameter], dict[torch.nn.Parameter, int], list[dict[str, Any]]]:
        """
        Assign parameters to ranks and create local parameter groups.

        Parameters are assigned to ranks in round-robin fashion based on their
        order in the param_groups. Each rank gets approximately 1/world_size
        of the parameters.

        Args:
            param_groups: List of parameter group dictionaries

        Returns:
            Tuple of (all_params, param_to_rank, local_param_groups):
                - all_params: Complete list of all parameters across all ranks
                - param_to_rank: Mapping from parameter to its assigned rank
                - local_param_groups: Parameter groups containing only this rank's params
        """
        all_params = []
        param_to_rank = {}
        local_param_groups = []

        for group in param_groups:
            group_params = list(group['params'])
            local_params = []

            for param in group_params:
                if param not in param_to_rank:
                    # Assign parameter to rank via round-robin
                    param_idx = len(all_params)
                    assigned_rank = param_idx % self.world_size
                    all_params.append(param)
                    param_to_rank[param] = assigned_rank

                    # Keep parameter if it belongs to this rank
                    if assigned_rank == self.rank:
                        local_params.append(param)

            # Create local param group if this rank has params from this group
            if local_params:
                local_group = {k: v for k, v in group.items() if k != 'params'}
                local_group['params'] = local_params
                local_param_groups.append(local_group)

        return all_params, param_to_rank, local_param_groups

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        Add a parameter group to the optimizer.

        This is called during construction by the parent class and may also be
        called during training (e.g., for gradually unfreezing layers). Parameters
        are assigned to ranks and only those belonging to this rank are added to
        the wrapped optimizer.

        Args:
            param_group: Dictionary defining the parameter group
        """
        params = list(param_group['params'])
        local_params = []

        for param in params:
            if param not in self.param_to_rank:
                # New parameter - assign to a rank
                param_idx = len(self.all_params)
                assigned_rank = param_idx % self.world_size
                self.all_params.append(param)
                self.param_to_rank[param] = assigned_rank

                if assigned_rank == self.rank:
                    local_params.append(param)

        # Add to wrapped optimizer if this rank has any params from this group
        if local_params:
            local_group = {k: v for k, v in param_group.items() if k != 'params'}
            local_group['params'] = local_params

            # Add to parent class
            super().add_param_group(local_group)

            # Add to wrapped optimizer
            if self.optimizer is None:
                # Create wrapped optimizer with this first param group
                self.optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
            else:
                self.optimizer.add_param_group(local_group)

    def step(self, closure: Optional[Callable] = None, **kwargs: Any) -> Optional[torch.Tensor]:
        """
        Perform a single optimization step and synchronize parameters.

        This calls the wrapped optimizer's step() method to update this rank's
        shard of parameters, then broadcasts updated parameters from each rank
        to all other ranks to keep the model synchronized.

        Args:
            closure: Optional closure to reevaluate the model and return the loss
            **kwargs: Additional keyword arguments forwarded to wrapped optimizer's step

        Returns:
            The loss value if closure is provided, otherwise None
        """
        # Update this rank's shard of parameters
        loss = None
        if self.optimizer is not None:
            loss = self.optimizer.step(closure=closure, **kwargs)

        # Synchronize all parameters across ranks
        # Each rank broadcasts its updated parameters to all other ranks
        if dist.is_initialized():
            for param in self.all_params:
                owner_rank = self.param_to_rank[param]
                dist.broadcast(param.data, src=owner_rank)

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Zero gradients for this rank's parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zero
        """
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """
        Return the state of the optimizer as a dict.

        Only includes state for this rank's shard of parameters.

        Returns:
            Dictionary containing optimizer state
        """
        if self.optimizer is not None:
            return self.optimizer.state_dict()
        return {'state': {}, 'param_groups': []}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the optimizer state.

        Args:
            state_dict: Optimizer state dictionary
        """
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict)
