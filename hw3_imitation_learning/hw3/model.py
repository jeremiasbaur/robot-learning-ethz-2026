"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor,) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError

# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int, action_dim: int, chunk_size: int
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size)
        self.hidden_dim = 256
        self.layer1 = torch.nn.Linear(self.state_dim, self.hidden_dim)
        self.relu = torch.nn.ReLU()
        self.ln1 = torch.nn.LayerNorm(self.hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.layer2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = torch.nn.LayerNorm(self.hidden_dim)
        self.layer3 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln3 = torch.nn.LayerNorm(self.hidden_dim)
        self.layer_out = torch.nn.Linear(self.hidden_dim, self.action_dim*self.chunk_size)

        

    def forward(
        self, state: torch.Tensor
    ) -> torch.Tensor:
        x = self.ln1(self.dropout(self.layer1(state)))
        x = self.relu(x)
        x = self.relu(self.ln2(self.dropout(self.layer2(x))))
        x = self.relu(self.ln3(self.dropout(self.layer3(x))))
        out = x = self.layer_out(x)

        return out.view(-1, self.chunk_size, self.action_dim)
        

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self.sample_actions(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def compute_loss(
        self,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        raise NotImplementedError


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
