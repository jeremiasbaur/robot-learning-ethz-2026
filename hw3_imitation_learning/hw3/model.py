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
        out = self.layer_out(x)

        return out.view(-1, self.chunk_size, self.action_dim)
        

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self.sample_actions(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self, state_dim: int, action_dim: int, chunk_size: int
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size)
        self.d_model = 64

        self.ee_encoder = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model)
        ) # gripper: 1 + ee: 3

        self.cube_encoder = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model)
        )
    
        self.bin_encoder = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model)
        ) # position of container
        
        self.mlp_policy = nn.Sequential(
            nn.Linear(3 * self.d_model, 4*self.d_model),
            nn.ReLU(),
            nn.LayerNorm(4*self.d_model),
            nn.Dropout(p=0.1),
            nn.Linear(4*self.d_model, 4*self.d_model),
            nn.ReLU(),
            nn.LayerNorm(4*self.d_model),
            nn.Dropout(p=0.1),
            nn.Linear(4*self.d_model, action_dim * chunk_size)
        )
    
    def compute_loss(
        self,
        state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        out = self.forward(state)
        
        return torch.nn.functional.mse_loss(out, action_chunk)


    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)

    def forward(
        self,
        state,
    ) -> torch.Tensor:
        B = state.shape[0]
        ee_gripper = state[:, :4]
        cube_red = state[:, 4:7]
        cube_green = state[:, 7:10]
        cube_blue = state[:, 10:13]
        goal_onehot = state[:, 13:16]
        goal_pos = state[:, 16:19]

        # relative positions to simplify learning problem
        ee_pos = state[:, :3]
        cube_red_rel = cube_red - ee_pos
        cube_green_rel = cube_green - ee_pos
        cube_blue_rel = cube_blue - ee_pos
        goal_pos_rel = goal_pos - ee_pos

        target_idx = torch.argmax(goal_onehot, dim=1)
        cubes_rel = torch.stack([cube_red_rel, cube_green_rel, cube_blue_rel], dim=1)
        batch_indices = torch.arange(B, device=state.device)
        target_cube_rel = cubes_rel[batch_indices, target_idx]

        ee_encoded = self.ee_encoder(ee_gripper)
        container_encoded = self.bin_encoder(goal_pos_rel)
        target_cube_encoded = self.cube_encoder(target_cube_rel)

        x = torch.cat([
            ee_encoded,
            target_cube_encoded,
            container_encoded
        ], dim=1)

        out = self.mlp_policy(x)  # (B, action_dim * chunk_size)
        return out.view(B, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model=None,
    depth=None,
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
            chunk_size=chunk_size
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
