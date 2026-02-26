from __future__ import annotations
import torch
from torch import Tensor, nn
from types import SimpleNamespace
from typing import TYPE_CHECKING

from ..core.config import config
from .critic import Critic

class Actor:
    def __init__(self) -> None:
        self._qkv_net = nn.Sequential(
            nn.Linear(config.dim.actor_flow, config.dim.actor_flow * 2),
            nn.LeakyReLU(),
            nn.Linear(config.dim.actor_flow * 2, config.dim.actor_flow * 4),
            nn.LeakyReLU(),
            nn.Linear(config.dim.actor_flow * 4, config.transformer.embed_dim * 8),
            nn.LeakyReLU(),
            nn.Linear(config.transformer.embed_dim * 8, config.transformer.embed_dim * 8),
            nn.LeakyReLU(),
            nn.Linear(config.transformer.embed_dim * 8, config.transformer.embed_dim * 3),
        )

        self._attn = nn.MultiheadAttention(
            embed_dim = config.transformer.embed_dim,
            num_heads = config.transformer.num_heads,
        )


        self._critic = nn.Sequential(
            nn.Linear(config.transformer.embed_dim, config.transformer.embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(config.transformer.embed_dim * 2, config.transformer.embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(config.transformer.embed_dim * 2, config.transformer.embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(config.transformer.embed_dim * 2, config.dim.action * 4),
            nn.LeakyReLU(),
            nn.Linear(config.dim.action * 4, config.dim.action),
        )

    
    def forward(self, actor_flow: Tensor) -> Tensor:
        qkv = self._qkv_net(actor_flow)
        q = qkv[: config.transformer.embed_dim]
        k = qkv[config.transformer.embed_dim : config.transformer.embed_dim * 2]
        v = qkv[config.transformer.embed_dim * 2 :]
        attn_out, _ = self._attn(q, k, v)
        actions = self._actor(attn_out)
        return actions
    

    def compute_loss(
            self,
            critic: "Critic",
            actor_flows: Tensor,
            critic_flows: Tensor,
    ) -> Tensor:
        self.actions = self(actor_flows)
        q_values = critic(self.actions, critic_flows)
        loss = -q_values.mean()
        return loss