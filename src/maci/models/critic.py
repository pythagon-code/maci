from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from ..core.config import config
from .actor import Actor


class Critic:
    def __init__(self) -> None:
        self._qkv_fc = nn.Sequential(
            nn.Linear(config.dim.critic_flow, config.dim.actor_flow * 2),
            nn.ELU(),
            nn.Linear(config.dim.actor_flow * 2, config.dim.actor_flow * 4),
            nn.ELU(),
            nn.Linear(config.dim.actor_flow * 4, config.transformer.embed_dim * 8),
            nn.ELU(),
            nn.Linear(config.transformer.embed_dim * 8, config.transformer.embed_dim * 8),
            nn.ELU(),
            nn.Linear(config.transformer.embed_dim * 8, config.transformer.embed_dim * 3),
        )

        self._attn = nn.MultiheadAttention(
            embed_dim=config.transformer.embed_dim,
            num_heads=config.transformer.num_heads,
        )

        combined_input_dim = config.transformer.embed_dim + config.dim.output

        self._critic = nn.Sequential(
            nn.Linear(combined_input_dim, combined_input_dim * 2),
            nn.ELU(),
            nn.Linear(combined_input_dim * 2, combined_input_dim * 2),
            nn.ELU(),
            nn.Linear(combined_input_dim * 2, combined_input_dim * 2),
            nn.ELU(),
            nn.Linear(combined_input_dim * 2, combined_input_dim),
            nn.ELU(),
            nn.Linear(combined_input_dim, 1),
        )


    def forward(self, critic_flows: Tensor, actions: Tensor) -> Tensor:
        qkv = self._qkv_fc(critic_flows)
        q = qkv[: config.transformer.embed_dim]
        k = qkv[config.transformer.embed_dim : config.transformer.embed_dim * 2]
        v = qkv[config.transformer.embed_dim * 2 :]
        attn_out, _ = self._attn(q, k, v)
        combined_input = torch.cat((attn_out, actions), dim=1)
        q_values = self._critic(combined_input)
        return q_values
        

    def compute_loss(
            self,
            actor: Actor,
            target_self: Critic,
            critic_flows: Tensor,
            actions: Tensor,
            rewards: Tensor,
            next_actor_flows: Tensor,
            next_critic_flows: Tensor,
    ) -> Tensor:
        q_values = self(critic_flows, actions)
        with torch.no_grad():
            next_actions = actor(next_actor_flows)
            next_q_values = target_self(next_critic_flows, next_actions)
            target_q_values = rewards + next_q_values
        loss = mse_loss(q_values, target_q_values)
        return loss