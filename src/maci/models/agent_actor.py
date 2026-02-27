from __future__ import annotations
import torch
from torch import Tensor, nn
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .actor import Actor
from .critic import Critic

if TYPE_CHECKING:
    from .agent_critic import AgentCritic

class AgentActor:
    def __init__(self, config: SimpleNamespace) -> None:
        self._actor = nn.Sequential(
            nn.Linear(config.dim.actor_flow, config.dim.actor_flow * 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(config.dim.actor_flow * 2, config.dim.actor_flow * 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(config.dim.actor_flow * 2, config.dim.actor_flow * 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(config.dim.actor_flow * 2, config.dim.actor_flow * 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(config.dim.actor_flow * 2, config.dim.actor_flow),
        )

        combined_flow_dim = config.dim.actor_flow * 2 + config.dim.critic_flow

        self._critic = nn.Sequential(
            nn.Linear(combined_flow_dim, combined_flow_dim * 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(combined_flow_dim * 2, combined_flow_dim * 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(combined_flow_dim * 2, combined_flow_dim * 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(combined_flow_dim * 2, combined_flow_dim),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(combined_flow_dim, config.dim.critic_flow),
        )


    def forward(self, actor_flows: Tensor) -> tuple[Tensor, Tensor]:
        new_actor_flows = self._actor(actor_flows)
        return new_actor_flows
    

    def compute_loss(
            self,
            agent_critic: AgentCritic,
            destination1_agent_actor: AgentActor,
            destination1_agent_critic: AgentCritic,
            destination2_agent_actor: AgentActor,
            destination2_agent_critic: AgentCritic,
            actor: Actor,
            critic: Critic,
            actor_flows: Tensor,
            critic_flows: Tensor,
            other_actor_flows: Tensor,
            other_critic_flows: Tensor,
    ) -> Tensor:
        self_actor_flows = self(actor_flows)
        self_critic_flows = agent_critic(actor_flows, self_actor_flows, critic_flows)
        destination1_actor_flows = destination1_agent_actor(self_critic_flows)
        destination1_critic_flows = destination1_agent_critic(
            self_actor_flows,
            destination1_actor_flows,
            self_critic_flows,
        )
        destination2_actor_flows = destination2_agent_actor(self_actor_flows)
        destination2_critic_flows = destination2_agent_critic(
            self_actor_flows,
            destination2_actor_flows,
            self_critic_flows,
        )

        actor_critic_flows = [
            (self_actor_flows, self_critic_flows),
            (destination1_actor_flows, destination1_critic_flows),
            (destination2_actor_flows, destination2_critic_flows),
        ]

        loss = 0
        for this_actor_flows, this_critic_flows in actor_critic_flows:
            actor_flows = other_actor_flows.append(this_actor_flows)
            critic_flows = other_critic_flows.append(this_critic_flows)
            actions = actor(this_actor_flows)
            q_values = critic(critic_flows, actions)
            loss -= q_values.mean()
        
        return loss