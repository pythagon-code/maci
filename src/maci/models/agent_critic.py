from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from types import SimpleNamespace

from .actor import Actor
from .agent_actor import AgentActor
from .critic import Critic

class AgentCritic:
    def __init__(self, config: SimpleNamespace) -> None:
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

        self._q_net = nn.Sequential(
            nn.Linear(combined_flow_dim, combined_flow_dim),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(combined_flow_dim, combined_flow_dim // 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(combined_flow_dim // 2, combined_flow_dim // 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(combined_flow_dim, 3),
        )


    def forward(self, actor_flows: Tensor, self_actor_flows: Tensor, critic_flows: Tensor) -> Tensor:
        combined_flows = torch.cat([actor_flows, self_actor_flows, critic_flows], dim=1)
        self_critic_flows = self.critic(combined_flows)
        return self_critic_flows


    def compute_loss(
            self,
            agent_critic: AgentCritic,
            destination1_agent_critic: AgentCritic,
            destination2_agent_critic: AgentCritic,
            actor: Actor,
            critic: Critic,
            actor_flows: Tensor,
            critic_flows: Tensor,
            self_actor_flows: Tensor,
            destination1_actor_flows,
            destination2_actor_flows,
            other_critic_flows: Tensor,
            actions: Tensor,
            rewards: Tensor,
            next_actor_flows: Tensor,
            next_critic_flows: Tensor,
            routes: Tensor,
            routes_target_q_values: Tensor
    ) -> Tensor:
        self_critic_flows = agent_critic(actor_flows, self_actor_flows, critic_flows)
        destination1_critic_flows = destination1_agent_critic(
            self_actor_flows,
            destination1_actor_flows,
            self_critic_flows,
        )
        destination2_critic_flows = destination2_agent_critic(
            self_actor_flows,
            destination2_actor_flows,
            self_critic_flows,
        )

        loss = 0
        routes_critic_flows = [self_critic_flows, destination1_critic_flows, destination2_critic_flows]
        for this_critic_flows in routes_critic_flows:
            critic_flows = other_critic_flows.append(this_critic_flows)
            q_values = critic(critic_flows, actions)
            with torch.no_grad():
                next_actions = actor(next_actor_flows)
                next_q_values = critic(next_critic_flows, next_actions)
                target_q_values = rewards + next_q_values
            loss += mse_loss(q_values, target_q_values)
        
        all_q_values = self._q_net(self_critic_flows)
        routes_q_values = all_q_values.gather(routes)
        loss += mse_loss(routes_q_values, routes_target_q_values)
        
        return loss