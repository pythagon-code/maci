from torch import Tensor, nn
from types import SimpleNamespace

class AgentRouter:
    def __init__(self, config: SimpleNamespace) -> None:
        combined_flow_dim = config.dim.actor_flow * 2 + config.critic_flow
        
        self._q_net = nn.Sequential(
            nn.Linear(combined_flow_dim, combined_flow_dim * 2),
            nn.LayerNorm(),
            nn.LeakyReLU(),
            nn.Linear(combined_flow_dim * 2, combined_flow_dim * 2),
        )