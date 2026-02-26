from __future__ import annotations
from torch import Tensor

class Neuron:
    def __init__(self, id) -> None:
        self._id = id
        self._sources = []
        self._targets = []

    
    def add_out_neighbor(self, neuron: Neuron) -> None:
        self._targets.append(neuron)
        neuron._sources.append(self)


    def process_flow(self, actor_flow: Tensor, critic_flow: Tensor) -> None:
