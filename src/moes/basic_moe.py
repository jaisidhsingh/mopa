import math
import torch
from typing import *
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.utils import _assert_no_nan, _round_to_4_decimals


@dataclass
class MoEConfig():
    d_model: int
    n_experts: int
    k: int # set to `None` when using ExpertChoice routing
    d_hidden: int
    capacity_factor: float
    normalize_gates: bool
    name: str


class MoEOutput():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __getitem__(self, key):
        return getattr(self, key)


class MoE(ABC, nn.Module):
    def __init__(self, config: MoEConfig, experts: nn.ModuleList):
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.capacity_factor = config.capacity_factor
        self.normalize_gates = config.normalize_gates
        self.name = config.name
        self.config = config

        self.experts = experts
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)
    
    @staticmethod
    @abstractmethod
    def _capacity(tokens:int, n_experts:int, capacity_factor:float) -> int:
        return math.ceil(capacity_factor * tokens  / n_experts)
    
    @abstractmethod
    def compute_routing_probabilities(self, x: Tensor) -> Tensor:
        assert x.ndim == 3 and x.size(-1) == self.d_model
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        _assert_no_nan(probs, f"{self.name}.probs")
        return probs

    @abstractmethod 
    def route_to_experts(self, input_tensor: Tensor, routing_probs: Tensor) -> Tuple:
        # implement the unique logic of the routing scheme you want to use
        pass

    def forward(self, x: Tensor) -> MoEOutput:
        routing_probs = self.compute_routing_probabilities(x)
        output_tensor, aux_losses, diagnostics = self.route_to_experts(x, routing_probs)
        return MoEOutput(
            output_tensor=output_tensor,
            aux_losses=aux_losses,
            diagnostics=diagnostics
        ) 