"""A compact heterogeneous graph policy for truck-order assignment.

The network is intentionally independent from route legality.  It scores the
economic value of all feasible truck-order edges jointly; the runtime applies a
frozen feasibility mask before decoding a conflict-free recommendation set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    truck_dim: int = 32
    order_dim: int = 24
    pair_dim: int = 16
    d_model: int = 192
    layers: int = 4
    dropout: float = 0.08
    max_trucks: int = 16
    max_orders: int = 32

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


class ResidualMLP(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state: Tensor, message: Tensor) -> Tensor:
        return self.norm(state + self.net(torch.cat((state, message), dim=-1)))


class RelationLayer(nn.Module):
    def __init__(self, d_model: int, pair_dim: int, dropout: float) -> None:
        super().__init__()
        self.relation = nn.Sequential(
            nn.Linear(pair_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.edge_gate = nn.Linear(d_model, 1)
        self.truck_update = ResidualMLP(d_model, dropout)
        self.order_update = ResidualMLP(d_model, dropout)

    def forward(
        self,
        truck_state: Tensor,
        order_state: Tensor,
        pair_features: Tensor,
        truck_mask: Tensor,
        order_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        relation = self.relation(pair_features)
        edge = truck_state.unsqueeze(2) + order_state.unsqueeze(1) + relation
        logits = self.edge_gate(torch.tanh(edge)).squeeze(-1)

        order_edge_mask = order_mask.unsqueeze(1).expand_as(logits)
        truck_edge_mask = truck_mask.unsqueeze(2).expand_as(logits)

        truck_attention = torch.softmax(logits.masked_fill(~order_edge_mask, -1.0e4), dim=2)
        truck_attention = truck_attention * order_edge_mask.to(truck_attention.dtype)
        truck_attention = truck_attention / truck_attention.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
        truck_message = (truck_attention.unsqueeze(-1) * (order_state.unsqueeze(1) + relation)).sum(dim=2)

        order_attention = torch.softmax(logits.masked_fill(~truck_edge_mask, -1.0e4), dim=1)
        order_attention = order_attention * truck_edge_mask.to(order_attention.dtype)
        order_attention = order_attention / order_attention.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        order_message = (order_attention.unsqueeze(-1) * (truck_state.unsqueeze(2) + relation)).sum(dim=1)

        truck_state = self.truck_update(truck_state, truck_message)
        order_state = self.order_update(order_state, order_message)
        truck_state = truck_state * truck_mask.unsqueeze(-1).to(truck_state.dtype)
        order_state = order_state * order_mask.unsqueeze(-1).to(order_state.dtype)
        return truck_state, order_state


class BackhaulGraphPolicy(nn.Module):
    """Scores every feasible truck-order edge from a complete market snapshot."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.truck_encoder = nn.Sequential(
            nn.Linear(config.truck_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.order_encoder = nn.Sequential(
            nn.Linear(config.order_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            RelationLayer(d_model, config.pair_dim, config.dropout)
            for _ in range(config.layers)
        )
        final_width = d_model * 3
        self.pair_fusion = nn.Sequential(
            nn.Linear(config.pair_dim, d_model),
            nn.SiLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(final_width, d_model * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model * 2, 1),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(final_width, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.log_variance_head = nn.Sequential(
            nn.Linear(final_width, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.eta_head = nn.Sequential(
            nn.Linear(final_width, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Softplus(),
        )
        self.wait_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        truck_features: Tensor,
        order_features: Tensor,
        pair_features: Tensor,
        truck_mask: Tensor,
        order_mask: Tensor,
        feasible_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        truck_state = self.truck_encoder(truck_features)
        order_state = self.order_encoder(order_features)
        truck_state = truck_state * truck_mask.unsqueeze(-1).to(truck_state.dtype)
        order_state = order_state * order_mask.unsqueeze(-1).to(order_state.dtype)

        for layer in self.layers:
            truck_state, order_state = layer(
                truck_state,
                order_state,
                pair_features,
                truck_mask,
                order_mask,
            )

        relation = self.pair_fusion(pair_features)
        fused = torch.cat(
            (
                truck_state.unsqueeze(2).expand(-1, -1, order_state.shape[1], -1),
                order_state.unsqueeze(1).expand(-1, truck_state.shape[1], -1, -1),
                relation,
            ),
            dim=-1,
        )
        policy_logits = self.policy_head(fused).squeeze(-1)
        utility = self.utility_head(fused).squeeze(-1)
        log_variance = self.log_variance_head(fused).squeeze(-1).clamp(-5.0, 3.0)
        eta_hours = self.eta_head(fused).squeeze(-1)
        wait_logits = self.wait_head(truck_state).squeeze(-1)

        valid_edges = feasible_mask & truck_mask.unsqueeze(2) & order_mask.unsqueeze(1)
        policy_logits = policy_logits.masked_fill(~valid_edges, -1.0e4)
        wait_logits = wait_logits.masked_fill(~truck_mask, -1.0e4)
        return policy_logits, wait_logits, utility, log_variance, eta_hours


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
