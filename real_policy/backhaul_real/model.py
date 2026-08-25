"""Unified policy trained from random initialization on real public records.

The model never constructs a fake joined trip.  Each public source enters via
its own adapter and supervised head while sharing a fleet-state representation.
The route branch is a graph pointer policy: it scores every currently feasible
stop jointly and is called autoregressively until STOP/ABSTAIN at runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    node_dim: int = 16
    context_dim: int = 16
    telemetry_dim: int = 8
    dtcargo_dim: int = 16
    vius_dim: int = 24
    health_dim: int = 340
    price_dim: int = 12
    d_model: int = 160
    graph_layers: int = 4
    temporal_layers: int = 3
    heads: int = 8
    dropout: float = 0.08
    max_candidates: int = 32
    telemetry_steps: int = 64

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.net(self.norm(value))


class GraphPointerBlock(nn.Module):
    """Candidate-to-candidate attention plus context-conditioned message flow."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.context_gate = nn.Sequential(
            nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.feed_forward = ResidualBlock(width, dropout)

    def forward(self, nodes: Tensor, context: Tensor, padding_mask: Tensor) -> Tensor:
        normalized = self.attention_norm(nodes)
        message, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        nodes = nodes + message
        expanded_context = context.unsqueeze(1).expand_as(nodes)
        nodes = nodes + self.context_gate(torch.cat((nodes, expanded_context), dim=-1))
        return self.feed_forward(nodes)


class RealBackhaulNet(nn.Module):
    """Graph routing policy with real-source auxiliary truck-IoT heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model

        self.node_adapter = nn.Sequential(
            nn.Linear(config.node_dim, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.context_adapter = nn.Sequential(
            nn.Linear(config.context_dim, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.graph = nn.ModuleList(
            GraphPointerBlock(width, config.heads, config.dropout)
            for _ in range(config.graph_layers)
        )
        self.route_query = nn.Sequential(
            nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.route_key = nn.Linear(width, width, bias=False)
        self.route_bias = nn.Sequential(
            nn.Linear(config.node_dim + config.context_dim, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.eta_head = nn.Sequential(
            nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, 2), nn.Softplus()
        )
        self.stop_head = nn.Sequential(
            nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, 2)
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.heads,
            dim_feedforward=width * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.telemetry_adapter = nn.Sequential(
            nn.Linear(config.telemetry_dim, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.telemetry_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=config.temporal_layers
        )
        self.dtcargo_adapter = nn.Sequential(
            nn.Linear(config.dtcargo_dim, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.vius_adapter = nn.Sequential(
            nn.Linear(config.vius_dim, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.health_adapter = nn.Sequential(
            nn.Linear(config.health_dim, width * 2),
            nn.SiLU(),
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
        )
        self.price_adapter = nn.Sequential(
            nn.Linear(config.price_dim, width), nn.SiLU(), nn.LayerNorm(width)
        )

        # This trunk is shared by every modality. Dataset-specific batches do
        # not need to be joined to update a common operational representation.
        self.shared_state = nn.Sequential(
            ResidualBlock(width, config.dropout),
            ResidualBlock(width, config.dropout),
            nn.LayerNorm(width),
        )
        self.telemetry_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 4)
        )
        self.dtcargo_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 4)
        )
        self.vius_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3), nn.Sigmoid()
        )
        self.health_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.price_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3)
        )

    def forward(
        self,
        nodes: Tensor,
        context: Tensor,
        candidate_mask: Tensor,
        feasible_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return next-stop logits, ETA quantiles, and stop-after-action logits."""
        valid = candidate_mask & feasible_mask
        state = self.node_adapter(nodes)
        context_state = self.shared_state(self.context_adapter(context))
        padding_mask = ~candidate_mask
        for block in self.graph:
            state = block(state, context_state, padding_mask)

        valid_float = candidate_mask.unsqueeze(-1).to(state.dtype)
        pooled = (state * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        query = self.route_query(torch.cat((context_state, pooled), dim=-1))
        keys = self.route_key(state)
        scale = float(self.config.d_model) ** -0.5
        dot_score = (keys * query.unsqueeze(1)).sum(dim=-1) * scale
        raw_context = context.unsqueeze(1).expand(-1, nodes.shape[1], -1)
        bias = self.route_bias(torch.cat((nodes, raw_context), dim=-1)).squeeze(-1)
        logits = (dot_score + bias).masked_fill(~valid, -1.0e4)

        expanded_context = context_state.unsqueeze(1).expand_as(state)
        eta = self.eta_head(torch.cat((state, expanded_context), dim=-1))
        stop = self.stop_head(torch.cat((context_state, pooled), dim=-1))
        return logits, eta, stop

    def telemetry(self, sequence: Tensor, sequence_mask: Tensor) -> Tensor:
        state = self.telemetry_adapter(sequence)
        encoded = self.telemetry_encoder(state, src_key_padding_mask=~sequence_mask)
        weight = sequence_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        # full-trip log fuel/duration from a prefix, unused load logit, and
        # future-half idle-heavy logit
        return self.telemetry_head(self.shared_state(pooled))

    def dtcargo(self, features: Tensor) -> Tensor:
        # log duration, signal-loss ratio, home-base logit, long-haul logit
        return self.dtcargo_head(self.shared_state(self.dtcargo_adapter(features)))

    def vius(self, features: Tensor) -> Tensor:
        # Annual observed deadhead, repositioning and loaded-mile fractions.
        return self.vius_head(self.shared_state(self.vius_adapter(features)))

    def health(self, features: Tensor) -> Tensor:
        return self.health_head(self.shared_state(self.health_adapter(features))).squeeze(-1)

    def price(self, features: Tensor) -> Tensor:
        # log-fare P10/P50/P90; monotonicity is imposed in the loss/runtime.
        return self.price_head(self.shared_state(self.price_adapter(features)))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
