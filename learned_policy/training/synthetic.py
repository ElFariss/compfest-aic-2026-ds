"""Vectorized Indonesia-style digital twin for backhaul policy training.

The generator creates complete state/action/outcome tensors without claiming
that synthetic samples are Haulio production data.  Every model feature maps to
an obtainable IoT, fleet-master, order, facility, or road-context field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from backhaul_policy.policy import ModelConfig


@dataclass
class SyntheticBatch:
    truck_features: Tensor
    order_features: Tensor
    pair_features: Tensor
    truck_mask: Tensor
    order_mask: Tensor
    feasible_mask: Tensor
    target_reward: Tensor
    target_wait_reward: Tensor
    target_eta_hours: Tensor
    teacher_actions: Tensor
    baseline_scores: Tensor


def _gather_hubs(centres: Tensor, indexes: Tensor) -> Tensor:
    return torch.gather(centres, 1, indexes.unsqueeze(-1).expand(-1, -1, 2))


def _rand(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    return torch.rand(shape, device=device, generator=generator)


def greedy_actions(
    reward: Tensor,
    wait_reward: Tensor,
    truck_mask: Tensor,
    order_mask: Tensor,
) -> Tensor:
    """Create conflict-free teacher actions from the simulated realized reward."""
    batch_size, trucks, orders = reward.shape
    actions = torch.full((batch_size, trucks), orders, dtype=torch.long, device=reward.device)
    available_trucks = truck_mask.clone()
    available_orders = order_mask.clone()
    batch_index = torch.arange(batch_size, device=reward.device)

    advantage = reward - wait_reward.unsqueeze(-1)
    for _ in range(trucks):
        allowed = available_trucks.unsqueeze(2) & available_orders.unsqueeze(1)
        flat = advantage.masked_fill(~allowed, -1.0e4).reshape(batch_size, -1)
        best_value, best_index = flat.max(dim=1)
        truck_index = torch.div(best_index, orders, rounding_mode="floor")
        order_index = best_index.remainder(orders)
        active = best_value > 0.0
        if not bool(active.any()):
            break
        active_batch = batch_index[active]
        active_truck = truck_index[active]
        active_order = order_index[active]
        actions[active_batch, active_truck] = active_order
        available_trucks[active_batch, active_truck] = False
        available_orders[active_batch, active_order] = False

    actions = actions.masked_fill(~truck_mask, -100)
    return actions


def make_batch(
    batch_size: int,
    config: ModelConfig,
    device: torch.device,
    generator: torch.Generator,
    stress: bool = False,
) -> SyntheticBatch:
    batch = batch_size
    trucks = config.max_trucks
    orders = config.max_orders

    truck_count = torch.randint(4, trucks + 1, (batch,), device=device, generator=generator)
    order_count = torch.randint(8, orders + 1, (batch,), device=device, generator=generator)
    truck_mask = torch.arange(trucks, device=device).unsqueeze(0) < truck_count.unsqueeze(1)
    order_mask = torch.arange(orders, device=device).unsqueeze(0) < order_count.unsqueeze(1)

    centres = 0.12 + 0.76 * _rand((batch, 8, 2), device, generator)
    truck_hubs = torch.randint(0, 8, (batch, trucks), device=device, generator=generator)
    home_hubs = torch.randint(0, 8, (batch, trucks), device=device, generator=generator)
    pickup_hubs = torch.randint(0, 8, (batch, orders), device=device, generator=generator)
    dropoff_hubs = torch.randint(0, 8, (batch, orders), device=device, generator=generator)
    truck_position = (_gather_hubs(centres, truck_hubs) + 0.025 * torch.randn(
        (batch, trucks, 2), device=device, generator=generator
    )).clamp(0.0, 1.0)
    home_position = _gather_hubs(centres, home_hubs)
    pickup_position = (_gather_hubs(centres, pickup_hubs) + 0.018 * torch.randn(
        (batch, orders, 2), device=device, generator=generator
    )).clamp(0.0, 1.0)
    dropoff_position = (_gather_hubs(centres, dropoff_hubs) + 0.018 * torch.randn(
        (batch, orders, 2), device=device, generator=generator
    )).clamp(0.0, 1.0)

    truck_type = torch.randint(0, 5, (batch, trucks), device=device, generator=generator)
    capacity_table = torch.tensor([8000.0, 9000.0, 16000.0, 14000.0, 18000.0], device=device)
    capacity_kg = capacity_table[truck_type] * (0.88 + 0.24 * _rand((batch, trucks), device, generator))
    fuel = 0.12 + 0.86 * _rand((batch, trucks), device, generator)
    speed = 0.85 * _rand((batch, trucks), device, generator)
    heading = 6.28318530718 * _rand((batch, trucks), device, generator)
    gps_accuracy = 4.0 + (85.0 if stress else 42.0) * _rand((batch, trucks), device, generator)
    available_after = 260.0 * _rand((batch, trucks), device, generator)
    telemetry_age = (900.0 if stress else 420.0) * _rand((batch, trucks), device, generator).pow(2)
    replayed = (_rand((batch, trucks), device, generator) < (0.14 if stress else 0.05)).float()
    cargo_empty = (_rand((batch, trucks), device, generator) < 0.58).float()
    current_load_ratio = (1.0 - cargo_empty) * (0.25 + 0.7 * _rand((batch, trucks), device, generator))

    sensor_dropout = 0.34 if stress else 0.12
    cargo_present = (_rand((batch, trucks), device, generator) > sensor_dropout).float()
    can_present = (_rand((batch, trucks), device, generator) > sensor_dropout).float()
    imu_present = (_rand((batch, trucks), device, generator) > sensor_dropout).float()
    health_present = (_rand((batch, trucks), device, generator) > sensor_dropout).float()
    rpm = can_present * (0.22 + 0.68 * speed + 0.07 * _rand((batch, trucks), device, generator))
    coolant = can_present * (0.60 + 0.34 * _rand((batch, trucks), device, generator))
    acceleration = imu_present * _rand((batch, trucks), device, generator).pow(2)
    gyro = imu_present * _rand((batch, trucks), device, generator).pow(2)
    power = health_present * (0.72 + 0.24 * _rand((batch, trucks), device, generator))
    signal = health_present * (0.18 + 0.78 * _rand((batch, trucks), device, generator))
    uptime = health_present * _rand((batch, trucks), device, generator)
    quality = (1.0 - telemetry_age.div(600.0).clamp(0.0, 1.0)) * (1.0 - gps_accuracy.div(100.0).clamp(0.0, 1.0))

    truck_features = torch.zeros((batch, trucks, config.truck_dim), device=device)
    truck_features[..., 0:2] = truck_position
    truck_features[..., 2:4] = home_position
    truck_features[..., 4] = capacity_kg / 20000.0
    truck_features[..., 5] = current_load_ratio * cargo_present
    truck_features[..., 6] = fuel
    truck_features[..., 7] = speed
    truck_features[..., 8] = torch.sin(heading)
    truck_features[..., 9] = torch.cos(heading)
    truck_features[..., 10] = gps_accuracy / 100.0
    truck_features[..., 11] = available_after / 720.0
    truck_features[..., 12] = telemetry_age.clamp_max(600.0) / 600.0
    truck_features[..., 13] = replayed
    truck_features[..., 14] = cargo_empty
    truck_features[..., 15] = cargo_present
    truck_features[..., 16] = can_present
    truck_features[..., 17] = imu_present
    truck_features[..., 18] = health_present
    truck_features[..., 19] = rpm
    truck_features[..., 20] = coolant
    truck_features[..., 21] = acceleration
    truck_features[..., 22] = gyro
    truck_features[..., 23] = power
    truck_features[..., 24] = signal
    truck_features[..., 25] = uptime
    truck_features[..., 26:31] = torch.nn.functional.one_hot(truck_type, num_classes=5).float()
    truck_features[..., 31] = quality

    order_type = torch.randint(0, 5, (batch, orders), device=device, generator=generator)
    cargo_class = torch.randint(0, 5, (batch, orders), device=device, generator=generator)
    weight_kg = 1200.0 + 17200.0 * _rand((batch, orders), device, generator).pow(1.35)
    pickup_by = 80.0 + 840.0 * _rand((batch, orders), device, generator)
    service_minutes = 25.0 + 75.0 * _rand((batch, orders), device, generator)
    order_age = 720.0 * _rand((batch, orders), device, generator)
    cancellation = 0.02 + 0.28 * _rand((batch, orders), device, generator).pow(2)
    manifest_bound = (_rand((batch, orders), device, generator) > (0.08 if stress else 0.025)).float()
    priority = _rand((batch, orders), device, generator)
    fragile = ((cargo_class == 1) | (cargo_class == 3)).float()

    loaded_direct = torch.linalg.vector_norm(dropoff_position - pickup_position, dim=-1) * 1280.0
    delivery_sla = pickup_by + service_minutes + loaded_direct / 42.0 * 60.0 + 120.0 + 480.0 * _rand(
        (batch, orders), device, generator
    )
    cargo_premium = 1.0 + 0.18 * (cargo_class == 1).float() + 0.28 * (cargo_class == 3).float()
    offer_idr = cargo_premium * (
        2_200_000.0 + loaded_direct * (14500.0 + 2500.0 * _rand((batch, orders), device, generator))
        + weight_kg * (250.0 + 110.0 * _rand((batch, orders), device, generator))
    )

    order_features = torch.zeros((batch, orders, config.order_dim), device=device)
    order_features[..., 0:2] = pickup_position
    order_features[..., 2:4] = dropoff_position
    order_features[..., 4] = weight_kg / 20000.0
    order_features[..., 5] = pickup_by / 1440.0
    order_features[..., 6] = delivery_sla / 2880.0
    order_features[..., 7] = offer_idr / 50_000_000.0
    order_features[..., 8] = order_age / 1440.0
    order_features[..., 9] = cancellation
    order_features[..., 10] = manifest_bound
    order_features[..., 11:16] = torch.nn.functional.one_hot(order_type, num_classes=5).float()
    order_features[..., 16:21] = torch.nn.functional.one_hot(cargo_class, num_classes=5).float()
    order_features[..., 21] = priority
    order_features[..., 22] = service_minutes / 180.0
    order_features[..., 23] = fragile

    deadhead_km = torch.linalg.vector_norm(
        truck_position.unsqueeze(2) - pickup_position.unsqueeze(1), dim=-1
    ) * 1280.0
    loaded_km = loaded_direct.unsqueeze(1).expand(-1, trucks, -1)
    weather = 0.7 * _rand((batch, 1, orders), device, generator)
    congestion = 0.75 * _rand((batch, 1, orders), device, generator)
    road_quality = 0.35 + 0.65 * _rand((batch, 1, orders), device, generator)
    road_factor = 1.0 + 0.42 * weather + 0.58 * congestion + 0.24 * (1.0 - road_quality)
    route_km = (deadhead_km + loaded_km) * (1.05 + 0.18 * (1.0 - road_quality))
    road_minutes = route_km / 52.0 * 60.0 * road_factor
    pickup_arrival = available_after.unsqueeze(2) + deadhead_km / 48.0 * 60.0 * road_factor
    delivery_arrival = pickup_arrival + service_minutes.unsqueeze(1) + loaded_km / 52.0 * 60.0 * road_factor
    pickup_slack = pickup_by.unsqueeze(1) - pickup_arrival
    delivery_slack = delivery_sla.unsqueeze(1) - delivery_arrival
    capacity_ratio = weight_kg.unsqueeze(1) / capacity_kg.unsqueeze(2).clamp_min(1.0)

    current_home_km = torch.linalg.vector_norm(truck_position - home_position, dim=-1).unsqueeze(2) * 1280.0
    drop_home_km = torch.linalg.vector_norm(
        dropoff_position.unsqueeze(1) - home_position.unsqueeze(2), dim=-1
    ) * 1280.0
    home_alignment = ((current_home_km - drop_home_km) / 1280.0).clamp(-1.0, 1.0)
    drop_to_pick = torch.linalg.vector_norm(
        dropoff_position.unsqueeze(2) - pickup_position.unsqueeze(1), dim=-1
    )
    demand_density_order = torch.exp(-drop_to_pick * 9.0).mean(dim=2)
    demand_density = demand_density_order.unsqueeze(1).expand(-1, trucks, -1)

    toll_idr = route_km * (720.0 + 540.0 * _rand((batch, 1, orders), device, generator))
    operating_cost = route_km * (5200.0 + 900.0 * capacity_ratio) + toll_idr + 120_000.0
    margin_idr = offer_idr.unsqueeze(1) - operating_cost
    thermal_risk = torch.relu(coolant - 0.82)
    health_risk = (
        0.34 * thermal_risk
        + 0.26 * (1.0 - signal)
        + 0.20 * (1.0 - fuel)
        + 0.20 * (1.0 - quality)
    ).unsqueeze(2).expand(-1, -1, orders)
    lateness = torch.relu(-pickup_slack / 240.0) + torch.relu(-delivery_slack / 360.0)
    uncertainty = (
        telemetry_age.unsqueeze(2) / 600.0
        + gps_accuracy.unsqueeze(2) / 100.0
        + 0.35 * replayed.unsqueeze(2)
    ).clamp(0.0, 2.0)

    stochastic = 0.055 * torch.randn((batch, trucks, orders), device=device, generator=generator)
    target_reward = (
        margin_idr / 10_000_000.0
        + 0.42 * demand_density
        + 0.30 * home_alignment
        - 1.10 * lateness
        - 0.55 * health_risk
        - 0.24 * uncertainty
        - 0.34 * cancellation.unsqueeze(1)
        + 0.10 * priority.unsqueeze(1)
        + stochastic
    )
    local_demand = torch.exp(
        -torch.linalg.vector_norm(
            truck_position.unsqueeze(2) - pickup_position.unsqueeze(1), dim=-1
        ) * 9.0
    ).mean(dim=2)
    target_wait_reward = -0.04 + 0.28 * local_demand - 0.14 * health_risk[..., 0]

    road_available = _rand((batch, 1, orders), device, generator) > (0.08 if stress else 0.025)
    feasible_mask = (
        truck_mask.unsqueeze(2)
        & order_mask.unsqueeze(1)
        & (truck_type.unsqueeze(2) == order_type.unsqueeze(1))
        & (capacity_ratio <= 1.0)
        & (fuel.unsqueeze(2) >= 0.10)
        & (telemetry_age.unsqueeze(2) <= 300.0)
        & (manifest_bound.unsqueeze(1) > 0.5)
        & (pickup_slack >= 0.0)
        & (delivery_slack >= 0.0)
        & road_available
    )
    target_reward = target_reward.masked_fill(~feasible_mask, -1.0e4)

    pair_features = torch.zeros((batch, trucks, orders, config.pair_dim), device=device)
    pair_features[..., 0] = deadhead_km / 1000.0
    pair_features[..., 1] = loaded_km / 1500.0
    pair_features[..., 2] = road_minutes / 1440.0
    pair_features[..., 3] = pickup_slack.clamp(-1440.0, 1440.0) / 1440.0
    pair_features[..., 4] = delivery_slack.clamp(-2880.0, 2880.0) / 2880.0
    pair_features[..., 5] = capacity_ratio.clamp(0.0, 2.0)
    pair_features[..., 6] = margin_idr.clamp(-50_000_000.0, 50_000_000.0) / 50_000_000.0
    pair_features[..., 7] = home_alignment
    pair_features[..., 8] = demand_density
    pair_features[..., 9] = weather.expand(-1, trucks, -1)
    pair_features[..., 10] = congestion.expand(-1, trucks, -1)
    pair_features[..., 11] = toll_idr / 5_000_000.0
    pair_features[..., 12] = road_quality.expand(-1, trucks, -1)
    pair_features[..., 13] = quality.unsqueeze(2)
    pair_features[..., 14] = health_risk
    pair_features[..., 15] = (road_factor > 1.75).float().expand(-1, trucks, -1)

    target_eta_hours = delivery_arrival / 60.0
    baseline_scores = (
        2.1 * pair_features[..., 6]
        - 0.35 * pair_features[..., 0]
        - 0.18 * pair_features[..., 2]
    ).masked_fill(~feasible_mask, -1.0e4)
    teacher_actions = greedy_actions(
        target_reward,
        target_wait_reward,
        truck_mask,
        order_mask,
    )
    return SyntheticBatch(
        truck_features=truck_features,
        order_features=order_features,
        pair_features=pair_features,
        truck_mask=truck_mask,
        order_mask=order_mask,
        feasible_mask=feasible_mask,
        target_reward=target_reward,
        target_wait_reward=target_wait_reward,
        target_eta_hours=target_eta_hours,
        teacher_actions=teacher_actions,
        baseline_scores=baseline_scores,
    )
