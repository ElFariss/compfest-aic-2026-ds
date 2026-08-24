#!/usr/bin/env python3
"""Train and freeze the Haulio backhaul graph policy on one A100.

This is an offline development script.  The generated submission directory
contains only the immutable TorchScript artifact and inference service.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from backhaul_policy.policy import BackhaulGraphPolicy, ModelConfig, parameter_count
from training.synthetic import SyntheticBatch, greedy_actions, make_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=8.75)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return max(0.02, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def realized_reward(batch: SyntheticBatch, actions: Tensor) -> Tensor:
    orders = batch.target_reward.shape[2]
    safe_action = actions.clamp(0, orders - 1)
    selected = torch.gather(batch.target_reward, 2, safe_action.unsqueeze(-1)).squeeze(-1)
    selected = torch.where(actions == orders, batch.target_wait_reward, selected)
    selected = selected.masked_fill(~batch.truck_mask, 0.0)
    return selected.sum(dim=1)


def duplicate_count(actions: Tensor, order_count: int, truck_mask: Tensor) -> int:
    total = 0
    for order_index in range(order_count):
        assigned = ((actions == order_index) & truck_mask).sum(dim=1)
        total += int(torch.relu(assigned - 1).sum().item())
    return total


@torch.no_grad()
def evaluate(
    model: BackhaulGraphPolicy,
    config: ModelConfig,
    device: torch.device,
    seed: int,
    batches: int,
    batch_size: int,
    stress: bool,
) -> Dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    learned_rewards = []
    teacher_rewards = []
    baseline_rewards = []
    agreements = []
    violations = 0
    duplicates = 0
    latency_ms = []

    for _ in range(batches):
        batch = make_batch(batch_size, config, device, generator, stress=stress)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            policy, wait, _, _, _ = model(
                batch.truck_features,
                batch.order_features,
                batch.pair_features,
                batch.truck_mask,
                batch.order_mask,
                batch.feasible_mask,
            )
        torch.cuda.synchronize()
        latency_ms.append((time.perf_counter() - started) * 1000.0 / batch_size)
        learned_action = greedy_actions(
            policy.float(), wait.float(), batch.truck_mask, batch.order_mask
        )
        baseline_action = greedy_actions(
            batch.baseline_scores,
            torch.zeros_like(batch.target_wait_reward),
            batch.truck_mask,
            batch.order_mask,
        )
        learned_rewards.append(realized_reward(batch, learned_action))
        teacher_rewards.append(realized_reward(batch, batch.teacher_actions))
        baseline_rewards.append(realized_reward(batch, baseline_action))
        valid_trucks = batch.truck_mask
        agreements.append(
            ((learned_action == batch.teacher_actions) & valid_trucks).sum().float()
            / valid_trucks.sum().clamp_min(1)
        )
        order_count = config.max_orders
        selected_order = learned_action.clamp(0, order_count - 1)
        selected_feasible = torch.gather(
            batch.feasible_mask, 2, selected_order.unsqueeze(-1)
        ).squeeze(-1)
        invalid = (learned_action != order_count) & valid_trucks & ~selected_feasible
        violations += int(invalid.sum().item())
        duplicates += duplicate_count(learned_action, order_count, valid_trucks)

    learned = torch.cat(learned_rewards)
    teacher = torch.cat(teacher_rewards)
    baseline = torch.cat(baseline_rewards)
    denominator = (teacher - baseline).abs().mean().clamp_min(1.0e-6)
    return {
        "learned_reward_mean": float(learned.mean().item()),
        "teacher_reward_mean": float(teacher.mean().item()),
        "baseline_reward_mean": float(baseline.mean().item()),
        "learned_vs_baseline": float((learned.mean() - baseline.mean()).item()),
        "teacher_gap_normalized": float(((teacher - learned).mean() / denominator).item()),
        "teacher_action_agreement": float(torch.stack(agreements).mean().item()),
        "hard_constraint_violations": float(violations),
        "duplicate_assignments": float(duplicates),
        "gpu_inference_ms_per_snapshot": float(np.percentile(latency_ms, 95)),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_frozen(
    model: BackhaulGraphPolicy,
    config: ModelConfig,
    output: Path,
    device: torch.device,
) -> Dict[str, Any]:
    export_model = copy.deepcopy(model).float().cpu().eval()
    example = (
        torch.zeros(1, config.max_trucks, config.truck_dim),
        torch.zeros(1, config.max_orders, config.order_dim),
        torch.zeros(1, config.max_trucks, config.max_orders, config.pair_dim),
        torch.ones(1, config.max_trucks, dtype=torch.bool),
        torch.ones(1, config.max_orders, dtype=torch.bool),
        torch.ones(1, config.max_trucks, config.max_orders, dtype=torch.bool),
    )
    traced = torch.jit.trace(export_model, example, strict=True)
    frozen = torch.jit.freeze(traced.eval())
    frozen_path = output / "backhaul_policy_frozen.pt"
    torch.jit.save(frozen, str(frozen_path))

    loaded = torch.jit.load(str(frozen_path), map_location="cpu").eval()
    generator = torch.Generator(device=device).manual_seed(99173)
    parity_batch = make_batch(8, config, device, generator, stress=True)
    eager_inputs = (
        parity_batch.truck_features.cpu(),
        parity_batch.order_features.cpu(),
        parity_batch.pair_features.cpu(),
        parity_batch.truck_mask.cpu(),
        parity_batch.order_mask.cpu(),
        parity_batch.feasible_mask.cpu(),
    )
    with torch.no_grad():
        eager_outputs = export_model(*eager_inputs)
        frozen_outputs = loaded(*eager_inputs)
    max_difference = max(
        float((eager - scripted).abs().max().item())
        for eager, scripted in zip(eager_outputs, frozen_outputs)
    )
    if max_difference > 1.0e-5:
        raise RuntimeError(f"Frozen parity failed: max difference {max_difference}")
    return {
        "path": frozen_path.name,
        "sha256": sha256(frozen_path),
        "size_bytes": frozen_path.stat().st_size,
        "max_abs_parity_difference": max_difference,
    }


def main() -> int:
    args = parse_args()
    if args.max_hours <= 0 or args.max_hours > 9.5:
        raise ValueError("--max-hours must be in (0, 9.5], preserving a safety buffer below 10 hours")
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA A100 CUDA device is required for this training run")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "A100" not in gpu_name:
        raise RuntimeError(f"Expected an A100, found {gpu_name}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "training_history.jsonl"

    config = ModelConfig(d_model=args.d_model, layers=args.layers)
    model = BackhaulGraphPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    start = time.monotonic()
    deadline = start + args.max_hours * 3600.0
    best_metric = -float("inf")
    best_state = None
    best_step = 0
    stale_evaluations = 0
    final_step = 0

    run_info = {
        "started_unix": time.time(),
        "seed": args.seed,
        "synthetic_digital_twin_only": True,
        "random_initialization": True,
        "gpu": gpu_name,
        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "parameter_count": parameter_count(model),
        "config": config.to_dict(),
        "arguments": vars(args) | {"output": str(output)},
    }
    write_json(output / "run_config.json", run_info)
    print(json.dumps({"event": "start", **run_info}, default=str), flush=True)

    for step in range(1, args.steps + 1):
        if time.monotonic() >= deadline:
            print(json.dumps({"event": "deadline", "step": step - 1}), flush=True)
            break
        learning_rate = args.learning_rate * schedule(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        stress_probability = min(0.35, step / max(1, args.steps) * 0.35)
        stress = bool(torch.rand((), device=device, generator=generator).item() < stress_probability)
        batch = make_batch(args.batch_size, config, device, generator, stress=stress)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            policy, wait, utility, log_variance, eta_hours = model(
                batch.truck_features,
                batch.order_features,
                batch.pair_features,
                batch.truck_mask,
                batch.order_mask,
                batch.feasible_mask,
            )
            all_logits = torch.cat((policy, wait.unsqueeze(-1)), dim=2)
            imitation_loss = functional.cross_entropy(
                all_logits.reshape(-1, config.max_orders + 1).float(),
                batch.teacher_actions.reshape(-1),
                ignore_index=-100,
            )
            valid = batch.feasible_mask
            residual = utility[valid].float() - batch.target_reward[valid]
            valid_log_variance = log_variance[valid].float()
            uncertainty_loss = 0.5 * (
                torch.exp(-valid_log_variance) * residual.square() + valid_log_variance
            ).mean()
            eta_loss = functional.smooth_l1_loss(
                torch.log1p(eta_hours[valid].float()),
                torch.log1p(batch.target_eta_hours[valid]),
            )
            probability = torch.softmax(all_logits.float(), dim=2)
            reward_with_wait = torch.cat(
                (batch.target_reward, batch.target_wait_reward.unsqueeze(-1)), dim=2
            )
            expected_reward = (
                probability * reward_with_wait.clamp_min(-20.0)
            ).sum(dim=2)
            expected_reward = (
                expected_reward * batch.truck_mask.float()
            ).sum() / batch.truck_mask.sum().clamp_min(1)
            collision = torch.relu(probability[..., :-1].sum(dim=1) - 1.0).square().mean()
            entropy = -(probability * probability.clamp_min(1.0e-8).log()).sum(dim=2)
            entropy = (entropy * batch.truck_mask.float()).sum() / batch.truck_mask.sum().clamp_min(1)
            direct_weight = 0.0 if step < args.warmup_steps else min(0.30, step / args.steps * 0.30)
            loss = (
                imitation_loss
                + 0.34 * uncertainty_loss
                + 0.16 * eta_loss
                + direct_weight * (-expected_reward + 0.45 * collision - 0.006 * entropy)
            )

        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_step = step

        if step == 1 or step % 50 == 0:
            elapsed = time.monotonic() - start
            event = {
                "event": "train",
                "step": step,
                "elapsed_s": round(elapsed, 2),
                "loss": round(float(loss.item()), 6),
                "imitation": round(float(imitation_loss.item()), 6),
                "utility_nll": round(float(uncertainty_loss.item()), 6),
                "eta": round(float(eta_loss.item()), 6),
                "expected_reward": round(float(expected_reward.item()), 6),
                "collision": round(float(collision.item()), 6),
                "lr": learning_rate,
                "gradient_norm": round(float(gradient_norm), 5),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1.0e9, 3),
            }
            with history_path.open("a", encoding="utf-8") as history:
                history.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            validation = evaluate(
                model,
                config,
                device,
                seed=args.seed + 100_000 + step,
                batches=args.eval_batches,
                batch_size=min(128, args.batch_size),
                stress=False,
            )
            stress_validation = evaluate(
                model,
                config,
                device,
                seed=args.seed + 200_000 + step,
                batches=max(3, args.eval_batches // 3),
                batch_size=min(96, args.batch_size),
                stress=True,
            )
            metric = validation["learned_reward_mean"] + 0.35 * stress_validation["learned_reward_mean"]
            event = {
                "event": "validation",
                "step": step,
                "metric": metric,
                "validation": validation,
                "stress_validation": stress_validation,
            }
            with history_path.open("a", encoding="utf-8") as history:
                history.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            if metric > best_metric:
                best_metric = metric
                best_step = step
                stale_evaluations = 0
                best_state = {
                    key: value.detach().float().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(
                    {
                        "state_dict": best_state,
                        "config": config.to_dict(),
                        "step": step,
                        "metric": metric,
                        "synthetic_digital_twin_only": True,
                    },
                    output / "best_checkpoint.pt",
                )
            else:
                stale_evaluations += 1
                if stale_evaluations >= args.patience:
                    print(json.dumps({"event": "early_stop", "step": step}), flush=True)
                    break

    if best_state is None:
        best_state = {key: value.detach().float().cpu() for key, value in model.state_dict().items()}
        best_step = final_step
    model.load_state_dict(best_state)
    model.to(device)
    final_validation = evaluate(
        model, config, device, args.seed + 700_001, max(20, args.eval_batches), 128, False
    )
    final_stress = evaluate(
        model, config, device, args.seed + 700_002, max(10, args.eval_batches // 2), 96, True
    )
    if final_validation["hard_constraint_violations"] != 0 or final_validation["duplicate_assignments"] != 0:
        raise RuntimeError("Safety gate failed: invalid or duplicate assignments were produced")
    frozen = export_frozen(model, config, output, device)
    metrics = {
        "best_step": best_step,
        "final_step": final_step,
        "elapsed_seconds": time.monotonic() - start,
        "validation": final_validation,
        "sensor_dropout_stress": final_stress,
        "frozen_artifact": frozen,
    }
    write_json(output / "metrics.json", metrics)
    manifest = {
        "schema_version": "haulio.backhaul-policy-manifest.v1",
        "model_name": "constraint-conditioned-backhaul-graph-policy",
        "model_version": "2026.08.25-a100",
        "model_type": "heterogeneous_graph_policy",
        "training": {
            "initialization": "random",
            "data": "synthetic Indonesia-style digital twin",
            "real_world_validation": False,
            "online_learning": False,
            "automatic_tuning": False,
            "best_step": best_step,
            "seed": args.seed,
        },
        "runtime": {
            "weights_static": True,
            "human_acceptance_required": True,
            "hard_constraints_external": True,
            "max_trucks": config.max_trucks,
            "max_orders": config.max_orders,
        },
        "feature_dimensions": {
            "truck": config.truck_dim,
            "order": config.order_dim,
            "pair": config.pair_dim,
        },
        "artifact": frozen,
        "metrics_file": "metrics.json",
    }
    write_json(output / "manifest.json", manifest)
    completion = {
        "status": "complete",
        "completed_unix": time.time(),
        "elapsed_seconds": metrics["elapsed_seconds"],
        "best_step": best_step,
        "artifact_sha256": frozen["sha256"],
    }
    write_json(output / "training_complete.json", completion)
    print(json.dumps({"event": "complete", **completion}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
