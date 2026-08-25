#!/usr/bin/env python3
"""Train RealBackhaulNet from scratch using only registered public observations."""

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
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from backhaul_real.model import ModelConfig, RealBackhaulNet, parameter_count


ROUTE_SOURCES = ("amazon", "lade")
TRAIN_FILES = {
    "amazon": "amazon_train.npz",
    "lade": "lade_train.npz",
    "telemetry": "singapore_train.npz",
    "dtcargo": "dtcargo_train.npz",
    "vius": "vius_train.npz",
    "health": "scania_train.npz",
    "price": "tlc_train.npz",
}
VAL_FILES = {key: value.replace("_train.npz", "_val.npz") for key, value in TRAIN_FILES.items()}
FINAL_FILES = {
    "amazon": "amazon_final.npz",
    "lade": "lade_test.npz",
    "telemetry": "singapore_test.npz",
    "dtcargo": "dtcargo_test.npz",
    "vius": "vius_test.npz",
    "health": "scania_final.npz",
    "price": "tlc_final.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--eval-batches", type=int, default=24)
    parser.add_argument("--final-eval-batches", type=int, default=24)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=9.25)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--graph-layers", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=3)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_split(data_root: Path, names: Mapping[str, str]) -> Dict[str, Mapping[str, np.ndarray]]:
    loaded: Dict[str, Mapping[str, np.ndarray]] = {}
    for source, filename in names.items():
        path = data_root / filename
        if path.exists():
            loaded[source] = np.load(path, mmap_mode="r", allow_pickle=False)
    return loaded


def first_array(dataset: Mapping[str, np.ndarray]) -> np.ndarray:
    for key in dataset.keys():
        array = dataset[key]
        if getattr(array, "ndim", 0) > 0:
            return array
    raise ValueError("dataset contains no arrays")


def dataset_size(dataset: Mapping[str, np.ndarray]) -> int:
    return int(first_array(dataset).shape[0])


def choose(dataset: Mapping[str, np.ndarray], aliases: Sequence[str]) -> np.ndarray:
    for name in aliases:
        if name in dataset:
            return dataset[name]
    raise KeyError(f"none of {aliases!r} found; available={list(dataset.keys())}")


def optional(
    dataset: Mapping[str, np.ndarray], aliases: Sequence[str], default: np.ndarray
) -> np.ndarray:
    for name in aliases:
        if name in dataset:
            return dataset[name]
    return default


def index_batch(size: int, batch_size: int, generator: np.random.Generator) -> np.ndarray:
    return generator.integers(0, size, size=min(batch_size, size), endpoint=False)


def tensor(array: np.ndarray, index: np.ndarray, device: torch.device, dtype=None) -> Tensor:
    value = torch.as_tensor(np.asarray(array[index]), device=device)
    return value if dtype is None else value.to(dtype=dtype)


def route_batch(
    dataset: Mapping[str, np.ndarray], index: np.ndarray, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    nodes_np = choose(dataset, ("nodes", "candidate_features"))
    context_np = choose(dataset, ("context", "context_features"))
    mask_np = choose(dataset, ("mask", "candidate_mask"))
    target_np = choose(dataset, ("target", "next_index"))
    count = nodes_np.shape[0]
    feasible_np = optional(dataset, ("feasible", "feasible_mask"), np.asarray(mask_np))
    eta_np = optional(dataset, ("eta_hours", "eta_target"), np.full((count,), np.nan, np.float32))
    stop_np = optional(dataset, ("stop_target", "end_target"), np.zeros((count,), np.int64))
    return (
        tensor(nodes_np, index, device, torch.float32),
        tensor(context_np, index, device, torch.float32),
        tensor(mask_np, index, device, torch.bool),
        tensor(feasible_np, index, device, torch.bool),
        tensor(target_np, index, device, torch.long),
        tensor(eta_np, index, device, torch.float32),
        tensor(stop_np, index, device, torch.long),
    )


def telemetry_batch(dataset, index, device):
    sequence_np = choose(dataset, ("sequence", "features"))
    count, steps = sequence_np.shape[:2]
    mask_np = optional(dataset, ("mask", "sequence_mask"), np.ones((count, steps), bool))
    targets_np = optional(dataset, ("targets", "telemetry_targets"), np.full((count, 4), np.nan, np.float32))
    if "targets" not in dataset and "telemetry_targets" not in dataset:
        labels = []
        for names in (("fuel", "fuel_l", "fuel_target"), ("duration", "duration_s"), ("load", "load_target"), ("idle", "idle_target")):
            labels.append(optional(dataset, names, np.full((count,), np.nan, np.float32)))
        targets_np = np.stack(labels, axis=1)
    return (
        tensor(sequence_np, index, device, torch.float32),
        tensor(mask_np, index, device, torch.bool),
        tensor(targets_np, index, device, torch.float32),
    )


def tabular_batch(dataset, index, device, feature_aliases=("features",), target_aliases=("targets", "target", "label")):
    return (
        tensor(choose(dataset, feature_aliases), index, device, torch.float32),
        tensor(choose(dataset, target_aliases), index, device, torch.float32),
    )


def masked_smooth_l1(prediction: Tensor, target: Tensor) -> Tensor:
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[valid], target[valid])


def quantile_loss(prediction: Tensor, target: Tensor, quantiles: Tensor) -> Tensor:
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    error = target[valid].unsqueeze(-1) - prediction[valid]
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


def task_loss(model: RealBackhaulNet, source: str, dataset, index, device) -> Tuple[Tensor, Dict[str, float]]:
    if source in ROUTE_SOURCES:
        nodes, context, mask, feasible, target, eta_target, stop_target = route_batch(dataset, index, device)
        logits, eta, stop = model(nodes, context, mask, feasible)
        route = F.cross_entropy(logits.float(), target)
        selected_eta = eta[torch.arange(target.shape[0], device=device), target]
        eta_loss = quantile_loss(selected_eta.float(), eta_target, torch.tensor([0.5, 0.9], device=device))
        stop_loss = F.cross_entropy(stop.float(), stop_target.clamp(0, 1))
        loss = route + 0.25 * eta_loss + 0.05 * stop_loss
        return loss, {"route": float(route.detach()), "eta": float(eta_loss.detach())}

    if source == "telemetry":
        sequence, mask, target = telemetry_batch(dataset, index, device)
        output = model.telemetry(sequence, mask).float()
        regression = masked_smooth_l1(output[:, :2], target[:, :2])
        classification = output.sum() * 0.0
        for column in (2, 3):
            valid = torch.isfinite(target[:, column])
            if bool(valid.any()):
                classification = classification + F.binary_cross_entropy_with_logits(
                    output[valid, column], target[valid, column]
                )
        return regression + 0.5 * classification, {"telemetry_reg": float(regression.detach())}

    features, target = tabular_batch(dataset, index, device)
    if source == "dtcargo":
        output = model.dtcargo(features).float()
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        regression = masked_smooth_l1(output[:, :2], target[:, :2])
        classification = output.sum() * 0.0
        for column in range(2, min(4, target.shape[1])):
            valid = torch.isfinite(target[:, column])
            if bool(valid.any()):
                classification = classification + F.binary_cross_entropy_with_logits(
                    output[valid, column], target[valid, column]
                )
        return regression + 0.5 * classification, {"dtcargo_reg": float(regression.detach())}
    if source == "vius":
        output = model.vius(features).float()
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        loss = masked_smooth_l1(output[:, : target.shape[1]], target)
        return loss, {"vius_deadhead_reg": float(loss.detach())}
    if source == "health":
        target = target.reshape(-1)
        logits = model.health(features).float()
        positives = target.sum().clamp_min(1.0)
        negatives = (target.numel() - target.sum()).clamp_min(1.0)
        positive_weight = (negatives / positives).clamp(1.0, 80.0)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
        return loss, {"health_bce": float(loss.detach())}
    if source == "price":
        target = target.reshape(-1)
        prediction = model.price(features).float()
        ordered = torch.stack(
            (
                prediction[:, 0],
                prediction[:, 0] + F.softplus(prediction[:, 1]),
                prediction[:, 0] + F.softplus(prediction[:, 1]) + F.softplus(prediction[:, 2]),
            ), dim=1,
        )
        loss = quantile_loss(ordered, target, torch.tensor([0.1, 0.5, 0.9], device=device))
        return loss, {"price_pinball": float(loss.detach())}
    raise KeyError(source)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order].astype(np.float64)
    positives = labels.sum()
    if positives <= 0:
        return float("nan")
    precision = np.cumsum(labels) / (np.arange(labels.size) + 1.0)
    return float((precision * labels).sum() / positives)


@torch.no_grad()
def evaluate_source(model, source, dataset, device, batch_size, max_batches=None) -> Dict[str, float]:
    model.eval()
    size = dataset_size(dataset)
    limit = size if max_batches is None else min(size, batch_size * max_batches)
    indices = np.arange(limit)
    result: Dict[str, list] = {}
    for start in range(0, limit, batch_size):
        index = indices[start : start + batch_size]
        if source in ROUTE_SOURCES:
            nodes, context, mask, feasible, target, eta_target, _ = route_batch(dataset, index, device)
            logits, eta, _ = model(nodes, context, mask, feasible)
            order = logits.argsort(dim=1, descending=True)
            prediction = order[:, 0]
            top3 = (order[:, :3] == target.unsqueeze(1)).any(dim=1)
            # Feature 2 is canonical current-to-candidate distance; this is the
            # deterministic nearest-neighbour baseline on the same real rows.
            baseline = nodes[:, :, 2].masked_fill(~(mask & feasible), 1.0e6).argmin(dim=1)
            selected_eta = eta[torch.arange(target.shape[0], device=device), target, 0]
            valid_eta = torch.isfinite(eta_target)
            for key, value in {
                "correct": (prediction == target).float(),
                "top3": top3.float(),
                "baseline_correct": (baseline == target).float(),
                "constraint_violation": (~feasible[torch.arange(target.shape[0], device=device), prediction]).float(),
            }.items():
                result.setdefault(key, []).append(value.cpu().numpy())
            if bool(valid_eta.any()):
                result.setdefault("eta_abs", []).append((selected_eta[valid_eta] - eta_target[valid_eta]).abs().cpu().numpy())
        elif source == "telemetry":
            sequence, mask, target = telemetry_batch(dataset, index, device)
            output = model.telemetry(sequence, mask).float()
            for column, name in ((0, "fuel_abs"), (1, "duration_abs")):
                valid = torch.isfinite(target[:, column])
                if bool(valid.any()):
                    result.setdefault(name, []).append((output[valid, column] - target[valid, column]).abs().cpu().numpy())
            for column, name in ((2, "load"), (3, "idle")):
                valid = torch.isfinite(target[:, column])
                if bool(valid.any()):
                    result.setdefault(name + "_label", []).append(target[valid, column].cpu().numpy())
                    result.setdefault(name + "_score", []).append(torch.sigmoid(output[valid, column]).cpu().numpy())
        else:
            features, target = tabular_batch(dataset, index, device)
            if source == "dtcargo":
                output = model.dtcargo(features).float()
                if target.ndim == 1:
                    target = target.unsqueeze(-1)
                for column, name in ((0, "duration_abs"), (1, "signal_abs")):
                    valid = torch.isfinite(target[:, column])
                    if bool(valid.any()):
                        result.setdefault(name, []).append((output[valid, column] - target[valid, column]).abs().cpu().numpy())
                for column, name in ((2, "home_base"), (3, "long_haul")):
                    if column < target.shape[1]:
                        valid = torch.isfinite(target[:, column])
                        if bool(valid.any()):
                            result.setdefault(name + "_label", []).append(target[valid, column].cpu().numpy())
                            result.setdefault(name + "_score", []).append(torch.sigmoid(output[valid, column]).cpu().numpy())
            elif source == "vius":
                if target.ndim == 1:
                    target = target.unsqueeze(-1)
                output = model.vius(features).float()
                for column, name in ((0, "deadhead_abs"), (1, "reposition_abs"), (2, "loaded_abs")):
                    if column < target.shape[1]:
                        valid = torch.isfinite(target[:, column])
                        if bool(valid.any()):
                            result.setdefault(name, []).append(
                                (output[valid, column] - target[valid, column]).abs().cpu().numpy()
                            )
            elif source == "health":
                label = target.reshape(-1).cpu().numpy()
                score = torch.sigmoid(model.health(features).float()).cpu().numpy()
                result.setdefault("label", []).append(label)
                result.setdefault("score", []).append(score)
            elif source == "price":
                label = target.reshape(-1)
                raw = model.price(features).float()
                median = raw[:, 0] + F.softplus(raw[:, 1])
                result.setdefault("price_abs", []).append((median - label).abs().cpu().numpy())

    merged = {key: np.concatenate(value) for key, value in result.items() if value}
    metrics: Dict[str, float] = {"examples": float(limit)}
    if source in ROUTE_SOURCES:
        metrics.update(
            next_stop_top1=float(merged["correct"].mean()),
            next_stop_top3=float(merged["top3"].mean()),
            nearest_neighbor_top1=float(merged["baseline_correct"].mean()),
            hard_constraint_violations=float(merged["constraint_violation"].sum()),
        )
        if "eta_abs" in merged:
            metrics["eta_mae_canonical"] = float(merged["eta_abs"].mean())
    elif source == "health":
        average_precision_value = average_precision(merged["label"], merged["score"])
        if math.isfinite(average_precision_value):
            metrics["aps_failure_average_precision"] = average_precision_value
        prediction = merged["score"] >= 0.5
        metrics["aps_failure_recall_at_0_5"] = float((prediction & (merged["label"] == 1)).sum() / max(1, (merged["label"] == 1).sum()))
    elif source == "telemetry":
        if "fuel_abs" in merged:
            metrics["fuel_mae_canonical"] = float(merged["fuel_abs"].mean())
        if "duration_abs" in merged:
            metrics["duration_mae_canonical"] = float(merged["duration_abs"].mean())
        for name in ("load", "idle"):
            if name + "_label" in merged:
                average_precision_value = average_precision(
                    merged[name + "_label"], merged[name + "_score"]
                )
                if math.isfinite(average_precision_value):
                    metrics[name + "_average_precision"] = average_precision_value
    else:
        for name, values in merged.items():
            if name.endswith("_abs"):
                metrics[name.replace("_abs", "_mae_canonical")] = float(values.mean())
        if source == "dtcargo":
            for name in ("home_base", "long_haul"):
                if name + "_label" in merged:
                    average_precision_value = average_precision(
                        merged[name + "_label"], merged[name + "_score"]
                    )
                    if math.isfinite(average_precision_value):
                        metrics[name + "_average_precision"] = average_precision_value
    return metrics


@torch.no_grad()
def validation_objective(model, datasets, device, batch_size, max_batches) -> Tuple[float, Dict[str, Dict[str, float]]]:
    metrics = {
        source: evaluate_source(model, source, dataset, device, batch_size, max_batches)
        for source, dataset in datasets.items()
    }
    scores: list[float] = []
    for source, values in metrics.items():
        source_scores: list[float] = []
        if source in ROUTE_SOURCES and "next_stop_top1" in values:
            source_scores.append(values["next_stop_top1"])
        for name, value in values.items():
            if name.endswith("_average_precision") and math.isfinite(value):
                source_scores.append(value)
            elif name.endswith("_mae_canonical") and math.isfinite(value):
                source_scores.append(1.0 / (1.0 + max(0.0, value)))
        if source_scores:
            scores.append(float(np.mean(source_scores)))
    objective = float(np.mean(scores)) if scores else -float("inf")
    return objective, metrics


def export_frozen(model, config, output, validation, device) -> Dict[str, Any]:
    if "amazon" not in validation:
        raise RuntimeError("Amazon validation observations are required for real-data parity export")
    dataset = validation["amazon"]
    index = np.arange(min(4, dataset_size(dataset)))
    nodes, context, mask, feasible, _, _, _ = route_batch(dataset, index, torch.device("cpu"))
    export_model = copy.deepcopy(model).float().cpu().eval()

    telemetry_source = validation.get("telemetry")
    if telemetry_source is None:
        raise RuntimeError("Singapore telemetry validation observations are required for export")
    tele_index = np.arange(min(4, dataset_size(telemetry_source)))
    sequence, sequence_mask, _ = telemetry_batch(telemetry_source, tele_index, torch.device("cpu"))

    def real_tabular(source: str, width: int) -> Tensor:
        if source not in validation:
            raise RuntimeError(f"{source} real validation observations are required for export")
        features, _ = tabular_batch(validation[source], np.arange(min(4, dataset_size(validation[source]))), torch.device("cpu"))
        if features.shape[1] != width:
            raise RuntimeError(f"{source} feature width {features.shape[1]} != {width}")
        return features

    examples = {
        "forward": (nodes, context, mask, feasible),
        "telemetry": (sequence, sequence_mask),
        "dtcargo": (real_tabular("dtcargo", config.dtcargo_dim),),
        "vius": (real_tabular("vius", config.vius_dim),),
        "health": (real_tabular("health", config.health_dim),),
        "price": (real_tabular("price", config.price_dim),),
    }
    # Explicit eager/frozen parity below is the authoritative export check.
    # Avoid tracing every method twice: on CPU that duplicate Transformer pass
    # can consume a material part of a deadline-bounded release run.
    traced = torch.jit.trace_module(
        export_model, examples, strict=True, check_trace=False
    )
    frozen = torch.jit.freeze(
        traced.eval(), preserved_attrs=["telemetry", "dtcargo", "vius", "health", "price"]
    )
    path = output / "real_backhaul_policy_frozen.pt"
    torch.jit.save(frozen, str(path))
    loaded = torch.jit.load(str(path), map_location="cpu").eval()
    differences = []
    for method, args in examples.items():
        eager_method = export_model if method == "forward" else getattr(export_model, method)
        frozen_method = loaded if method == "forward" else getattr(loaded, method)
        eager_output = eager_method(*args)
        frozen_output = frozen_method(*args)
        eager_values = eager_output if isinstance(eager_output, tuple) else (eager_output,)
        frozen_values = frozen_output if isinstance(frozen_output, tuple) else (frozen_output,)
        differences.extend(float((a - b).abs().max()) for a, b in zip(eager_values, frozen_values))
    maximum = max(differences)
    if maximum > 1.0e-5:
        raise RuntimeError(f"frozen parity failed: {maximum}")
    return {"path": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size, "max_abs_parity_difference": maximum}


def learning_rate_scale(step: int, total: int, warmup: int = 300) -> float:
    if step <= warmup:
        return max(0.02, step / warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> int:
    args = parse_args()
    if not 0 < args.max_hours <= 9.5:
        raise ValueError("--max-hours must be in (0, 9.5]")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("training must run on an NVIDIA A100")
    data_root = args.data.resolve()
    data_manifest = data_root / "manifest.json"
    if not data_manifest.exists():
        raise RuntimeError("prepared real-data manifest.json is missing")
    manifest_payload = json.loads(data_manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("contains_synthetic") is not False:
        raise RuntimeError("manifest must explicitly declare contains_synthetic=false")

    train = load_split(data_root, TRAIN_FILES)
    validation = load_split(data_root, VAL_FILES)
    required = set(TRAIN_FILES)
    if set(train) != required or set(validation) != required:
        raise RuntimeError(f"all real-source tasks are required; train={sorted(train)}, val={sorted(validation)}")
    # Final files are deliberately not opened until model selection and export.

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    config = ModelConfig(
        d_model=args.d_model,
        graph_layers=args.graph_layers,
        temporal_layers=args.temporal_layers,
    )
    for source in ROUTE_SOURCES:
        nodes = choose(train[source], ("nodes", "candidate_features"))
        if nodes.shape[1:] != (config.max_candidates, config.node_dim):
            raise RuntimeError(f"{source} nodes shape {nodes.shape} conflicts with config")

    model = RealBackhaulNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    generator = np.random.default_rng(args.seed + 17)
    schedule = ("amazon", "amazon", "lade", "telemetry", "dtcargo", "vius", "health", "price")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "training_history.jsonl"
    run_config = {
        "started_unix": time.time(),
        "seed": args.seed,
        "random_initialization": True,
        "pretrained_weights": False,
        "contains_synthetic": False,
        "cross_dataset_rows_joined": False,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "parameters": parameter_count(model),
        "model_config": config.to_dict(),
        "data_manifest_sha256": sha256(data_manifest),
        "arguments": {
            **vars(args),
            "data": "$REAL_DATA_ROOT",
            "output": "real_policy/artifacts",
        },
        "split_sizes": {
            split: {source: dataset_size(dataset) for source, dataset in values.items()}
            for split, values in (("train", train), ("validation", validation))
        },
    }
    write_json(output / "run_config.json", run_config)
    print(json.dumps({"event": "start", **run_config}, default=str), flush=True)

    started = time.monotonic()
    deadline = started + args.max_hours * 3600.0
    best_objective = -float("inf")
    best_state = None
    best_step = 0
    best_validation_metrics = None
    stale = 0
    completed_step = 0
    history_path.write_text("", encoding="utf-8")

    for step in range(1, args.steps + 1):
        if time.monotonic() >= deadline:
            break
        source = schedule[(step - 1) % len(schedule)]
        dataset = train[source]
        index = index_batch(dataset_size(dataset), args.batch_size, generator)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * learning_rate_scale(step, args.steps)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, parts = task_loss(model, source, dataset, index, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {source} loss at step {step}")
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        completed_step = step

        if step % 100 == 0:
            print(json.dumps({"event": "train", "step": step, "source": source, "loss": float(loss.detach()), **parts}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            objective, validation_metrics = validation_objective(model, validation, device, args.batch_size, args.eval_batches)
            event = {"event": "validation", "step": step, "objective": objective, "metrics": validation_metrics, "elapsed_seconds": time.monotonic() - started}
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps(event, allow_nan=False), flush=True)
            if objective > best_objective + 1.0e-5:
                best_objective = objective
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
                best_validation_metrics = validation_metrics
                stale = 0
                torch.save(
                    {
                        "state_dict": best_state,
                        "model_config": config.to_dict(),
                        "step": best_step,
                        "validation_objective": best_objective,
                    },
                    output / "best_checkpoint.pt",
                )
            else:
                stale += 1
                if stale >= args.patience:
                    print(json.dumps({"event": "early_stop", "step": step}), flush=True)
                    break

    if best_state is None or best_validation_metrics is None:
        raise RuntimeError("training finished without a validated checkpoint")
    model.load_state_dict(best_state)
    validation_objective_value = best_objective
    validation_metrics = best_validation_metrics
    artifact = export_frozen(model, config, output, validation, device)

    # This is the first and only opening of official final partitions.
    final = load_split(data_root, FINAL_FILES)
    if set(final) != required:
        raise RuntimeError(f"all final real-source partitions are required; final={sorted(final)}")
    final_metrics = {
        source: evaluate_source(
            model,
            source,
            dataset,
            device,
            args.batch_size,
            args.final_eval_batches,
        )
        for source, dataset in final.items()
    }
    metrics = {
        "selection": {"best_step": best_step, "validation_objective": validation_objective_value},
        "validation": validation_metrics,
        "final_held_out": final_metrics,
        "final_evaluation": {
            "max_batches_per_source": args.final_eval_batches,
            "batch_size": args.batch_size,
            "ordering": "first N rows of each immutable deterministically ordered held-out split",
            "reason": "deadline-bounded competition release; examples field records the exact evaluated count",
        },
        "artifact": artifact,
        "wall_seconds": time.monotonic() - started,
        "completed_steps": completed_step,
        "claim_boundary": "component-level real-public-data validation; no Haulio operational uplift claim",
    }
    write_json(output / "metrics.json", metrics)
    artifact_manifest = {
        "artifact": artifact,
        "model": "RealBackhaulNet",
        "model_config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "random_initialization": True,
        "pretrained_weights": False,
        "contains_synthetic": False,
        "cross_dataset_rows_joined": False,
        "data_manifest_sha256": sha256(data_manifest),
        "official_final_opened_after_freeze": True,
        "dispatcher_approval_required": True,
        "runtime_auto_update": False,
        "preprocessing": {
            "route": {
                "feature_schema": "real-route-canonical-16.v1",
                "candidate_selection": "nearest published or road-derived distance without target access",
            },
            "telemetry": {"feature_schema": "singapore-obd-canonical-8.v1"},
            "dtcargo": {"feature_schema": "dtcargo-summary-8-values-plus-mask.v1"},
            "vius": manifest_payload["sources"]["vius"]["normalization"],
            "health": manifest_payload["sources"]["scania"]["normalization"],
            "price": {"feature_schema": "nyc-tlc-logfare-12.v1"},
        },
        "output_schema": {
            "route_eta": "source-domain canonical hours",
            "telemetry": "first-half-prefix forecast of full-trip log1p fuel/duration, unavailable load, future idle-heavy logit",
            "dtcargo": "log1p duration, signal ratio, home and long-haul logits",
            "vius": "annual fractions",
            "health": "APS failure logit",
            "price": "log1p NYC metered fare quantiles",
        },
    }
    write_json(output / "manifest.json", artifact_manifest)
    completion = {
        "status": "complete",
        "completed_unix": time.time(),
        "best_step": best_step,
        "artifact_sha256": artifact["sha256"],
        "wall_seconds": metrics["wall_seconds"],
        "final_evaluation_capped": True,
    }
    write_json(output / "training_complete.json", completion)
    print(json.dumps({"event": "complete", **completion}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
