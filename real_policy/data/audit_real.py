#!/usr/bin/env python3
"""Audit raw lineage, processed hashes, split isolation, and tensor contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPECTED_SOURCES = {"amazon", "lade", "singapore", "dtcargo", "vius", "scania", "tlc"}
GROUP_KEYS = ("route_group", "split_group")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    require(path == root or root in path.parents, f"path escapes declared root: {relative}")
    return path


def check_manifest_contract(manifest: Mapping[str, Any]) -> None:
    require(manifest.get("contains_synthetic") is False, "contains_synthetic must be false")
    require(manifest.get("cross_source_rows_joined") is False, "cross-source rows must not be joined")
    require(manifest.get("random_feature_generation") is False, "random feature generation must be false")
    require(manifest.get("source_specific_batches") is True, "source-specific batching must be true")
    require(manifest.get("candidate_selection_reads_target") is False, "candidate selection must not read target")
    selected = set(manifest.get("selected_sources", []))
    require(selected == EXPECTED_SOURCES, f"expected all seven sources, found {sorted(selected)}")
    require(set(manifest.get("sources", {})) == EXPECTED_SOURCES, "source manifest keys do not match")


def validate_arrays(source: str, split: str, arrays: Mapping[str, np.ndarray], rows: int) -> None:
    require(rows > 0, f"{source}/{split} is empty")
    require("sample_id" in arrays and "source_id" in arrays, f"{source}/{split} lacks lineage IDs")
    sample_ids = np.asarray(arrays["sample_id"]).astype(str)
    source_ids = np.asarray(arrays["source_id"]).astype(str)
    require(len(sample_ids) == rows and len(source_ids) == rows, f"{source}/{split} lineage row mismatch")
    require(len(set(sample_ids.tolist())) == rows, f"{source}/{split} contains duplicate sample IDs")
    require(np.all(np.char.str_len(sample_ids) > 0), f"{source}/{split} has blank sample IDs")
    require(np.all(np.char.str_len(source_ids) > 0), f"{source}/{split} has blank source IDs")

    if source in {"amazon", "lade"}:
        for key in ("nodes", "context", "mask", "target"):
            require(key in arrays, f"{source}/{split} lacks {key}")
        nodes = np.asarray(arrays["nodes"])
        context = np.asarray(arrays["context"])
        mask = np.asarray(arrays["mask"], dtype=bool)
        target = np.asarray(arrays["target"], dtype=np.int64)
        require(nodes.shape == (rows, 32, 16), f"{source}/{split} node shape is {nodes.shape}")
        require(context.shape == (rows, 16), f"{source}/{split} context shape is {context.shape}")
        require(mask.shape == (rows, 32), f"{source}/{split} mask shape is {mask.shape}")
        require(np.all((target >= 0) & (target < 32)), f"{source}/{split} has invalid targets")
        require(np.all(mask[np.arange(rows), target]), f"{source}/{split} has masked targets")
        require(np.all(np.isfinite(context)), f"{source}/{split} has non-finite context")
        require(np.all(np.isfinite(nodes[mask])), f"{source}/{split} has non-finite active nodes")
    elif source == "singapore":
        sequence = np.asarray(arrays["sequence"])
        mask = np.asarray(arrays["mask"], dtype=bool)
        targets = np.asarray(arrays["targets"])
        require(sequence.shape == (rows, 64, 8), f"singapore/{split} sequence shape is {sequence.shape}")
        require(mask.shape == (rows, 64) and np.all(mask.any(axis=1)), f"singapore/{split} mask invalid")
        require(np.all(np.isfinite(sequence)), f"singapore/{split} sequence contains non-finite values")
        require(targets.shape == (rows, 4), f"singapore/{split} targets shape is {targets.shape}")
        require(np.all(np.isnan(targets[:, 2])), "Singapore cargo-load target must remain unavailable")
        require(np.isfinite(targets[:, 0]).any(), f"Singapore/{split} has no usable held-out fuel labels")
        require(np.all(np.isfinite(targets[:, 1])), "Singapore duration target must be finite")
    else:
        features = np.asarray(arrays["features"])
        target_key = next(
            (name for name in ("targets", "target", "label") if name in arrays),
            None,
        )
        require(target_key is not None, f"{source}/{split} lacks a target array")
        targets = np.asarray(arrays[target_key])
        expected_width = {"dtcargo": 16, "vius": 24, "scania": 340, "tlc": 12}[source]
        require(features.shape == (rows, expected_width), f"{source}/{split} feature shape is {features.shape}")
        require(np.all(np.isfinite(features)), f"{source}/{split} features contain non-finite values")
        finite_by_row = np.isfinite(targets.reshape(rows, -1)).any(axis=1)
        require(np.all(finite_by_row), f"{source}/{split} has rows without any target")


def audit(raw_root: Path, processed_root: Path) -> dict[str, Any]:
    manifest_path = processed_root / "manifest.json"
    require(manifest_path.is_file(), "processed manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check_manifest_contract(manifest)
    preprocessor_path = Path(__file__).with_name("prepare_real.py")
    require(preprocessor_path.is_file(), "prepare_real.py is missing beside the auditor")
    preprocessor_sha256 = sha256(preprocessor_path)
    require(
        manifest.get("preprocessing_sha256") == preprocessor_sha256,
        "preprocessing code SHA-256 does not match the manifest",
    )
    split_report: dict[str, Any] = {}
    raw_count = 0
    total_rows = 0

    for source in sorted(EXPECTED_SOURCES):
        entry = manifest["sources"][source]
        raw_files = entry.get("raw_files", [])
        require(bool(raw_files), f"{source} has no raw lineage records")
        for raw_record in raw_files:
            relative = raw_record.get("path")
            require(isinstance(relative, str) and relative, f"{source} raw path missing")
            path = safe_child(raw_root, relative)
            require(path.is_file(), f"raw file missing: {relative}")
            require(path.stat().st_size == raw_record.get("bytes"), f"raw size mismatch: {relative}")
            require(raw_record.get("sha256") == sha256(path), f"raw SHA-256 mismatch: {relative}")
            raw_count += 1

        seen_groups: set[str] = set()
        split_report[source] = {}
        for split, file_info in sorted(entry.get("files", {}).items()):
            path = safe_child(processed_root, file_info["path"])
            require(path.is_file(), f"processed file missing: {path.name}")
            require(path.stat().st_size == file_info["bytes"], f"processed size mismatch: {path.name}")
            require(sha256(path) == file_info["sha256"], f"processed SHA-256 mismatch: {path.name}")
            with np.load(path, allow_pickle=False) as bundle:
                arrays = {key: np.asarray(bundle[key]) for key in bundle.files}
            rows = int(file_info["rows"])
            for key, descriptor in file_info["arrays"].items():
                require(key in arrays, f"{path.name} lacks declared array {key}")
                require(list(arrays[key].shape) == descriptor["shape"], f"{path.name}/{key} shape mismatch")
                require(str(arrays[key].dtype) == descriptor["dtype"], f"{path.name}/{key} dtype mismatch")
                require(arrays[key].shape[0] == rows, f"{path.name}/{key} row mismatch")
                require(arrays[key].dtype.kind != "O", f"{path.name}/{key} uses object/pickle data")
            validate_arrays(source, split, arrays, rows)
            sample_hash = hashlib.sha256("\n".join(map(str, arrays["sample_id"])).encode()).hexdigest()
            require(sample_hash == file_info["sample_id_order_sha256"], f"{path.name} sample order hash mismatch")
            group_key = next((key for key in GROUP_KEYS if key in arrays), None)
            require(group_key is not None, f"{path.name} has no split group")
            groups = set(map(str, arrays[group_key]))
            require(not seen_groups.intersection(groups), f"{source} group leakage into {split}")
            seen_groups.update(groups)
            group_hash = hashlib.sha256("\n".join(sorted(groups)).encode()).hexdigest()
            require(group_hash == file_info["group_set_sha256"], f"{path.name} group hash mismatch")
            total_rows += rows
            split_report[source][split] = {
                "rows": rows,
                "sha256": file_info["sha256"],
                "groups": len(groups),
            }

    return {
        "status": "pass",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(manifest_path),
        "preprocessing_sha256": preprocessor_sha256,
        "contains_synthetic": False,
        "cross_source_rows_joined": False,
        "candidate_selection_reads_target": False,
        "raw_files_hash_verified": raw_count,
        "processed_examples_verified": total_rows,
        "splits": split_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.raw.resolve(), args.processed.resolve())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
