"""On-disk TITO artifact schema (v1) shared by the proxy writer and the
train-buffer loader.

One trajectory sample = one ``<stem>.npz`` (binary arrays) plus one
``<stem>.json`` sidecar (small metadata). The manifest path submitted to
``/v1/train`` is always the ``.npz`` path; the sidecar is optional for
loading.

npz entries:
    schema_version          int64 scalar, currently 1
    tokens                  int32  [total_len]            prompt + response
    loss_mask               int32  [response_length]      1 per trainable token
    rollout_log_probs       float32[response_length]
    response_length         int64 scalar
    rollout_routed_experts  int32  [total_len - 1, num_layers, topk]    (optional; trailing
                            suffix rows are -1 — appended client-side, never prefilled)

json sidecar:
    {"schema_version": 1, "task_id": ..., "response": ..., "metadata": {...}}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Stamped into every artifact; load_artifact rejects any other version. Bump on
# any format change: old artifacts then fail loudly instead of loading wrong.
SCHEMA_VERSION = 1


def write_artifact(artifact_root: str | Path, task_id: str, data: dict, metadata: dict | None = None) -> str:
    """Write one finalized TITO trajectory; returns the ``.npz`` path. The
    write goes to a temporary file first so a submitted path never names a
    partial artifact.

    Args:
        artifact_root: shared-filesystem directory, created if missing.
        task_id: names the file pair (sanitized); recorded verbatim in the sidecar.
        data: ``TaskState.finalize()`` output dict.
        metadata: optional caller dict, stored in the json sidecar.
    """
    _check_lengths(data, f"write task_id={task_id}")
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = root / _safe_stem(task_id)

    arrays = {
        "schema_version": np.int64(SCHEMA_VERSION),
        "tokens": np.asarray(data["tokens"], dtype=np.int32),
        "loss_mask": np.asarray(data["loss_mask"], dtype=np.int32),
        "rollout_log_probs": np.asarray(data["rollout_log_probs"], dtype=np.float32),
        "response_length": np.int64(data["response_length"]),
    }
    if data.get("rollout_routed_experts") is not None:
        arrays["rollout_routed_experts"] = np.asarray(data["rollout_routed_experts"], dtype=np.int32)

    npz_path = stem.with_suffix(".npz")
    if npz_path.exists():  # sanitized-stem collision or rerun into a live root
        raise FileExistsError(f"artifact already exists for task_id={task_id!r}: {npz_path}")
    tmp_path = stem.with_suffix(".npz.tmp")
    with open(tmp_path, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp_path, npz_path)

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "response": data.get("response", ""),
        "metadata": metadata or {},
    }
    sidecar_tmp = stem.with_suffix(".json.tmp")
    sidecar_tmp.write_text(json.dumps(sidecar), encoding="utf-8")
    os.replace(sidecar_tmp, stem.with_suffix(".json"))
    return str(npz_path)


def load_artifact(path: str | Path) -> dict:
    """Load one artifact back into the ``TaskState.finalize()`` dict shape.
    Raises on missing/malformed files; validation happens at manifest-submit
    time, so queued groups should never fail here."""
    path = Path(path)
    with np.load(path) as npz:
        version = int(npz["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported TITO artifact schema_version {version} in {path}")
        data = {
            "tokens": npz["tokens"].astype(np.int64).tolist(),
            "loss_mask": npz["loss_mask"].astype(np.int64).tolist(),
            "rollout_log_probs": npz["rollout_log_probs"].astype(np.float64).tolist(),
            "rollout_routed_experts": npz["rollout_routed_experts"] if "rollout_routed_experts" in npz else None,
            "response_length": int(npz["response_length"]),
        }

    data["response"] = ""
    sidecar_path = path.with_suffix(".json")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        data["response"] = sidecar.get("response", "")
        data["sidecar_metadata"] = sidecar.get("metadata", {})

    _check_lengths(data, f"load {path}")
    return data


def validate_artifact(path: str | Path) -> None:
    """Cheap existence/readability check used at manifest-submit time."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"TITO artifact not found: {path}")


def _check_lengths(data: dict, where: str) -> None:
    """Invariant enforced on write and load: loss_mask/log_probs cover exactly
    the ``response_length`` tail of ``tokens``."""
    n = data["response_length"]
    if len(data["loss_mask"]) != n or len(data["rollout_log_probs"]) != n or len(data["tokens"]) < n:
        raise ValueError(
            f"inconsistent TITO artifact ({where}): response_length={n}, "
            f"loss_mask={len(data['loss_mask'])}, log_probs={len(data['rollout_log_probs'])}, "
            f"tokens={len(data['tokens'])}"
        )


def _safe_stem(task_id: str) -> str:
    """Filesystem-safe filename stem: any char outside [alnum-_] becomes _."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)
