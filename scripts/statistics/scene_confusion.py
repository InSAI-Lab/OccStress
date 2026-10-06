#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Per-scene sufficient statistics for clustered occupancy bootstrapping."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _confusion(pred: Any, target: Any, classes: int) -> np.ndarray:
    if hasattr(pred, "detach") and hasattr(target, "detach"):
        import torch

        pred = pred.reshape(-1)
        target = target.reshape(-1)
        valid = (
            (target >= 0)
            & (target < classes)
            & (pred >= 0)
            & (pred < classes)
        )
        encoded = classes * target[valid].to(torch.int64) + pred[valid].to(torch.int64)
        return (
            torch.bincount(encoded, minlength=classes * classes)
            .reshape(classes, classes)
            .cpu()
            .numpy()
        )
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    valid = (target >= 0) & (target < classes) & (pred >= 0) & (pred < classes)
    encoded = classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def _confusion_horizons(
    pred: Any,
    target: Any,
    classes: int,
) -> np.ndarray:
    """Compute one confusion matrix per leading-axis horizon in one scan."""

    horizons = pred.shape[0]
    if hasattr(pred, "detach") and hasattr(target, "detach"):
        import torch

        pred = pred.to(torch.int64)
        target = target.to(torch.int64)
        valid = (
            (target >= 0)
            & (target < classes)
            & (pred >= 0)
            & (pred < classes)
        )
        offsets = torch.arange(
            horizons,
            device=target.device,
            dtype=torch.int64,
        ).reshape((horizons,) + (1,) * (target.ndim - 1))
        encoded = (
            offsets * (classes * classes)
            + classes * target
            + pred
        )
        return (
            torch.bincount(
                encoded[valid],
                minlength=horizons * classes * classes,
            )
            .reshape(horizons, classes, classes)
            .cpu()
            .numpy()
        )

    pred = np.asarray(pred, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    valid = (
        (target >= 0)
        & (target < classes)
        & (pred >= 0)
        & (pred < classes)
    )
    offsets = np.arange(horizons, dtype=np.int64).reshape(
        (horizons,) + (1,) * (target.ndim - 1)
    )
    encoded = offsets * (classes * classes) + classes * target + pred
    return np.bincount(
        encoded[valid],
        minlength=horizons * classes * classes,
    ).reshape(horizons, classes, classes)


def metrics_from_histograms(
    semantic_hist: np.ndarray,
    binary_hist: np.ndarray,
    *,
    free_class: int = 17,
    model: str | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute the benchmark metrics from summed confusion matrices."""

    semantic_hist = np.asarray(semantic_hist, dtype=np.float64)
    binary_hist = np.asarray(binary_hist, dtype=np.float64)
    semantic_union = (
        semantic_hist.sum(axis=-1)
        + semantic_hist.sum(axis=-2)
        - np.diagonal(semantic_hist, axis1=-2, axis2=-1)
    )
    semantic_seen = semantic_hist.sum(axis=-1)
    semantic_iou = np.divide(
        np.diagonal(semantic_hist, axis1=-2, axis2=-1),
        semantic_union,
        out=np.full_like(semantic_union, np.nan),
        where=semantic_union != 0,
    )
    normalized_model = (model or "").lower().replace("-", "").replace("_", "")
    if normalized_model == "iiworld":
        semantic_iou[semantic_iou == 0] = np.nan
    elif normalized_model in {"occworld", "come"}:
        semantic_iou[semantic_seen == 0] = 1.0
    miou = np.nanmean(semantic_iou[..., :free_class], axis=-1) * 100.0

    occupied_intersection = binary_hist[..., 1, 1]
    occupied_union = (
        binary_hist[..., 1, :].sum(axis=-1)
        + binary_hist[..., :, 1].sum(axis=-1)
        - occupied_intersection
    )
    occupied_iou = np.divide(
        occupied_intersection,
        occupied_union,
        out=np.full_like(occupied_union, np.nan),
        where=occupied_union != 0,
    ) * 100.0

    main_indices = np.asarray([1, 3, 5], dtype=np.int64)
    main_indices = main_indices[main_indices < miou.shape[-1]]
    return {
        "miou": miou,
        "iou": occupied_iou,
        "main_miou": float(np.nanmean(miou[..., main_indices])),
        "main_iou": float(np.nanmean(occupied_iou[..., main_indices])),
    }


class SceneConfusionAccumulator:
    """Accumulate semantic and occupancy confusion matrices by scene."""

    def __init__(
        self,
        *,
        output_path: str | os.PathLike[str],
        num_horizons: int = 6,
        num_classes: int = 18,
        free_class: int = 17,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.free_class = free_class
        self.metadata = dict(metadata or {})
        self._semantic: dict[str, np.ndarray] = {}
        self._binary: dict[str, np.ndarray] = {}
        self._anchor_counts: dict[str, int] = {}

    def update_batch(
        self,
        prediction: Any,
        target: Any,
        scene_tokens: Iterable[str],
    ) -> None:
        scene_tokens = [str(token) for token in scene_tokens]

        if len(scene_tokens) == 1 and prediction.ndim == 4:
            prediction = prediction[None, ...]
            target = target[None, ...]
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
        if prediction.shape[0] != len(scene_tokens):
            raise ValueError(
                f"Batch size {prediction.shape[0]} does not match {len(scene_tokens)} scene tokens"
            )
        if prediction.shape[1] != self.num_horizons:
            raise ValueError(
                f"Expected {self.num_horizons} horizons, got shape {prediction.shape}"
            )

        for sample_index, scene_token in enumerate(scene_tokens):
            semantic = self._semantic.setdefault(
                scene_token,
                np.zeros(
                    (self.num_horizons, self.num_classes, self.num_classes),
                    dtype=np.uint64,
                ),
            )
            binary = self._binary.setdefault(
                scene_token,
                np.zeros((self.num_horizons, 2, 2), dtype=np.uint64),
            )
            sample_prediction = prediction[sample_index]
            sample_target = target[sample_index]
            semantic += _confusion_horizons(
                sample_prediction,
                sample_target,
                self.num_classes,
            ).astype(np.uint64)
            binary += _confusion_horizons(
                sample_prediction != self.free_class,
                sample_target != self.free_class,
                2,
            ).astype(np.uint64)
            self._anchor_counts[scene_token] = self._anchor_counts.get(scene_token, 0) + 1

    def save(self) -> dict[str, Any]:
        if not self._semantic:
            raise RuntimeError("No samples were accumulated")

        scene_tokens = sorted(self._semantic)
        semantic_hist = np.stack([self._semantic[token] for token in scene_tokens])
        binary_hist = np.stack([self._binary[token] for token in scene_tokens])
        anchor_counts = np.asarray(
            [self._anchor_counts[token] for token in scene_tokens], dtype=np.int64
        )
        aggregate = metrics_from_histograms(
            semantic_hist.sum(axis=0),
            binary_hist.sum(axis=0),
            free_class=self.free_class,
            model=self.metadata.get("model"),
        )
        normalized_model = str(self.metadata.get("model", "")).lower()
        normalized_model = normalized_model.replace("-", "").replace("_", "")
        includes_current = normalized_model in {"occworld", "come"}
        horizon_seconds = [
            0.5 * (index if includes_current else index + 1)
            for index in range(self.num_horizons)
        ]
        metadata = {
            **self.metadata,
            "format_version": 1,
            "evaluation_contract": "submission_compatible_native_horizon",
            "num_scenes": len(scene_tokens),
            "num_anchors": int(anchor_counts.sum()),
            "num_horizons": self.num_horizons,
            "num_classes": self.num_classes,
            "free_class": self.free_class,
            "semantic_metric_convention": (
                "exclude_zero_iou_classes"
                if str(self.metadata.get("model", "")).lower()
                .replace("-", "")
                .replace("_", "")
                == "iiworld"
                else "absent_gt_class_iou_is_one"
            ),
            "horizon_seconds": horizon_seconds,
            "score_horizon_indices": [1, 3, 5],
            "score_horizon_seconds": [
                horizon_seconds[index]
                for index in (1, 3, 5)
                if index < len(horizon_seconds)
            ],
            "includes_current_reconstruction": includes_current,
            "aggregate_miou": np.asarray(aggregate["miou"]).tolist(),
            "aggregate_iou": np.asarray(aggregate["iou"]).tolist(),
            "aggregate_main_miou": aggregate["main_miou"],
            "aggregate_main_iou": aggregate["main_iou"],
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.output_path.parent,
            prefix=f".{self.output_path.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        try:
            np.savez_compressed(
                temporary_path,
                scene_tokens=np.asarray(scene_tokens),
                anchor_counts=anchor_counts,
                semantic_hist=semantic_hist,
                binary_hist=binary_hist,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            os.replace(temporary_path, self.output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        digest = hashlib.sha256(self.output_path.read_bytes()).hexdigest()
        metadata["sha256"] = digest
        sidecar = self.output_path.with_suffix(".json")
        sidecar_temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
        sidecar_temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(sidecar_temporary, sidecar)
        return metadata


def accumulator_from_environment(model: str, num_horizons: int = 6):
    output_path = os.environ.get("OCCSTRESS_SCENE_STATS")
    if not output_path:
        return None
    metadata = {
        "model": model,
        "protocol": os.environ.get("OCCSTRESS_PROTOCOL_NAME"),
        "protocol_path": os.environ.get("OCCSTRESS_PROTOCOL_PATH"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    checkpoint_sha256 = os.environ.get("OCCSTRESS_CHECKPOINT_SHA256")
    checkpoint_path = os.environ.get("OCCSTRESS_CHECKPOINT_PATH")
    if checkpoint_sha256:
        metadata["checkpoint_sha256"] = checkpoint_sha256
    if checkpoint_path:
        metadata["checkpoint_path"] = checkpoint_path
    robustocc_root = Path(
        os.environ.get("OCCSTRESS_CODE_ROOT", Path(__file__).resolve().parents[2])
    )
    provenance_path = Path(
        os.environ.get(
            "OCCSTRESS_PROVENANCE",
            robustocc_root
            / "work_dirs/statistical_uncertainty/analysis_provenance.json",
        )
    )
    if provenance_path.is_file() and not checkpoint_sha256:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        normalized_model = model.lower().replace("-", "").replace("_", "")
        checkpoint_key = {
            "occworld": "occworld_checkpoint",
            "come": "come_controlnet_checkpoint",
            "iiworld": "ii_world_checkpoint",
        }[normalized_model]
        metadata.update(
            {
                "provenance_path": str(provenance_path.resolve()),
                "checkpoint_sha256": provenance["files"][checkpoint_key][
                    "sha256"
                ],
            }
        )
    return SceneConfusionAccumulator(
        output_path=output_path,
        num_horizons=num_horizons,
        metadata=metadata,
    )


def scene_tokens_from_metas(metas: Any) -> list[str]:
    if isinstance(metas, dict):
        tokens = metas["scene_token"]
        if isinstance(tokens, str):
            return [tokens]
        return [str(token) for token in tokens]
    return [str(meta["scene_token"]) for meta in metas]
