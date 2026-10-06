#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Deterministic RoboBEV-style camera corruptions for five-view Waymo."""

from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from PIL import Image, ImageOps


SEVERITY = {
    "CameraCrash": {"easy": 2, "mid": 4, "hard": 5},
    "FrameLost": {"easy": 2, "mid": 4, "hard": 5},
    "MotionBlur": {"easy": 2, "mid": 4, "hard": 5},
    "ColorQuant": {"easy": 1, "mid": 2, "hard": 3},
    "Brightness": {"easy": 2, "mid": 4, "hard": 5},
    "LowLight": {"easy": 2, "mid": 3, "hard": 4},
    "Fog": {"easy": 2, "mid": 4, "hard": 5},
    "Snow": {"easy": 1, "mid": 2, "hard": 3},
}

IMAGECORRUPTIONS_NAMES = {
    "MotionBlur": "motion_blur",
    "Brightness": "brightness",
    "Fog": "fog",
    "Snow": "snow",
}

SLOW_IMAGE_CORRUPTIONS = {
    "Brightness",
    "Fog",
    "LowLight",
    "MotionBlur",
    "Snow",
}

WAYMO_CAMERA_NAMES = (
    "FRONT",
    "FRONT_LEFT",
    "SIDE_LEFT",
    "FRONT_RIGHT",
    "SIDE_RIGHT",
)


class Executor(Protocol):
    def submit(self, fn, /, *args, **kwargs): ...


def stable_seed(sample_key: str, corruption: str, severity: str) -> int:
    value = f"{sample_key}\0{corruption}\0{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def camera_crash_draw(severity: str) -> list[int]:
    amount = SEVERITY["CameraCrash"][severity]
    seed = stable_seed("protocol_fixed", "CameraCrash", severity)
    return np.random.RandomState(seed).choice(5, size=amount).tolist()


@contextmanager
def numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def low_light(image: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    brightness = [0.60, 0.50, 0.40, 0.30, 0.20][severity]
    image_float = image.astype(np.float64) / 255.0
    minimum = image_float.min()
    maximum = image_float.max()
    if maximum > minimum:
        scaled = ((image_float - minimum) / (maximum - minimum)) ** 2
    else:
        scaled = np.zeros_like(image_float)
    scaled = scaled * brightness

    poisson_scale = 10 * [60, 25, 12, 5, 3][severity]
    noisy = np.clip(
        rng.poisson(scaled * poisson_scale) / poisson_scale,
        0,
        1,
    )
    gaussian_scale = 0.1 * [0.08, 0.12, 0.18, 0.26, 0.38][severity]
    noisy = np.clip(
        noisy + rng.normal(size=noisy.shape, scale=gaussian_scale),
        0,
        1,
    )
    return np.uint8(noisy * 255)


def corrupt_view(
    image: np.ndarray,
    corruption: str,
    amount: int,
    seed: int,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if corruption == "ColorQuant":
        bits = 5 - amount
        return np.asarray(ImageOps.posterize(Image.fromarray(image), bits))
    if corruption == "LowLight":
        return low_light(image, amount, np.random.RandomState(seed))

    from imagecorruptions import corrupt

    with numpy_seed(seed):
        return corrupt(
            image,
            corruption_name=IMAGECORRUPTIONS_NAMES[corruption],
            severity=amount,
        )


def corrupt_rgb_file(
    filename: str,
    corruption: str,
    amount: int,
    seed: int,
) -> np.ndarray:
    with Image.open(filename) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return corrupt_view(rgb, corruption, amount, seed)


def validated_views(images: Iterable[np.ndarray]) -> list[np.ndarray]:
    views = [np.asarray(image, dtype=np.uint8).copy() for image in images]
    if len(views) != 5:
        raise ValueError(f"Waymo corruption expects 5 views, got {len(views)}")
    if any(image.ndim != 3 or image.shape[2] != 3 for image in views):
        raise ValueError("Waymo corruption expects five HxWx3 images")
    return views


def corrupt_views(
    images: Iterable[np.ndarray],
    corruption: str,
    severity: str,
    sample_key: str,
) -> list[np.ndarray]:
    if corruption not in SEVERITY:
        raise ValueError(f"Unsupported corruption: {corruption}")
    if severity not in SEVERITY[corruption]:
        raise ValueError(f"Unsupported severity: {severity}")

    views = validated_views(images)

    amount = SEVERITY[corruption][severity]
    if corruption == "CameraCrash":
        # RoboBEV samples camera indices with replacement and keeps the sampled
        # subset fixed for the protocol. Preserve that distribution on 5 views.
        for camera_idx in camera_crash_draw(severity):
            views[camera_idx].fill(0)
        return views

    seed = stable_seed(sample_key, corruption, severity)
    rng = np.random.RandomState(seed)
    if corruption == "FrameLost":
        for camera_idx in range(5):
            if rng.rand() < amount / 6.0:
                views[camera_idx].fill(0)
        return views

    outputs = []
    for camera_idx, image in enumerate(views):
        view_seed = seed ^ ((camera_idx + 1) * 0x9E3779B1)
        outputs.append(
            corrupt_view(
                image,
                corruption,
                amount,
                view_seed & 0xFFFFFFFF,
            )
        )
    return outputs


def corrupt_views_parallel(
    images: Iterable[np.ndarray],
    corruption: str,
    severity: str,
    sample_key: str,
    executor: Executor | None,
) -> list[np.ndarray]:
    if executor is None or corruption not in SLOW_IMAGE_CORRUPTIONS:
        return corrupt_views(images, corruption, severity, sample_key)
    if corruption not in SEVERITY:
        raise ValueError(f"Unsupported corruption: {corruption}")
    if severity not in SEVERITY[corruption]:
        raise ValueError(f"Unsupported severity: {severity}")

    views = validated_views(images)
    amount = SEVERITY[corruption][severity]
    seed = stable_seed(sample_key, corruption, severity)
    futures = []
    for camera_idx, image in enumerate(views):
        view_seed = seed ^ ((camera_idx + 1) * 0x9E3779B1)
        futures.append(
            executor.submit(
                corrupt_view,
                image,
                corruption,
                amount,
                view_seed & 0xFFFFFFFF,
            )
        )
    return [future.result() for future in futures]


def corrupt_waymo_view_file(
    filename: str,
    corruption: str,
    amount: int,
    seed: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    import cv2

    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"failed to read source image: {filename}")
    corrupted = corrupt_view(image, corruption, amount, seed)
    padded_height = target_height * 2
    padded_width = target_width * 2
    if corrupted.shape[0] != padded_height:
        if (
            corrupted.shape[0] > padded_height
            or corrupted.shape[1] > padded_width
        ):
            raise ValueError(
                "source image exceeds the CVT Waymo padding canvas: "
                f"{corrupted.shape} vs {(padded_height, padded_width)}"
            )
        padded = np.zeros((padded_height, padded_width, 3))
        padded[: corrupted.shape[0], : corrupted.shape[1]] = corrupted
        corrupted = padded
    resized = cv2.resize(
        corrupted.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.ascontiguousarray(resized.transpose(2, 0, 1))


def corrupt_waymo_view_file_into_memmap(
    filename: str,
    corruption: str,
    amount: int,
    seed: int,
    target_height: int,
    target_width: int,
    output_path: str,
    camera_idx: int,
) -> int:
    output = np.memmap(
        output_path,
        mode="r+",
        dtype=np.float32,
        shape=(5, 3, target_height, target_width),
    )
    output[camera_idx] = corrupt_waymo_view_file(
        filename,
        corruption,
        amount,
        seed,
        target_height,
        target_width,
    )
    del output
    return camera_idx


def corrupt_waymo_frame_files_into_memmap(
    filenames: Iterable[str],
    corruption: str,
    severity: str,
    sample_key: str,
    target_height: int,
    target_width: int,
    output_path: str,
    output_shape: tuple[int, int, int, int, int],
    output_index: int,
) -> int:
    paths = list(filenames)
    if len(paths) != 5:
        raise ValueError(f"Waymo corruption expects 5 views, got {len(paths)}")
    amount = SEVERITY[corruption][severity]
    seed = stable_seed(sample_key, corruption, severity)
    output = np.memmap(
        output_path,
        mode="r+",
        dtype=np.float32,
        shape=output_shape,
    )
    for camera_idx, filename in enumerate(paths):
        view_seed = seed ^ ((camera_idx + 1) * 0x9E3779B1)
        output[output_index, camera_idx] = corrupt_waymo_view_file(
            filename,
            corruption,
            amount,
            view_seed & 0xFFFFFFFF,
            target_height,
            target_width,
        )
    del output
    return output_index


def corrupt_waymo_files_parallel(
    filenames: Iterable[str],
    corruption: str,
    severity: str,
    sample_key: str,
    target_height: int,
    target_width: int,
    executor: Executor,
    output: np.ndarray | None = None,
    output_path: str | None = None,
) -> np.ndarray:
    paths = list(filenames)
    if len(paths) != 5:
        raise ValueError(f"Waymo corruption expects 5 views, got {len(paths)}")
    if corruption not in SLOW_IMAGE_CORRUPTIONS:
        raise ValueError(f"file-parallel corruption is not enabled for {corruption}")
    amount = SEVERITY[corruption][severity]
    seed = stable_seed(sample_key, corruption, severity)
    if (output is None) != (output_path is None):
        raise ValueError("output and output_path must be provided together")
    futures = []
    for camera_idx, filename in enumerate(paths):
        view_seed = seed ^ ((camera_idx + 1) * 0x9E3779B1)
        if output is None:
            futures.append(
                executor.submit(
                    corrupt_waymo_view_file,
                    filename,
                    corruption,
                    amount,
                    view_seed & 0xFFFFFFFF,
                    target_height,
                    target_width,
                )
            )
        else:
            futures.append(
                executor.submit(
                    corrupt_waymo_view_file_into_memmap,
                    filename,
                    corruption,
                    amount,
                    view_seed & 0xFFFFFFFF,
                    target_height,
                    target_width,
                    output_path,
                    camera_idx,
                )
            )
    results = [future.result() for future in futures]
    if output is not None:
        if sorted(results) != list(range(5)):
            raise RuntimeError("incomplete Waymo corruption memmap output")
        return output
    return np.stack(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("corruption", choices=sorted(SEVERITY))
    parser.add_argument("severity", choices=("easy", "mid", "hard"))
    parser.add_argument("sample_key")
    parser.add_argument("images", nargs=5, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = [np.asarray(Image.open(path).convert("RGB")) for path in args.images]
    outputs = corrupt_views(
        images,
        args.corruption,
        args.severity,
        args.sample_key,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source, output in zip(args.images, outputs):
        Image.fromarray(output).save(args.output_dir / source.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
