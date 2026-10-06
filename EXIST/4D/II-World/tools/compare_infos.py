#!/usr/bin/env python3
import argparse
import collections
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple


def load_infos(path: str) -> Tuple[List[Dict[str, Any]], Any]:
    """Load a pickle file and return (infos_list, raw_obj)."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        infos = raw.get("infos") or raw.get("data") or raw.get("samples")
    else:
        infos = raw
    if not isinstance(infos, list):
        raise ValueError(f"Unsupported format in {path}: expected list, got {type(infos)}")
    return infos, raw


def extract_token(info: Dict[str, Any]) -> Optional[str]:
    for key in ("token", "sample_token", "lidar_token", "scene_token"):
        if key in info:
            return info[key]
    return None


def class_histogram(infos: Sequence[Dict[str, Any]]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for info in infos:
        names = info.get("gt_names")
        if names:
            counter.update(names)
    return counter


def compare_orders(tokens_a: Sequence[Optional[str]], tokens_b: Sequence[Optional[str]]) -> List[int]:
    limit = min(len(tokens_a), len(tokens_b))
    return [i for i in range(limit) if tokens_a[i] != tokens_b[i]]


def summarize_keys(infos: Sequence[Dict[str, Any]]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for info in infos:
        counter.update(info.keys())
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two nuScenes info pickle files.")
    parser.add_argument("--file-a", default="data/nuscenes_infos_train_temporal_v3_scene.pkl", help="First pickle path.")
    parser.add_argument("--file-b", default="data/nuscenes/world-nuscenes_infos_train.pkl", help="Second pickle path.")
    parser.add_argument("--top-n", type=int, default=20, help="How many mismatch indices to print.")
    args = parser.parse_args()

    file_a = os.path.expanduser(args.file_a)
    file_b = os.path.expanduser(args.file_b)

    infos_a, raw_a = load_infos(file_a)
    infos_b, raw_b = load_infos(file_b)

    print(f"A: {file_a}\n  entries: {len(infos_a)}")
    print(f"B: {file_b}\n  entries: {len(infos_b)}")

    keys_a = summarize_keys(infos_a)
    keys_b = summarize_keys(infos_b)
    print("\nTop keys (A):", keys_a.most_common(10))
    print("Top keys (B):", keys_b.most_common(10))
    missing_in_b = [k for k in keys_a if k not in keys_b]
    missing_in_a = [k for k in keys_b if k not in keys_a]
    if missing_in_a or missing_in_b:
        print("\nKeys only in A:", missing_in_b)
        print("Keys only in B:", missing_in_a)

    tokens_a = [extract_token(info) for info in infos_a]
    tokens_b = [extract_token(info) for info in infos_b]
    set_diff_a = set(tokens_a) - set(tokens_b)
    set_diff_b = set(tokens_b) - set(tokens_a)
    print(f"\nToken set delta: A-only {len(set_diff_a)}, B-only {len(set_diff_b)}")
    if set_diff_a:
        print("  sample A-only token examples:", list(set_diff_a)[:5])
    if set_diff_b:
        print("  sample B-only token examples:", list(set_diff_b)[:5])

    order_mismatch = compare_orders(tokens_a, tokens_b)
    print(f"\nOrder mismatches (up to shared length): {len(order_mismatch)}")
    if order_mismatch:
        show = order_mismatch[: args.top_n]
        print("  first mismatch indices:", show)
        for idx in show:
            print(f"    idx {idx}: A={tokens_a[idx]} | B={tokens_b[idx]}")

    if len(infos_a) != len(infos_b):
        print("\nLength differs, order comparison stops at shorter list.")

    hist_a = class_histogram(infos_a)
    hist_b = class_histogram(infos_b)
    if hist_a and hist_b:
        merged_classes = sorted(set(hist_a.keys()) | set(hist_b.keys()))
        print("\nClass count differences (A-B):")
        for cls in merged_classes:
            diff = hist_a.get(cls, 0) - hist_b.get(cls, 0)
            if diff != 0:
                print(f"  {cls}: {hist_a.get(cls, 0)} (A) vs {hist_b.get(cls, 0)} (B) | delta {diff:+}")

    print("\nDone.")


if __name__ == "__main__":
    main()
