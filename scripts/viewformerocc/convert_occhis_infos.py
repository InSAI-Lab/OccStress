#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path


def default_root() -> Path:
    env_root = os.environ.get("OCCSTRESS_CODE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def convert_scene_infos(scene_name, items):
    converted = []
    total = len(items)
    for idx, item in enumerate(items):
        token = item["token"]
        out = dict(item)
        out["scene_token"] = scene_name
        out["occ_gt_path"] = f"gts/{scene_name}/{token}/labels.npz"
        out["prev"] = items[idx - 1]["token"] if idx > 0 else ""
        out["next"] = items[idx + 1]["token"] if idx + 1 < total else ""
        converted.append(out)
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert OccHis temporal nuscenes infos to ViewFormer-Occ format."
    )
    parser.add_argument("--root", default=str(default_root()))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    data_root = root / "data" / "nuscenes"

    with input_path.open("rb") as f:
        payload = pickle.load(f)

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    scene_infos = payload.get("infos", payload) if isinstance(payload, dict) else payload

    flat_infos = []
    missing_occ = []

    for scene_name in sorted(scene_infos):
        items = scene_infos[scene_name]
        converted = convert_scene_infos(scene_name, items)
        for entry in converted:
            occ_path = data_root / entry["occ_gt_path"]
            if not occ_path.exists():
                missing_occ.append(str(occ_path))
        flat_infos.extend(converted)

    flat_infos.sort(key=lambda x: x["timestamp"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"infos": flat_infos, "metadata": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"scenes={len(scene_infos)}")
    print(f"infos={len(flat_infos)}")
    print(f"missing_occ_paths={len(missing_occ)}")
    if missing_occ:
        print(f"first_missing={missing_occ[0]}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
