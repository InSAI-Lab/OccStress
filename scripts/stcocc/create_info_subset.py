#!/usr/bin/env python3
import argparse
import os

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description="Create a small STCOcc info subset for smoke tests.")
    parser.add_argument("--src", required=True, help="Source ann/info pickle path.")
    parser.add_argument("--dst", required=True, help="Output subset pickle path.")
    parser.add_argument("--limit", type=int, default=8, help="Number of infos to keep.")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = mmcv.load(args.src)

    if isinstance(payload, dict):
        if "infos" in payload and isinstance(payload["infos"], list):
            payload = dict(payload)
            payload["infos"] = payload["infos"][:args.limit]
        else:
            raise TypeError("Expected a dict with an 'infos' list.")
    elif isinstance(payload, list):
        payload = payload[:args.limit]
    else:
        raise TypeError(f"Unsupported payload type: {type(payload).__name__}")

    mmcv.mkdir_or_exist(os.path.dirname(args.dst))
    mmcv.dump(payload, args.dst)
    kept = len(payload["infos"]) if isinstance(payload, dict) else len(payload)
    print(f"saved {kept} infos to {args.dst}")


if __name__ == "__main__":
    main()
