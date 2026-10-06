#!/usr/bin/env python3
"""Check a canonical protocol without opening voxel assets."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occstress.protocols.validation import load_records, validate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('--expected-anchors', type=int)
    parser.add_argument('--trust-pickle', action='store_true', help='Only for your trusted local protocol files')
    args = parser.parse_args()
    print(json.dumps(validate_records(load_records(args.protocol, trusted_pickle=args.trust_pickle),
                                      args.expected_anchors), indent=2))


if __name__ == '__main__':
    main()
