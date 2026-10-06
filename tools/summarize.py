#!/usr/bin/env python3
"""Normalize native counts, merge disjoint shards, or summarize a declared suite."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occstress.adapters.native_results import extract_counts
from occstress.protocols.temporal import HORIZONS
from occstress.results.io import load_json, sha256_file, write_json
from occstress.results.schema import build_result
from occstress.results.summary import merge_shards, summarize_suite


def normalize(native, metadata, native_sha256):
    if metadata.get('future_horizons_seconds') != HORIZONS:
        raise ValueError('Explicit actual F6 horizon declaration required')
    if native.get('future_horizons_seconds', HORIZONS) != HORIZONS:
        raise ValueError('Native future horizon mismatch')
    if native.get('evaluated_records') != len(metadata['anchor_ids']):
        raise ValueError('Native evaluated-record count differs from declared anchor IDs')
    return build_result(metadata['identity'], metadata['anchor_ids'], extract_counts(native),
                        {'native_sha256': native_sha256,
                         'anchor_provenance': 'explicit metadata supplied by evaluator'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    normal = subs.add_parser('normalize')
    normal.add_argument('--native', required=True, type=Path)
    normal.add_argument('--metadata', required=True, type=Path)
    normal.add_argument('--output', required=True, type=Path)
    merge = subs.add_parser('merge')
    merge.add_argument('inputs', nargs='+', type=Path)
    merge.add_argument('--output', required=True, type=Path)
    suite = subs.add_parser('suite')
    suite.add_argument('--suite', required=True, type=Path)
    suite.add_argument('inputs', nargs='+', type=Path)
    suite.add_argument('--output', required=True, type=Path)
    suite.add_argument('--csv', type=Path)
    suite.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    inputs = [args.native, args.metadata] if args.action == 'normalize' else args.inputs
    protected = inputs + ([args.suite] if args.action == 'suite' else [])
    if args.output.resolve() in {p.resolve() for p in protected}:
        parser.error('Output must not overwrite an input')
    if args.action == 'normalize':
        result = normalize(load_json(args.native), load_json(args.metadata), sha256_file(args.native))
    elif args.action == 'merge':
        result = merge_shards([load_json(p) for p in inputs])
    else:
        result = summarize_suite(load_json(args.suite), [load_json(p) for p in inputs], args.allow_partial)
    write_json(args.output, result)
    if args.action == 'suite' and args.csv:
        if args.csv.resolve() in {p.resolve() for p in protected + [args.output]}:
            parser.error('CSV path must not overwrite JSON inputs/output')
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id', 'kind', 'family', 'severity', 'pattern',
                                    'anchors', 'complete', 'miou', 'iou', 'miou_drop_pp',
                                    'miou_relative_drop_percent'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(result['protocols'])
    print(args.output)


if __name__ == '__main__':
    main()
