#!/usr/bin/env python3
"""Select a whitelisted model/dataset entrypoint; dry-run unless --execute."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_formal_configs import issues


from occstress.adapters.forecasting import command_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--dataset', choices=('nuscenes', 'waymo', 'carla'))
    parser.add_argument('--python', help='Python executable in this model\'s isolated environment')
    parser.add_argument('--manifest', type=Path, help='Explicit native tasks with verified resume receipts')
    parser.add_argument('--stage', choices=('world', 'tokenizer'), default='world')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('native_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    errors = issues()
    if errors:
        raise SystemExit('\n'.join(errors))
    if args.manifest:
        if args.model or args.dataset or args.python or args.native_args or args.stage != 'world':
            parser.error('--manifest cannot be combined with single-task arguments')
        from tools.run_manifest import run_manifest
        run_manifest(args.manifest, ROOT, args.execute)
        return
    if not all((args.model, args.dataset, args.python)):
        parser.error('--model, --dataset and --python are required without --manifest')
    native = args.native_args[1:] if args.native_args[:1] == ['--'] else args.native_args
    try:
        directory, command, env = command_plan(args.model, args.dataset, args.python, native, args.stage)
    except ValueError as error:
        parser.error(str(error))
    print('Working directory: ' + str(directory))
    print(shlex.join(command))
    print('PYTHONPATH=' + env['PYTHONPATH'])
    if args.execute:
        subprocess.run(command, cwd=directory, env=env, check=True)
    else:
        print('Dry run only; add --execute after the documented reproduction gates.')


if __name__ == '__main__':
    main()
