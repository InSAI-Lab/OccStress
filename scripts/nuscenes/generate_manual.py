#!/usr/bin/env python3
"""Select a nuScenes manual generator, preserving its native command-line flags."""
import argparse
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('family', choices=('semantic', 'dropout', 'hole', 'traffic', 'misalignment'))
    args, native = parser.parse_known_args()
    name = ('scripts.nuscenes.generate_misalignment_subset' if args.family == 'misalignment'
            else 'occstress.corruptions.' + args.family)
    sys.argv = [sys.argv[0], *native]
    import_module(name).main()


if __name__ == '__main__':
    main()
