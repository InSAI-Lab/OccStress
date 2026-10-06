#!/usr/bin/env python3
"""Build a new, history-free source candidate without mutating the maintainer tree."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occstress.distribution import (
    component_for_path, load_distribution, placeholder_text, public_source_issues, source_status,
)
from tools.check_formal_configs import make_lock
from tools.check_release import (
    FORBIDDEN_DIR_NAMES, FORBIDDEN_FILE_SUFFIXES, private_release_path,
)
from tools.update_release_metadata import derived_methods


def omit(relative, manifest):
    path = Path(relative)
    if private_release_path(path):
        return True
    if manifest['profile'] != 'full_source_candidate' and component_for_path(path.as_posix(), manifest):
        return True
    if any(part in FORBIDDEN_DIR_NAMES | {'.git', '.pytest_cache', 'build', 'dist'}
           or part.endswith('.egg-info') for part in path.parts):
        return True
    if path.parts[0] == 'data' and path.parts[1:] != ('README.md',):
        return True
    return path.suffix in FORBIDDEN_FILE_SUFFIXES or path.name.endswith(('.tar.gz', '.tar.zst'))


def copy_source(source, destination, manifest):
    """Never follow directory symlinks or silently restore a withheld alias."""
    copied = 0
    for directory, dirs, files in os.walk(source, followlinks=False):
        directory = Path(directory)
        for name in list(dirs):
            path = directory / name
            relative = path.relative_to(source)
            if path.is_symlink():
                dirs.remove(name)
                files.append(name)
            elif omit(relative, manifest) and relative.as_posix() != 'data':
                dirs.remove(name)
        for name in files:
            path = directory / name
            relative = path.relative_to(source)
            if omit(relative, manifest):
                continue
            if path.is_symlink() and name == 'data' and path.resolve() != source / 'data':
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                link = path.readlink()
                resolved = path.resolve(strict=True)
                if link.is_absolute() or not resolved.is_relative_to(source):
                    raise ValueError('External symlink cannot enter a source candidate: ' + str(relative))
                if omit(resolved.relative_to(source), manifest) and resolved != source / 'data':
                    raise ValueError('Symlink aliases omitted content: ' + str(relative))
                target.symlink_to(link)
            elif path.is_file():
                shutil.copy2(path, target)
            else:
                raise ValueError('Not a regular source file: ' + str(relative))
            copied += 1
    return copied


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def build(source, destination, profile='full_source_candidate'):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Candidate and maintainer tree must be separate, non-nested directories')
    if destination.exists():
        raise FileExistsError('Candidate output already exists: ' + str(destination))
    manifest = load_distribution(source)
    if profile not in {'public_candidate', 'full_source_candidate'}:
        raise ValueError('Unknown source candidate profile: ' + profile)
    manifest['profile'] = profile
    manifest['public_redistribution_ready'] = False
    if profile == 'full_source_candidate':
        errors = public_source_issues(source, manifest)
        if errors:
            raise ValueError('\n'.join(errors))
    destination.mkdir(parents=True, exist_ok=False)
    copied = copy_source(source, destination, manifest)
    write_json(destination / 'configs/distribution.json', manifest)
    for component in manifest['pending_components'] if profile == 'public_candidate' else []:
        directory = destination / component['directory']
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('README.md', 'README_OCCSTRESS.md'):
            (directory / name).write_text(placeholder_text(component))
    ledger_path = destination / 'docs/source-imports.json'
    ledger = json.loads(ledger_path.read_text())
    included = [row for row in ledger['files'] if not omit(row['path'], manifest)]
    ledger['withheld_file_count'] = ledger.get('withheld_file_count', 0) + len(ledger['files']) - len(included)
    ledger['files'] = included
    ledger['distribution_profile'] = profile
    write_json(ledger_path, ledger)
    for relative, group in [('environments/profiles.json', 'models'),
                            ('configs/upstream_methods.json', 'sources')]:
        path = destination / relative
        registry = json.loads(path.read_text())
        entries = registry[group].values() if isinstance(registry[group], dict) else registry[group]
        for spec in entries:
            spec['source_status'] = source_status(destination, spec['directory'])
        write_json(path, registry)
    write_json(destination / 'configs/methods.json', derived_methods(destination))
    write_json(destination / 'configs/formal_configs.lock.json', make_lock(destination))
    errors = public_source_issues(destination)
    if errors:
        raise ValueError('\n'.join(errors))
    return {'profile': manifest['profile'], 'copied_files': copied,
            'pending_components': [c['id'] for c in manifest['pending_components']],
            'git_history_included': False, 'public_redistribution_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--profile', choices=('public_candidate', 'full_source_candidate'),
                        default='full_source_candidate',
                        help='Full source retains unresolved license records; it does not grant permission.')
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory only; existing outputs are never overwritten.')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.profile), indent=2))


if __name__ == '__main__':
    main()
