#!/usr/bin/env python3
"""Inspect code and explicit resource placeholders without GPU dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occstress.distribution import load_distribution, public_source_issues, withheld


def resource_issues(manifest):
    issues = []
    if manifest.get('schema_version') != 1:
        issues.append('Unsupported resource schema_version')
    repository = manifest.get('dataset_repository_url')
    if repository is not None:
        url = urlparse(repository)
        if (url.scheme != 'https' or url.hostname != 'huggingface.co'
                or not re.fullmatch(r'/datasets/[^/]+/[^/]+', url.path)):
            issues.append('Invalid dataset_repository_url')
    ids = set()
    for kind, provider in [('datasets', 'huggingface'), ('checkpoints', 'google_drive')]:
        if not manifest.get(kind):
            issues.append(f'Missing resource group: {kind}')
        for item in manifest.get(kind, []):
            key = item.get('id')
            if not key or key in ids:
                issues.append(f'Missing or duplicate resource id: {key}')
            ids.add(key)
            actual_provider = item.get('provider')
            providers = {'huggingface'} if kind == 'datasets' else {'google_drive', 'upstream'}
            if actual_provider not in providers:
                issues.append(f'{key}: unsupported provider')
            local = Path(item.get('local_path', ''))
            if not item.get('local_path') or local.is_absolute() or '..' in local.parts:
                issues.append(f'{key}: local_path must stay inside the checkout')
            if item.get('expected_sha256') is not None and not re.fullmatch('[0-9a-f]{64}', item['expected_sha256']):
                issues.append(f'{key}: invalid expected checkpoint SHA256')
            if item.get('status') == 'publishing':
                url = urlparse(item.get('url') or '')
                if (kind != 'datasets' or url.scheme != 'https'
                        or url.hostname != 'huggingface.co' or not url.path.endswith('/packages.json')):
                    issues.append(f'{key}: publishing requires a Hugging Face package index')
                if item.get('sha256') is not None:
                    issues.append(f'{key}: a changing publication index has no frozen checksum')
            elif item.get('status') == 'pending':
                if item.get('url') is not None or item.get('sha256') is not None:
                    issues.append(f'{key}: pending resources must have null URL and checksum')
            elif item.get('status') == 'external_reference':
                url = urlparse(item.get('url') or '')
                if kind != 'checkpoints' or actual_provider != 'upstream':
                    issues.append(f'{key}: external_reference requires an upstream checkpoint')
                if (url.scheme != 'https' or not url.hostname or url.username or url.password
                        or url.hostname not in {'drive.google.com', 'huggingface.co', 'cloud.tsinghua.edu.cn', 'github.com'}):
                    issues.append(f'{key}: invalid official reference URL')
                if item.get('sha256') is not None or item.get('redistribution_review') != 'not_mirrored':
                    issues.append(f'{key}: a reference is not a verified or cleared mirror')
                if item.get('identity_status') != 'exact_benchmark_file_not_verified':
                    issues.append(f'{key}: external reference must disclose unverified identity')
            elif item.get('status') == 'available':
                url = urlparse(item.get('url') or '')
                hosts = ({'huggingface.co'} if kind == 'datasets' else
                         {'drive.google.com', 'huggingface.co', 'cloud.tsinghua.edu.cn'}
                         if actual_provider == 'upstream' else {'drive.google.com'})
                if url.scheme != 'https' or url.hostname not in hosts:
                    issues.append(f'{key}: invalid provider URL')
                if not re.fullmatch('[0-9a-f]{64}', item.get('sha256') or ''):
                    issues.append(f'{key}: supply an archive or checksum-manifest SHA256')
                if kind == 'datasets':
                    revision = item.get('revision') or ''
                    if (not re.fullmatch('[0-9a-f]{40}', revision)
                            or url.path != f'/datasets/insailab/OccStress/resolve/{revision}/packages.json'):
                        issues.append(f'{key}: available dataset must pin its verified package-index revision')
                if kind == 'checkpoints':
                    if actual_provider == 'upstream':
                        if (item.get('redistribution_review') != 'not_mirrored'
                                or item.get('identity_status') != 'exact_benchmark_file_verified'):
                            issues.append(f'{key}: official download identity must be verified without claiming mirror rights')
                    elif item.get('redistribution_review') != 'cleared':
                        issues.append(f'{key}: checkpoint redistribution review is not cleared')
            else:
                issues.append(f'{key}: unsupported publication status')
    return issues


def inspect(root=ROOT):
    root = Path(root)
    resources = json.loads((root / 'configs/resources.json').read_text())
    registry = json.loads((root / 'configs/methods.json').read_text())
    distribution = load_distribution(root)
    issues = resource_issues(resources)
    if distribution['profile'] in {'public_candidate', 'full_source_candidate'}:
        issues.extend(public_source_issues(root, distribution))
    if (root / 'configs/formal_configs.lock.json').exists():
        from tools.check_formal_configs import issues as formal_issues
        issues.extend(formal_issues(root))
    for method in registry['methods']:
        if withheld(root, method['directory']):
            continue
        paths = method['entrypoints'] + ([method['configuration']] if method['configuration'] else [])
        for relative in paths:
            path = root / method['directory'] / relative
            if not path.is_file():
                issues.append(f'Missing {method["id"]} source entry: {relative}')
    pending = [item['id'] for group in ('datasets', 'checkpoints')
               for item in resources[group] if item['status'] != 'available']
    contract = json.loads((root / 'configs/evaluation_contract.json').read_text())
    return {'issues': issues, 'pending_resources': pending,
            'code_checks_passed': not issues,
            'runtime_gate': contract.get('runtime_gate', {}),
            'dataset_package_index_url': resources.get('dataset_package_index_url'),
            'external_checkpoint_references': [item['id'] for item in resources['checkpoints']
                                               if item['status'] == 'external_reference'],
            'dataset_repository_url': resources.get('dataset_repository_url'),
            'distribution_profile': distribution['profile'],
            'pending_source_permissions': [item['id'] for item in distribution['pending_components']],
            'withheld_source_components': [item['id'] for item in distribution['pending_components']
                                            if withheld(root, item['directory'])],
            'public_redistribution_ready': False,
            'methods': [{key: method[key] for key in ('id', 'status', 'tracks', 'source_status')}
                        for method in registry['methods']],
            'gpu_reproduction_verified': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--require-downloads', action='store_true',
                   help='Fail if any download is still a placeholder (no network request).')
    args = p.parse_args()
    report = inspect(args.root)
    print(json.dumps(report, indent=2))
    return int(bool(report['issues']) or
               (args.require_downloads and bool(report['pending_resources'])))


if __name__ == '__main__':
    raise SystemExit(main())
