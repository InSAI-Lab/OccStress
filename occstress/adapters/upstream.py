"""Upstream source descriptors without importing competing model registries."""
import json

from occstress.datasets.paths import repo_root


def source_spec(source_id, root=None):
    root = repo_root(root)
    sources = json.loads((root / 'configs/upstream_methods.json').read_text())['sources']
    for spec in sources:
        if spec['id'] == source_id:
            return spec
    raise ValueError(f'Unknown declared upstream source: {source_id}')
