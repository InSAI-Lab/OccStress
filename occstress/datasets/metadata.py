"""Control metadata loading shared by isolated model adapters."""
import json
import pickle
from pathlib import Path

from .paths import resolve_occstress_path


def load_metadata(path, *, trusted_pickle=False, occstress_root=None):
    # MMCV's FileClient passes an open local file instead of its path.
    if hasattr(path, 'read'):
        name = getattr(path, 'name', None)
        if not isinstance(name, (str, Path)):
            raise ValueError('Metadata streams must have a filename with a known extension')
        suffix = Path(name).suffix
        if suffix == '.json':
            return json.load(path)
        if suffix == '.pkl' and trusted_pickle:
            return pickle.load(path)
        raise ValueError('Use controls.json, or explicitly trust an official local info PKL')
    path = resolve_occstress_path(path, occstress_root=occstress_root)
    if path.suffix == '.json':
        return json.loads(path.read_text())
    if path.suffix == '.pkl' and trusted_pickle:
        with path.open('rb') as handle:
            return pickle.load(handle)
    raise ValueError('Use controls.json, or explicitly trust an official local info PKL')
