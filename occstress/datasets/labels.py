"""Map raw dataset labels using an explicit, versioned mapping."""
import numpy as np


def remap_labels(labels, mapping, ignore_label=255):
    labels = np.asarray(labels)
    if labels.dtype.kind not in 'iu':
        raise ValueError('Labels must be integers')
    mapping = {int(k): int(v) for k, v in mapping.items()}
    unknown = set(np.unique(labels).tolist()) - set(mapping) - {ignore_label}
    if unknown:
        raise ValueError(f'Unmapped raw labels: {sorted(unknown)}')
    if any(v < 0 or v > 17 for v in mapping.values()):
        raise ValueError('Expected canonical Occ3D-18 labels')
    result = np.full(labels.shape, ignore_label, dtype=np.uint8)
    for source, target in mapping.items():
        result[labels == source] = target
    return result
