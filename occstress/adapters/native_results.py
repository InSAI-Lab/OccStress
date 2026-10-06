"""Extract sufficient statistics; never reinterpret a rounded native score."""
import numpy as np

from occstress.metrics.occupancy import from_confusion, integer_counts, validate_counts


def extract_counts(payload):
    if payload.get('status', 'success') not in {'success', 'completed', 'ok'}:
        raise ValueError('Native evaluation did not succeed')
    counts = payload.get('aggregate_counts')
    if counts is None:
        counts = payload.get('metrics', {}).get('_aggregate_counts')
    if counts is None:
        counts = payload.get('counts')
    if counts is None:
        counts = payload.get('raw_confusion')
    if counts is None:
        counts = payload
    if 'semantic_pred' in counts:
        counts = dict(counts)
        # COME exports the free-class column as well; canonical counts keep 0..16.
        for key in ('semantic_pred', 'semantic_gt', 'semantic_tp'):
            if np.shape(counts[key]) == (6, 18):
                counts[key] = integer_counts(counts[key], (6, 18))[:, :17]
        return validate_counts(counts)
    if 'semantic_confusion' in counts:
        binary = counts.get('binary_confusion', counts.get('occupancy_confusion'))
        return from_confusion(counts['semantic_confusion'], binary)
    if 'semantic' in counts and 'binary' in counts:
        return from_confusion(counts['semantic'], counts['binary'])
    raise ValueError('Raw F6 counts required; metrics-only native logs are not convertible')
