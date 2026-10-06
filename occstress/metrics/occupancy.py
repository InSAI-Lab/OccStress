"""Count-based F6 scores, with GT rows and prediction columns in histograms."""
import numpy as np

from occstress.protocols.temporal import HORIZONS

METRIC_POLICY = 'occstress-present-gt-v1'
COUNT_SHAPES = {f'semantic_{key}': (6, 17) for key in ('pred', 'gt', 'tp')}
COUNT_SHAPES.update({f'binary_{key}': (6,) for key in ('pred', 'gt', 'tp')})


def integer_counts(value, shape):
    value = np.asarray(value)
    if (value.shape != shape or value.dtype.kind not in 'iuf' or
            not np.isfinite(value).all() or (value < 0).any() or
            (value > 10**15).any() or (value != np.floor(value)).any()):
        raise ValueError(f'Expected finite nonnegative integer counts of shape {shape}')
    return value.astype(np.int64)


def validate_counts(counts):
    if set(counts) != set(COUNT_SHAPES):
        raise ValueError('Missing or unexpected sufficient-count keys')
    result = {key: integer_counts(counts[key], shape) for key, shape in COUNT_SHAPES.items()}
    for prefix in ('semantic', 'binary'):
        tp, pred, gt = (result[f'{prefix}_{key}'] for key in ('tp', 'pred', 'gt'))
        if ((tp > pred) | (tp > gt)).any():
            raise ValueError('True positives exceed prediction/GT totals')
    if not np.array_equal(result['semantic_gt'].sum(1), result['binary_gt']):
        raise ValueError('Semantic GT totals and binary occupied totals disagree')
    if not np.array_equal(result['semantic_pred'].sum(1), result['binary_pred']):
        raise ValueError('Semantic prediction totals and binary occupied totals disagree')
    if (result['semantic_tp'].sum(1) > result['binary_tp']).any():
        raise ValueError('Semantic true positives exceed binary true positives')
    return result


def from_confusion(semantic, binary):
    semantic = integer_counts(semantic, (6, 18, 18))
    binary = integer_counts(binary, (6, 2, 2))
    if not np.array_equal(semantic.sum((1, 2)), binary.sum((1, 2))):
        raise ValueError('Semantic/binary histogram voxel counts disagree')
    collapsed = np.zeros_like(binary)
    collapsed[:, 0, 0] = semantic[:, 17, 17]
    collapsed[:, 0, 1] = semantic[:, 17, :17].sum(1)
    collapsed[:, 1, 0] = semantic[:, :17, 17].sum(1)
    collapsed[:, 1, 1] = semantic[:, :17, :17].sum((1, 2))
    if not np.array_equal(binary, collapsed):
        raise ValueError('Binary histograms must use occupied=1 and the same voxel mask')
    return validate_counts({
        'semantic_gt': semantic.sum(2)[:, :17],
        'semantic_pred': semantic.sum(1)[:, :17],
        'semantic_tp': np.diagonal(semantic, axis1=1, axis2=2)[:, :17],
        'binary_gt': binary[:, 1, :].sum(1),
        'binary_pred': binary[:, :, 1].sum(1),
        'binary_tp': binary[:, 1, 1],
    })


def confusion_from_labels(prediction, target, *, classes=18, ignore_label=255):
    pred, gt = np.asarray(prediction), np.asarray(target)
    if pred.shape != gt.shape or pred.dtype.kind not in 'iu' or gt.dtype.kind not in 'iu':
        raise ValueError('Prediction/target must be same-shape integer arrays')
    mask = gt != ignore_label
    pred, gt = pred[mask].astype(np.int64), gt[mask].astype(np.int64)
    if ((pred < 0) | (pred >= classes) | (gt < 0) | (gt >= classes)).any():
        raise ValueError('Invalid class labels')
    return np.bincount(gt * classes + pred, minlength=classes**2).reshape(classes, classes)


def score_counts(counts):
    counts = validate_counts(counts)
    rows = []
    for h, seconds in enumerate(HORIZONS):
        gt, pred, tp = (counts[f'semantic_{key}'][h] for key in ('gt', 'pred', 'tp'))
        present = gt > 0
        per_class = np.divide(tp, pred + gt - tp, out=np.zeros(17), where=(pred + gt - tp) > 0)
        union = counts['binary_gt'][h] + counts['binary_pred'][h] - counts['binary_tp'][h]
        rows.append({'seconds': seconds, 'miou': float(per_class[present].mean() * 100) if present.any() else None,
                     'iou': float(counts['binary_tp'][h] / union * 100) if union else None,
                     'present_classes': np.flatnonzero(present).tolist(),
                     'class_iou': [float(v * 100) if p else None for v, p in zip(per_class, present)]})

    def average(indices):
        return {key: float(np.mean([rows[i][key] for i in indices]))
                if all(rows[i][key] is not None for i in indices) else None
                for key in ('miou', 'iou')}

    return {'metric_policy': METRIC_POLICY, 'units': 'percent', 'horizons': rows,
            'paper_1_2_3s': average([1, 3, 5]), 'all_six': average(range(6))}
