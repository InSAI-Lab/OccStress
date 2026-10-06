#!/usr/bin/env python3
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics


HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
METRIC_COLUMNS = tuple(
    f'{metric}_{horizon}'
    for horizon in HORIZONS
    for metric in ('miou', 'iou'))
POSITION_OFFSETS = {
    'tminus4': -2.0,
    'tminus3': -1.5,
    'tminus2': -1.0,
    'tminus1': -0.5,
    't': 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize completed GenieDrive OccStress evaluations.')
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    return parser.parse_args()


def flatten_metrics(payload):
    row = {}
    for horizon in HORIZONS:
        metrics = payload['horizon_metrics'][str(horizon)]
        row[f'miou_{horizon}'] = float(metrics['miou'])
        row[f'iou_{horizon}'] = float(metrics['iou'])
    row['average_miou'] = float(payload['average_miou'])
    row['average_iou'] = float(payload['average_iou'])
    return row


def standard_identity(relative):
    parts = relative.parts
    if parts[0] == 'manual':
        track = 'manual'
        detail = parts[1:]
    elif parts[:3] == ('upstream', 'camera_only', 'stcocc'):
        track = 'stcocc'
        detail = parts[3:]
    elif parts[:3] == ('upstream', 'pointcloud_fusion', 'sdgocc'):
        track = 'sdgocc'
        detail = parts[3:]
    else:
        raise ValueError(f'unrecognized standard result path: {relative}')

    corruption = detail[0]
    severity = ''
    if len(detail) >= 3 and detail[1] in {
            'easy', 'mid', 'hard', 'light', 'moderate', 'heavy'}:
        severity = detail[1]
    pattern = detail[-1].split('_H4_', 1)[0]
    return track, corruption, severity, pattern


def position_name(filename):
    stem = filename.removesuffix('.json')
    for name in ('tminus4', 'tminus3', 'tminus2', 'tminus1'):
        if f'_{name}_H4_' in stem:
            return name
    if '_clean_H4_' in stem:
        return 'clean'
    if '_t_H4_' in stem:
        return 't'
    raise ValueError(f'unrecognized position result: {filename}')


def mean_metrics(rows):
    return {
        column: statistics.fmean(row[column] for row in rows)
        for column in METRIC_COLUMNS + ('average_miou', 'average_iou')
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_heatmaps(position_rows, output_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    for setting, rows in sorted(position_rows.items()):
        by_position = {row['position']: row for row in rows}
        clean = by_position['clean']
        positions = ('tminus4', 'tminus3', 'tminus2', 'tminus1', 't')
        matrix = np.array([
            [
                clean[f'miou_{horizon}'] -
                by_position[position][f'miou_{horizon}']
                for position in positions
            ]
            for horizon in HORIZONS
        ])
        figure, axis = plt.subplots(figsize=(7.4, 4.5))
        image = axis.imshow(matrix, cmap='RdYlBu_r', aspect='auto')
        axis.set_xticks(
            range(len(positions)),
            [str(POSITION_OFFSETS[position]) for position in positions])
        axis.set_yticks(
            range(len(HORIZONS)),
            [str(horizon) for horizon in HORIZONS])
        axis.set_xlabel('Corrupted input offset (s)')
        axis.set_ylabel('Forecast horizon (s)')
        axis.set_title(f'{setting}: mIoU drop from clean')
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f'{matrix[row_index, column_index]:.2f}',
                    ha='center',
                    va='center',
                    fontsize=8)
        figure.colorbar(image, ax=axis, label='mIoU drop')
        figure.tight_layout()
        figure.savefig(
            output_root / f'position_heatmap_{setting}.png', dpi=180)
        plt.close(figure)


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    standard_rows = []
    position_rows = defaultdict(list)
    invalid = []
    for result_path in sorted(input_root.rglob('*.json')):
        if result_path.name == 'dispatch_manifest.json':
            continue
        try:
            with result_path.open() as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as error:
            invalid.append({'path': str(result_path), 'error': str(error)})
            continue
        if payload.get('status') != 'success':
            continue
        relative = result_path.relative_to(input_root)
        if int(payload.get('evaluated_records', -1)) != 4519:
            invalid.append({
                'path': str(relative),
                'error': (
                    f"evaluated_records={payload.get('evaluated_records')}"),
            })
            continue
        metrics = flatten_metrics(payload)
        if relative.parts[0] == 'position_sweep':
            setting = relative.parts[1]
            position = position_name(relative.name)
            position_rows[setting].append({
                'setting': setting,
                'position': position,
                'offset_seconds': (
                    '' if position == 'clean'
                    else POSITION_OFFSETS[position]),
                'result_path': str(relative),
                **metrics,
            })
        else:
            track, corruption, severity, pattern = standard_identity(relative)
            standard_rows.append({
                'track': track,
                'corruption': corruption,
                'severity': severity,
                'frame_protocol': pattern,
                'result_path': str(relative),
                **metrics,
            })

    standard_count = len(standard_rows)
    position_count = sum(len(rows) for rows in position_rows.values())
    complete = standard_count == 184 and position_count == 18 and not invalid
    if not args.allow_incomplete and not complete:
        raise SystemExit(
            f'incomplete results: standard={standard_count}/184, '
            f'position={position_count}/18, invalid={len(invalid)}')

    protocol_fields = (
        'track', 'corruption', 'severity', 'frame_protocol', 'result_path',
        *METRIC_COLUMNS, 'average_miou', 'average_iou')
    write_csv(
        output_root / 'protocol_metrics.csv',
        standard_rows,
        protocol_fields)

    aggregate_rows = []
    groups = defaultdict(list)
    for row in standard_rows:
        groups[('track', row['track'])].append(row)
        groups[(
            'corruption',
            row['track'],
            row['corruption'])].append(row)
        groups[(
            'severity',
            row['track'],
            row['corruption'],
            row['severity'])].append(row)
        groups[(
            'frame_protocol',
            row['track'],
            row['corruption'],
            row['severity'],
            row['frame_protocol'])].append(row)
    for key, rows in sorted(groups.items()):
        aggregate_rows.append({
            'group_type': key[0],
            'group': '/'.join(key[1:]),
            'protocol_count': len(rows),
            **mean_metrics(rows),
        })
    write_csv(
        output_root / 'aggregate_metrics.csv',
        aggregate_rows,
        (
            'group_type', 'group', 'protocol_count',
            *METRIC_COLUMNS, 'average_miou', 'average_iou'))

    flat_position_rows = [
        row
        for setting in sorted(position_rows)
        for row in sorted(
            position_rows[setting],
            key=lambda item: (
                item['position'] != 'clean',
                item['offset_seconds'] if item['position'] != 'clean' else -3))
    ]
    write_csv(
        output_root / 'position_sweep.csv',
        flat_position_rows,
        (
            'setting', 'position', 'offset_seconds', 'result_path',
            *METRIC_COLUMNS, 'average_miou', 'average_iou'))

    visible_rows = []
    ignored_max_difference = 0.0
    for setting, rows in sorted(position_rows.items()):
        by_position = {row['position']: row for row in rows}
        if set(by_position) != {'clean', *POSITION_OFFSETS}:
            if not args.allow_incomplete:
                raise SystemExit(
                    f'{setting} position set is incomplete: {sorted(by_position)}')
            continue
        clean = by_position['clean']
        for position, offset in POSITION_OFFSETS.items():
            drops = [
                clean[f'miou_{horizon}'] -
                by_position[position][f'miou_{horizon}']
                for horizon in HORIZONS
            ]
            visible_rows.append({
                'setting': setting,
                'position': position,
                'offset_seconds': offset,
                'visible_to_geniedrive': offset >= -1.5,
                'mean_miou_drop': statistics.fmean(drops),
                'max_abs_miou_difference': max(abs(value) for value in drops),
            })
        ignored_row = next(
            row for row in visible_rows
            if row['setting'] == setting and row['position'] == 'tminus4')
        ignored_max_difference = max(
            ignored_max_difference,
            ignored_row['max_abs_miou_difference'])

    write_csv(
        output_root / 'position_common_visible.csv',
        visible_rows,
        (
            'setting', 'position', 'offset_seconds',
            'visible_to_geniedrive', 'mean_miou_drop',
            'max_abs_miou_difference'))

    if position_count == 18:
        render_heatmaps(position_rows, output_root)
        if ignored_max_difference > 0.01:
            raise SystemExit(
                't=-2.0 differs from clean by more than 0.01 mIoU: '
                f'{ignored_max_difference}')

    clean_rows = [
        row for row in standard_rows if row['corruption'] == 'clean'
    ]
    robust_rows = [
        row for row in standard_rows if row['corruption'] != 'clean'
    ]
    summary = {
        'complete': complete,
        'standard_protocols': standard_count,
        'position_sweep_protocols': position_count,
        'invalid_results': invalid,
        'clean': {
            row['track']: mean_metrics([row]) for row in clean_rows
        },
        'robust_mean': (
            mean_metrics(robust_rows) if robust_rows else None),
        'robust_mean_by_track': {
            track: mean_metrics([
                row for row in robust_rows if row['track'] == track
            ])
            for track in sorted({row['track'] for row in robust_rows})
        },
        'ignored_tminus4_max_abs_miou_difference':
            ignored_max_difference,
        'common_visible_mean_miou_drop': (
            statistics.fmean(
                row['mean_miou_drop']
                for row in visible_rows
                if row['visible_to_geniedrive'])
            if visible_rows else None),
    }
    temporary = output_root / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    temporary.replace(output_root / 'summary.json')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
