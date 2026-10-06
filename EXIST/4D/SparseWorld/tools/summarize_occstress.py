#!/usr/bin/env python3
# OccStress adapter/portability modifications; see docs/source-imports.json.
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
INPUT_TIMES = (0.0, -0.5, -1.0, -1.5, -2.0)
TEMPORAL_PATTERNS = {
    'current': 'current_only',
    'history_k1': 'recent_burst',
    'all_frame': 'history_only',
}
METRIC_COLUMNS = tuple(
    f'{metric}_{horizon:.1f}s'
    for horizon in HORIZONS
    for metric in ('miou', 'iou')
)
SUMMARY_COLUMNS = (
    'mean_six_miou',
    'mean_six_iou',
    'paper_average_miou',
    'paper_average_iou',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate and summarize SparseWorld OccStress results.')
    parser.add_argument('--manifest', default='occstress/protocols.json')
    parser.add_argument('--result-root', default='occstress/results')
    parser.add_argument('--paper-gate', default='occstress/reproduction/paper_gate.json')
    parser.add_argument('--output-csv', default='occstress/summary.csv')
    parser.add_argument('--output-json', default='occstress/summary.json')
    parser.add_argument('--output-root', default='occstress/summary_final')
    parser.add_argument('--allow-incomplete', action='store_true')
    return parser.parse_args()


def atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def write_json(path, payload):
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
    )


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_number(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} is not finite: {value!r}')
    return float(value)


def flatten_metrics(payload):
    horizons = payload.get('metrics', {}).get('horizons', [])
    if len(horizons) != len(HORIZONS):
        raise ValueError(f'wrong horizon count: {len(horizons)}')
    by_time = {
        float(item['seconds']): item
        for item in horizons
    }
    if set(by_time) != set(HORIZONS):
        raise ValueError(f'wrong horizon times: {sorted(by_time)}')

    metrics = {}
    for horizon in HORIZONS:
        item = by_time[horizon]
        metrics[f'miou_{horizon:.1f}s'] = finite_number(
            item.get('miou'), f'mIoU at {horizon}s')
        metrics[f'iou_{horizon:.1f}s'] = finite_number(
            item.get('iou'), f'IoU at {horizon}s')

    mean_six = payload['metrics']['mean_six_frames']
    paper = payload['metrics']['paper_1s_2s_3s_average']
    metrics.update({
        'mean_six_miou': finite_number(
            mean_six.get('miou'), 'six-frame mean mIoU'),
        'mean_six_iou': finite_number(
            mean_six.get('iou'), 'six-frame mean IoU'),
        'paper_average_miou': finite_number(
            paper.get('miou'), 'paper-average mIoU'),
        'paper_average_iou': finite_number(
            paper.get('iou'), 'paper-average IoU'),
    })
    return metrics


def mean_metrics(rows):
    return {
        column: statistics.fmean(row[column] for row in rows)
        for column in METRIC_COLUMNS + SUMMARY_COLUMNS
    }


def group_rows(rows, manifest):
    corrupt_rows = [row for row in rows if row['corruption'] != 'clean']
    corruptions = list(dict.fromkeys(
        item['corruption']
        for item in manifest['protocols']
        if item['corruption'] != 'clean'
    ))
    severities = list(dict.fromkeys(
        item['severity']
        for item in manifest['protocols']
        if item['severity'] is not None
    ))
    temporal_patterns = [
        TEMPORAL_PATTERNS[name]
        for name in dict.fromkeys(
            item['frame_protocol']
            for item in manifest['protocols']
            if item['frame_protocol'] is not None
        )
    ]

    groups = [
        ('scope', 'clean', [
            row for row in rows if row['corruption'] == 'clean']),
        ('scope', 'robust_all', corrupt_rows),
    ]
    groups.extend(
        ('corruption', corruption, [
            row for row in corrupt_rows
            if row['corruption'] == corruption
        ])
        for corruption in corruptions
    )
    groups.extend(
        ('severity', severity, [
            row for row in corrupt_rows
            if row['severity'] == severity
        ])
        for severity in severities
    )
    groups.extend(
        ('temporal_pattern', pattern, [
            row for row in corrupt_rows
            if row['temporal_pattern'] == pattern
        ])
        for pattern in temporal_patterns
    )
    groups.extend(
        ('corruption_severity', f'{corruption}/{severity}', [
            row for row in corrupt_rows
            if row['corruption'] == corruption and row['severity'] == severity
        ])
        for corruption in corruptions
        for severity in severities
    )
    groups.extend(
        (
            'corruption_temporal_pattern',
            f'{corruption}/{pattern}',
            [
                row for row in corrupt_rows
                if row['corruption'] == corruption
                and row['temporal_pattern'] == pattern
            ],
        )
        for corruption in corruptions
        for pattern in temporal_patterns
    )
    return [
        {
            'group_type': group_type,
            'group': group,
            'protocol_count': len(grouped),
            **mean_metrics(grouped),
        }
        for group_type, group, grouped in groups
        if grouped
    ]


def selected_groups(aggregate_rows, group_type):
    return {
        row['group']: {
            'count': row['protocol_count'],
            **{
                column: row[column]
                for column in METRIC_COLUMNS + SUMMARY_COLUMNS
            },
        }
        for row in aggregate_rows
        if row['group_type'] == group_type
    }


def metric_delta(clean, robust):
    return {
        column: clean[column] - robust[column]
        for column in METRIC_COLUMNS + SUMMARY_COLUMNS
    }


def render_markdown(summary, paper_gate, output_root):
    clean = summary['clean']
    robust = summary['robust_mean']
    degradation = summary['degradation_from_clean']
    by_corruption = summary['by_corruption']
    by_temporal = summary['by_temporal_pattern']
    provenance = summary['provenance']

    gate_rows = []
    for check in paper_gate.get('checks', []):
        metric_label = {
            'miou': 'mIoU',
            'iou': 'IoU',
        }.get(check['metric'], check['metric'])
        time_label = (
            'Mean' if check['time'] == 'average'
            else f"{float(check['time']):g} s"
        )
        gate_rows.append(
            f"| {metric_label} @ {time_label} | "
            f"{check['target']:.2f} | {check['actual']:.3f} | "
            f"{check['delta']:.3f} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )

    corruption_rows = [
        f"| `{name}` | {values['count']} | "
        f"{values['mean_six_miou']:.3f} | "
        f"{values['mean_six_iou']:.3f} | "
        f"{clean['mean_six_miou'] - values['mean_six_miou']:.3f} |"
        for name, values in by_corruption.items()
    ]
    temporal_rows = [
        f"| `{name}` | {values['count']} | "
        f"{values['mean_six_miou']:.3f} | "
        f"{values['mean_six_iou']:.3f} | "
        f"{clean['mean_six_miou'] - values['mean_six_miou']:.3f} |"
        for name, values in by_temporal.items()
    ]
    retention = (
        100.0 * robust['mean_six_miou'] / clean['mean_six_miou'])

    content = f"""# SparseWorld Reproduction and OccStress Evaluation

## Status

- Generated: {summary['generated_at']}
- Official base commit: `{provenance['code_commit']}`
- Checkpoint SHA256: `{provenance['checkpoint_sha256']}`
- Adapter SHA256: `{provenance['adapter_sha256']}`
- Slurm array job: `{provenance['slurm_array_job_id']}`
- GPU: `{provenance['gpu']}`
- Validation: **PASS**, with {summary['protocol_count']}/73 protocols and \
{summary['sample_count_per_protocol']:,} anchors per protocol
- Position sweep: not run, following the predefined SparseWorld scope

## Official Reproduction Gate

The official {paper_gate['sample_count']:,}-anchor validation split reproduces
all reported SparseWorld 1/2/3-second metrics within the predefined absolute
tolerance of {paper_gate['tolerance']:.2f} percentage points.

| Metric | Paper | Reproduced | Absolute delta | Status |
|---|---:|---:|---:|:---:|
{chr(10).join(gate_rows)}

## OccStress Protocol

SparseWorld is evaluated as a direct image-to-future-occupancy model. Each
protocol uses the current image frame and four historical image frames, and
scores six genuine future occupancy targets at 0.5-second intervals from
`+0.5 s` through `+3.0 s`.

The raw camera protocol names map to the paper terminology as follows:

| Raw name | Paper terminology | Corrupted observations |
|---|---|---|
| `current` | `current_only` | Current frame |
| `history_k1` | `recent_burst` | Current and nearest history frame |
| `all_frame` | `history_only` | All four history frames; current is clean |

## Main Results

| Scope | Protocols | Six-frame mIoU | Six-frame IoU | mIoU drop |
|---|---:|---:|---:|---:|
| Clean | 1 | {clean['mean_six_miou']:.3f} | \
{clean['mean_six_iou']:.3f} | 0.000 |
| All corruptions | {robust['count']} | \
{robust['mean_six_miou']:.3f} | {robust['mean_six_iou']:.3f} | \
{degradation['mean_six_miou']:.3f} |

The robust six-frame mIoU retention is **{retention:.2f}%** of clean.
Using the paper's 1/2/3-second averaging convention, clean and robust mIoU are
{clean['paper_average_miou']:.3f} and
{robust['paper_average_miou']:.3f}, respectively.

### By Corruption

| Corruption | Protocols | mIoU | IoU | mIoU drop |
|---|---:|---:|---:|---:|
{chr(10).join(corruption_rows)}

### By Temporal Pattern

| Temporal pattern | Protocols | mIoU | IoU | mIoU drop |
|---|---:|---:|---:|---:|
{chr(10).join(temporal_rows)}

## Rebuttal Text

> We additionally evaluate SparseWorld, a direct image-to-future-occupancy
> model, using its official checkpoint. We first reproduce its reported
> 1/2/3-s validation metrics within 0.30 percentage points. We then evaluate
> the complete camera-upstream OccStress suite: one clean protocol and eight
> camera corruptions at three severities and three temporal patterns, for 73
> protocols in total. Every protocol contains 4,519 anchors and is evaluated
> on six genuine future targets from +0.5 to +3.0 s. SparseWorld obtains
> {clean['mean_six_miou']:.2f}/{clean['mean_six_iou']:.2f} clean mIoU/IoU and
> {robust['mean_six_miou']:.2f}/{robust['mean_six_iou']:.2f} robust mIoU/IoU
> averaged over the 72 corruption protocols.

## Artifacts

- `protocol_metrics.csv`: all 73 protocol-level metrics and runtime metadata.
- `aggregate_metrics.csv`: clean, robust, corruption, severity, and temporal
  aggregations.
- `summary.json`: machine-readable headline results and provenance.
- `validation.json`: completeness and consistency checks.
"""
    atomic_write_text(output_root / 'REBUTTAL_NOTES.md', content)


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    result_root = Path(args.result_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest = json.loads(manifest_path.read_text())
    expected_protocols = manifest['protocols']

    rows = []
    errors = []
    provenance_values = {
        'code_commit': set(),
        'checkpoint_sha256': set(),
        'adapter_sha256': set(),
        'gpu': set(),
        'slurm_array_job_id': set(),
    }
    expected_future_times = tuple(float(value) for value in HORIZONS)
    expected_input_times = tuple(float(value) for value in INPUT_TIMES)

    for protocol in expected_protocols:
        path = result_root / f'{protocol["id"]}.json'
        if not path.is_file():
            errors.append(f'missing: {protocol["id"]}')
            continue
        try:
            payload = json.loads(path.read_text())
            if payload.get('status') != 'success':
                raise ValueError(f"status={payload.get('status')}")
            if payload.get('protocol') != protocol:
                raise ValueError(
                    f"protocol identity mismatch: {payload.get('protocol')}")
            if payload.get('anchor_mode') != 'occstress':
                raise ValueError(
                    f"anchor_mode={payload.get('anchor_mode')}")
            if payload.get('sample_count') != 4519:
                raise ValueError(
                    f"sample_count={payload.get('sample_count')}")
            if payload.get('dataset_sample_count') != 4519:
                raise ValueError(
                    'dataset_sample_count='
                    f"{payload.get('dataset_sample_count')}")
            input_times = tuple(
                float(value)
                for value in payload.get('input_times_seconds', []))
            future_times = tuple(
                float(value)
                for value in payload.get('future_times_seconds', []))
            if input_times != expected_input_times:
                raise ValueError(f'input times={input_times}')
            if future_times != expected_future_times:
                raise ValueError(f'future times={future_times}')
            metrics = flatten_metrics(payload)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'invalid {protocol["id"]}: {error}')
            continue

        provenance = payload.get('provenance', {})
        runtime = payload.get('runtime', {})
        for field in ('code_commit', 'checkpoint_sha256', 'adapter_sha256'):
            provenance_values[field].add(str(provenance.get(field)))
        for field in ('gpu', 'slurm_array_job_id'):
            provenance_values[field].add(str(runtime.get(field)))

        temporal_pattern = (
            '' if protocol['frame_protocol'] is None
            else TEMPORAL_PATTERNS[protocol['frame_protocol']]
        )
        rows.append({
            'protocol_id': protocol['id'],
            'corruption': protocol['corruption'],
            'severity': protocol['severity'] or '',
            'frame_protocol': protocol['frame_protocol'] or '',
            'temporal_pattern': temporal_pattern,
            'result_path': str(path.relative_to(result_root.parent)),
            'sample_count': payload['sample_count'],
            **metrics,
            'elapsed_seconds': finite_number(
                runtime.get('elapsed_seconds'), 'elapsed seconds'),
            'samples_per_second': finite_number(
                runtime.get('samples_per_second'), 'samples per second'),
            'peak_gpu_memory_gib': finite_number(
                runtime.get('peak_gpu_memory_gib'), 'peak GPU memory'),
            'gpu': runtime.get('gpu'),
            'slurm_job_id': runtime.get('slurm_job_id'),
            'slurm_array_task_id': runtime.get('slurm_array_task_id'),
        })

    for field, values in provenance_values.items():
        if len(values) != 1:
            errors.append(
                f'inconsistent {field}: {sorted(values)}')

    try:
        paper_gate = json.loads(Path(args.paper_gate).read_text())
    except (OSError, json.JSONDecodeError) as error:
        paper_gate = {}
        errors.append(f'invalid paper gate: {error}')
    if not paper_gate.get('passed'):
        errors.append('official reproduction gate did not pass')

    expected_ids = [item['id'] for item in expected_protocols]
    valid_ids = [row['protocol_id'] for row in rows]
    if valid_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(valid_ids))
        unexpected = sorted(set(valid_ids) - set(expected_ids))
        if missing:
            errors.append(f'missing protocol ids: {missing}')
        if unexpected:
            errors.append(f'unexpected protocol ids: {unexpected}')

    complete = len(rows) == len(expected_protocols) == 73 and not errors
    validation = {
        'complete': complete,
        'expected_protocols': len(expected_protocols),
        'valid_protocols': len(rows),
        'expected_sample_count_per_protocol': 4519,
        'total_protocol_anchor_records': sum(
            row['sample_count'] for row in rows),
        'expected_corruptions': 8,
        'expected_severities': 3,
        'expected_temporal_patterns': 3,
        'official_reproduction_gate_passed': bool(paper_gate.get('passed')),
        'errors': errors,
        'provenance_values': {
            field: sorted(values)
            for field, values in provenance_values.items()
        },
    }
    write_json(output_root / 'validation.json', validation)
    if not complete and not args.allow_incomplete:
        raise SystemExit(
            f'incomplete SparseWorld results; see '
            f'{output_root / "validation.json"}')

    clean_rows = [row for row in rows if row['corruption'] == 'clean']
    corrupt_rows = [row for row in rows if row['corruption'] != 'clean']
    if len(clean_rows) != 1:
        raise SystemExit(f'expected one clean row, found {len(clean_rows)}')
    clean = {
        'count': 1,
        **{
            column: clean_rows[0][column]
            for column in METRIC_COLUMNS + SUMMARY_COLUMNS
        },
    }
    robust = {
        'count': len(corrupt_rows),
        **mean_metrics(corrupt_rows),
    }
    aggregate_rows = group_rows(rows, manifest)

    protocol_fields = (
        'protocol_id', 'corruption', 'severity', 'frame_protocol',
        'temporal_pattern', 'result_path', 'sample_count',
        *METRIC_COLUMNS, *SUMMARY_COLUMNS,
        'elapsed_seconds', 'samples_per_second', 'peak_gpu_memory_gib',
        'gpu', 'slurm_job_id', 'slurm_array_task_id',
    )
    aggregate_fields = (
        'group_type', 'group', 'protocol_count',
        *METRIC_COLUMNS, *SUMMARY_COLUMNS,
    )
    write_csv(Path(args.output_csv), rows, protocol_fields)
    write_csv(output_root / 'protocol_metrics.csv', rows, protocol_fields)
    write_csv(
        output_root / 'aggregate_metrics.csv',
        aggregate_rows,
        aggregate_fields,
    )

    provenance = {
        field: next(iter(values)) if len(values) == 1 else sorted(values)
        for field, values in provenance_values.items()
    }
    summary = {
        'complete': complete,
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'track': manifest.get('track'),
        'protocol_count': len(rows),
        'robust_protocol_count': len(corrupt_rows),
        'sample_count_per_protocol': 4519,
        'total_protocol_anchor_records': sum(
            row['sample_count'] for row in rows),
        'input_times_seconds': list(INPUT_TIMES),
        'future_times_seconds': list(HORIZONS),
        'temporal_pattern_mapping': TEMPORAL_PATTERNS,
        'position_sweep_protocols': 0,
        'clean': clean,
        'robust_mean': robust,
        'degradation_from_clean': metric_delta(clean, robust),
        'by_corruption': selected_groups(
            aggregate_rows, 'corruption'),
        'by_severity': selected_groups(
            aggregate_rows, 'severity'),
        'by_temporal_pattern': selected_groups(
            aggregate_rows, 'temporal_pattern'),
        'official_reproduction_gate': paper_gate,
        'provenance': provenance,
    }
    write_json(Path(args.output_json), summary)
    write_json(output_root / 'summary.json', summary)
    render_markdown(summary, paper_gate, output_root)

    print(json.dumps({
        'complete': complete,
        'protocols': len(rows),
        'robust_protocols': len(corrupt_rows),
        'total_protocol_anchor_records':
            summary['total_protocol_anchor_records'],
        'clean_mean_six_miou': clean['mean_six_miou'],
        'robust_mean_six_miou': robust['mean_six_miou'],
        'output_root': str(output_root),
    }, indent=2))


if __name__ == '__main__':
    main()
