#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


FORBIDDEN_DIR_NAMES = {
    ".codex",
    "__pycache__",
    "ckpt",
    "ckpts",
    "checkpoint",
    "checkpoints",
    "logs",
    "outputs",
    "tmp_results",
    "wandb",
    "work_dir",
    "work_dirs",
}

FORBIDDEN_FILE_SUFFIXES = {
    ".ckpt",
    ".egg",
    ".log",
    ".npy",
    ".npz",
    ".out",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".so",
    ".tar",
    ".whl",
    ".zip",
}

def _join(*parts: str) -> str:
    return "".join(parts)


ANONYMOUS_NAME_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        _join("hore", "ka"),
        _join("s", "lurm"),
        _join("nuscenes", "-sc"),
        _join("robust", "occ"),
    ]
]

ANONYMOUS_TEXT_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\btp[0-9]{4}\b",
        r"/" + "hkfs",
        r"/" + "data1",
        _join("hk", "-project"),
        _join("hkn", r"[0-9]+"),
        _join("Hore", "Ka"),
        _join("hore", "ka"),
        _join("scc", r"\.kit"),
        _join("kit", "-hpc"),
        _join("s", r"sh\s+-"),
        _join("r", "sync"),
        _join("S", "BATCH"),
        _join("S", "LURM"),
        _join("Robust", "Occ"),
        _join("nuScenes", "-SC"),
        _join("ROBUST", "OCC"),
    ]
]

FORBIDDEN_TEXT_PATTERNS = [
    re.compile(pattern) for pattern in [
        r'/data[01]/(?:userdata|public)/',
        r'/hkfs/', r'/home/hk-project-',
        r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
    ]
]

# Deployment inventories and research notes are not user-facing release docs.
PRIVATE_RELEASE_PATHS = {
    'docs/md', 'docs/archive', 'docs/code-layout-migration.json',
    'ANONYMIZATION_NOTES.md',
}
DEPLOYMENT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in (
        _join('hore', 'ka'), _join('hk', r'-login\d*'),
        _join('hk', '-project'), _join('scc', r'\.kit'),
        _join('vpn', 'kit'), r'\b(?:tp|dk)\d{4}\b',
        _join(r'\bhk', r'n\d+\b'),
        _join('10.157.', '195.126'), _join('129.146.', '137.23'),
        _join('47.119.', '176.238'),
    )
]

DATASET_NAME_PATTERNS = [
    re.compile(r'\b(?:nuscenes|waymo|carla)[-\u2010\u2011\u2013\u2014]sc\b', re.IGNORECASE),
]

BENCHMARK_IDENTIFIER_PATTERNS = [
    re.compile(r'(?<=_)sc(?=_|\b)|\bsc(?=_)'),
    re.compile(r'\b(?:IIWORLD_WAYMO|IIWORLD|CARLA|WAYMO|COME|OCCWORLD)_SC_'),
    re.compile(r'(?:NuScenes|Waymo)SC(?:World|FutureTokenizer|Tokenizer)Dataset'),
    re.compile(r'(?:SceneDataset|SceneDatasetLidarTraverse|IISceneTokenizer|LoadStreamOcc3D|Latent(?:History|Vote)?Token)SC'),
    re.compile(r'--(?:output-)?sc-root\b|\bSC\b|scaware'),
]


def benchmark_name_issues(relative, text):
    # In native 3D estimators, SC is the standard scene-completion metric.
    if relative.startswith('EXIST/3D/'):
        patterns = BENCHMARK_IDENTIFIER_PATTERNS[:-1]
    else:
        patterns = BENCHMARK_IDENTIFIER_PATTERNS
    return any(pattern.search(text) for pattern in patterns)


def private_release_path(relative):
    path = Path(relative)
    return (any(path == Path(prefix) or Path(prefix) in path.parents
                for prefix in PRIVATE_RELEASE_PATHS)
            or (path.parent == Path('environments')
                and path.name.startswith('observed-') and path.suffix == '.json'))


TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".cff", ".bib", ".ipynb", ".patch", ".diff",
    ".c", ".cpp", ".cu", ".h", ".hpp",
}


def is_text_candidate(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.name in {"README", "LICENSE"}


def scan(root: Path, *, anonymous=False, syntax=False, public=False) -> list[str]:
    issues: list[str] = []
    root = root.resolve()
    public_patterns = DEPLOYMENT_PATTERNS + DATASET_NAME_PATTERNS if public else []
    name_patterns = (ANONYMOUS_NAME_PATTERNS if anonymous else []) + public_patterns
    text_patterns = FORBIDDEN_TEXT_PATTERNS + (ANONYMOUS_TEXT_PATTERNS if anonymous else []) + public_patterns

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if ".git" in path.parts:
            continue
        mounts = {'OccStress', 'OccStress-nuScenes', 'OccStress-Waymo', 'OccStress-CARLA', 'nuscenes'}
        relative_parts = path.relative_to(root).parts
        if public and private_release_path(rel):
            issues.append(f'private release artifact: {rel}')
        if public and len(relative_parts) >= 2 and relative_parts[0] == 'data' and rel != 'data/README.md':
            issues.append(f'data asset or mount in source release: {rel}')
        if len(relative_parts) >= 2 and relative_parts[0] == 'data' and relative_parts[1] in mounts:
            continue
        if rel == "tools/check_release.py":
            continue
        if public and benchmark_name_issues(rel, rel):
            issues.append(f'forbidden name: {rel} (obsolete benchmark identifier)')

        if path.is_symlink():
            target = path.readlink()
            if target.is_absolute():
                issues.append(f"absolute symlink: {rel} -> {target}")
            else:
                resolved = (path.parent / target).resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    issues.append(f"symlink escapes repo: {rel} -> {target}")
                if not path.exists():
                    issues.append(f"broken symlink: {rel} -> {target}")
            continue

        if path.is_dir():
            if any(pattern.search(rel) for pattern in name_patterns):
                issues.append(f"forbidden name: {rel}")
            if path.name in FORBIDDEN_DIR_NAMES:
                issues.append(f"forbidden directory: {rel}")
            continue

        if any(pattern.search(rel) for pattern in name_patterns):
            issues.append(f"forbidden name: {rel}")

        suffixes = path.suffixes
        if path.suffix in FORBIDDEN_FILE_SUFFIXES or any("".join(suffixes[-n:]) in FORBIDDEN_FILE_SUFFIXES for n in (2, 3)):
            issues.append(f"forbidden generated/binary file: {rel}")

        if not is_text_candidate(path):
            continue
        try:
            text = path.read_text(errors="strict")
        except UnicodeDecodeError:
            continue
        if public and benchmark_name_issues(rel, text):
            issues.append(f'forbidden text: {rel} (obsolete benchmark identifier)')
        if syntax and path.suffix == '.py':
            try:
                ast.parse(text, filename=rel)
            except SyntaxError as exc:
                issues.append(f'Python syntax: {rel}:{exc.lineno}: {exc.msg}')
        for pattern in text_patterns:
            if pattern.search(text):
                issues.append(f"forbidden text {pattern.pattern!r}: {rel}")
                break

    if (root / 'configs/resources.json').is_file():
        if __package__:
            from .release_status import inspect
        else:
            from release_status import inspect
        try:
            issues.extend(inspect(root)['issues'])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            issues.append(f'invalid release registry: {exc}')
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check code-release hygiene and declared entrypoints, not GPU reproducibility.")
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan.")
    parser.add_argument('--anonymous', action='store_true', help='Also apply legacy anonymous-submission naming rules.')
    parser.add_argument('--syntax', action='store_true', help='Parse all Python source without importing model dependencies.')
    parser.add_argument('--public', action='store_true', help='Check source-only candidate completeness and distribution profile, not legal clearance.')
    args = parser.parse_args()

    issues = scan(Path(args.root), anonymous=args.anonymous, syntax=args.syntax, public=args.public)
    if args.public:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from occstress.distribution import public_source_issues
        issues.extend(public_source_issues(Path(args.root)))
    if issues:
        print("Release check failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Code release check passed (download availability and GPU reproduction not tested).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
