"""Explicit source availability, independent of historical evaluation coverage."""
import json
from pathlib import Path, PurePosixPath


def load_distribution(root):
    path = Path(root) / 'configs/distribution.json'
    value = json.loads(path.read_text())
    if value.get('schema_version') != 1 or value.get('profile') not in {
            'maintainer', 'public_candidate', 'full_source_candidate'}:
        raise ValueError('Invalid distribution manifest')
    seen = set()
    for component in value['pending_components']:
        if component['id'] in seen or component['status'] != 'pending_upstream_permission':
            raise ValueError('Invalid pending component: ' + component['id'])
        seen.add(component['id'])
        for relative in [component['directory'], *component['excluded_paths']]:
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts or not path.parts or str(path) != relative:
                raise ValueError('Unsafe distribution path: ' + relative)
        if component['directory'] not in component['excluded_paths']:
            raise ValueError('Pending model directory must be excluded')
    return value


def component_for_path(relative, manifest):
    path = PurePosixPath(relative)
    for component in manifest['pending_components']:
        for prefix in component['excluded_paths']:
            if path == PurePosixPath(prefix) or PurePosixPath(prefix) in path.parents:
                return component
    return None


def source_status(root, relative):
    manifest = load_distribution(root)
    if component_for_path(relative, manifest):
        return ('pending_upstream_permission' if manifest['profile'] == 'public_candidate'
                else 'included_permission_unconfirmed')
    return 'included_subject_to_component_terms'


def withheld(root, relative):
    manifest = load_distribution(root)
    return (manifest['profile'] == 'public_candidate'
            and component_for_path(relative, manifest) is not None)


def require_source(root, relative):
    manifest = load_distribution(root)
    component = component_for_path(relative, manifest)
    if manifest['profile'] == 'public_candidate' and component:
        raise ValueError(
            f"{component['name']}: Pending upstream permission. Adapted source is "
            "not included in this distribution; see docs/CODE_LAYOUT.md#source-packaging. "
            "Cloning the unmodified upstream repository does not restore the "
            "validated OccStress adapter.")


def placeholder_text(component):
    return (
        f"# {component['name']}\n\n"
        "**Source not included in this package profile.**\n\n"
        "This is an OccStress-authored availability notice, not model source.\n"
        "The adapted source and associated model-specific helpers are omitted\n"
        "by the selected source profile.\n"
        "Internal experiments are retained; this placeholder is not runnable.\n\n"
        f"- Official source: {component['upstream_url']}\n"
        f"- Audited upstream revision: {component['audited_revision']}\n"
        "- The audited revision is a license-audit reference, not a claim that\n"
        "  an unmodified upstream checkout reproduces the adapted experiments.\n"
        "\nSee docs/CODE_LAYOUT.md#source-packaging at the repository root\n"
        "for source profiles and packaging instructions.\n"
    )


def public_source_issues(root, manifest=None):
    root = Path(root)
    manifest = manifest or load_distribution(root)
    if manifest['profile'] == 'full_source_candidate':
        errors = []
        if manifest.get('public_redistribution_ready') is not False:
            errors.append('Full source inclusion does not establish redistribution permission.')
        for component in manifest['pending_components']:
            for relative in component['excluded_paths']:
                path = root / relative
                if not path.exists() or path.is_symlink():
                    errors.append('Missing full-source integration: ' + relative)
            directory = root / component['directory']
            if not any(directory.rglob('*.py')):
                errors.append('Missing model implementation: ' + component['directory'])
            for name in ('README.md', 'README_OCCSTRESS.md'):
                path = directory / name
                if path.is_file() and path.read_text() == placeholder_text(component):
                    errors.append('Permission placeholder remains in full source: ' + str(path.relative_to(root)))
        return errors
    if manifest['profile'] != 'public_candidate':
        return ['Maintainer tree: build a separate public candidate; do not publish this checkout or its Git history.']
    errors = []
    for component in manifest['pending_components']:
        allowed = {component['directory'] + '/' + name
                   for name in ('README.md', 'README_OCCSTRESS.md')}
        for prefix in component['excluded_paths']:
            path = root / prefix
            entries = [path, *path.rglob('*')] if path.is_dir() and not path.is_symlink() else [path]
            for entry in entries:
                relative = entry.relative_to(root).as_posix()
                if entry.is_symlink():
                    errors.append('Withheld source symlink: ' + relative)
                elif entry.is_file():
                    if relative not in allowed:
                        errors.append('Withheld source included: ' + relative)
                    elif entry.read_text() != placeholder_text(component):
                        errors.append('Changed permission placeholder: ' + relative)
        for relative in allowed:
            if not (root / relative).is_file():
                errors.append('Missing permission placeholder: ' + relative)
    return errors
