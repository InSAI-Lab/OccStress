#!/usr/bin/env python3
"""Check local file links in OccStress documentation without network requests."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_release import private_release_path


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs
                          if key in {'href', 'src'} and value)


def local_links(text):
    # Fenced command examples are not rendered links.
    text = re.sub(r'^ *(`{3,}|~{3,})[^\n]*\n.*?^ *\1 *$', '', text,
                  flags=re.MULTILINE | re.DOTALL)
    parser = HTMLLinks()
    parser.feed(text)
    for match in re.finditer(r'!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\s*\)', text):
        parser.links.append(match.group(1).strip('<>'))
    for match in re.finditer(r'^ *\[[^\]\n]+\]:\s*(<[^>]+>|\S+)', text, re.MULTILINE):
        parser.links.append(match.group(1).strip('<>'))
    for link in parser.links:
        parsed = urlsplit(link)
        if not parsed.scheme and not parsed.netloc and parsed.path:
            yield unquote(parsed.path)


def documentation_files(root):
    files = set(root.glob('*.md')) | set((root / 'docs').rglob('*.md'))
    files.update((root / 'EXIST').rglob('README_OCCSTRESS.md'))
    for directory in ('data', 'configs', 'results', 'tools/adapters'):
        files.update((root / directory).rglob('*.md'))
    return sorted(path for path in files
                  if not private_release_path(path.relative_to(root)))


def issues(root):
    root = Path(root).resolve()
    errors = []
    for path in documentation_files(root):
        for link in local_links(path.read_text()):
            target = (path.parent / link).resolve()
            if not target.is_relative_to(root) or not target.exists():
                errors.append(f'{path.relative_to(root)}: missing or external local link: {link}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=ROOT)
    args = parser.parse_args()
    errors = issues(args.root)
    if errors:
        print('\n'.join(errors))
    else:
        print('OccStress documentation file links passed (external URLs and heading anchors not checked).')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
