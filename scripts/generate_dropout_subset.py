# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Compatibility entrypoint; implementation moved to occstress.corruptions.dropout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occstress.corruptions import dropout as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
