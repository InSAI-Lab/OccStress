# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Compatibility entrypoint; implementation moved to scripts.nuscenes.generate_protocols."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nuscenes import generate_protocols as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
