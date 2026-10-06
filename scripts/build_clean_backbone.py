# OccStress adapter/portability modifications; see docs/source-imports.json.
"""Compatibility entrypoint; implementation moved to scripts.nuscenes.build_clean_backbone."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nuscenes import build_clean_backbone as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
