"""Model-private adapters, separate from the shared ``occstress`` package."""
from pathlib import Path

# Keep the imported source files in place without shadowing the public SDK.
__path__.append(str(Path(__file__).resolve().parents[1] / 'occstress'))
