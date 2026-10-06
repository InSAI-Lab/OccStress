"""Lightweight OccStress interfaces; no model framework is imported here."""
from pkgutil import extend_path

# SparseWorld's native datasets share this namespace in their isolated process.
__path__ = extend_path(__path__, __name__)
__version__ = '0.1.0'
