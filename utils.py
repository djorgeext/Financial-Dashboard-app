"""Backward-compatible module alias for backend.utils."""
import sys

from backend import utils as _module

sys.modules[__name__] = _module
