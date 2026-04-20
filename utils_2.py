"""Backward-compatible module alias for backend.utils_2."""
import sys

from backend import utils_2 as _module

sys.modules[__name__] = _module
