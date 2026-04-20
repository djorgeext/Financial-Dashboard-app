"""Backward-compatible module alias for backend.config."""
import sys

from backend import config as _module

sys.modules[__name__] = _module
