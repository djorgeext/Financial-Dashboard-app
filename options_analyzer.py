"""Backward-compatible module alias for backend.options_analyzer."""
import sys

from backend import options_analyzer as _module

sys.modules[__name__] = _module
