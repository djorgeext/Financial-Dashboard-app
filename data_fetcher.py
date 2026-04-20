"""Backward-compatible module alias for backend.data_fetcher."""
import sys

from backend import data_fetcher as _module

sys.modules[__name__] = _module
