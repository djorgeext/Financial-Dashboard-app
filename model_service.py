"""Backward-compatible module alias for backend.model_service."""
import sys

from backend import model_service as _module

sys.modules[__name__] = _module
