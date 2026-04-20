"""Backward-compatible module alias for backend.inference_service."""
import sys

from backend import inference_service as _module

sys.modules[__name__] = _module
