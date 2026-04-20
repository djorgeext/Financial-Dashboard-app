"""Backward-compatible module alias for backend.news_service."""
import sys

from backend import news_service as _module

sys.modules[__name__] = _module
