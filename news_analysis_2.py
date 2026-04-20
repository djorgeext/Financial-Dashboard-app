"""Backward-compatible module alias for backend.news_analysis_2."""
import sys

from backend import news_analysis_2 as _module

sys.modules[__name__] = _module
