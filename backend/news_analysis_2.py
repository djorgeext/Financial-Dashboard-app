"""Import-safe compatibility shim for the legacy notebook-derived module.

The original exploratory notebook script was moved to
`legacy/news_analysis_2_legacy_notebook.txt` because it contains Colab/Jupyter
magics and side effects that are not safe to execute during normal imports.
"""
from pathlib import Path
from typing import Final

LEGACY_NOTEBOOK_MODULE_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent
    / "legacy"
    / "news_analysis_2_legacy_notebook.txt"
)


def get_legacy_notebook_path() -> Path:
    """Return filesystem path to the legacy notebook-derived script."""
    return LEGACY_NOTEBOOK_MODULE_PATH


__all__ = ["LEGACY_NOTEBOOK_MODULE_PATH", "get_legacy_notebook_path"]
