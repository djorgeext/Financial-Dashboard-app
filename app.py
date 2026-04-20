"""Backward-compatible entrypoint and module alias for backend.app."""
import sys

from backend import app as _module


if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
