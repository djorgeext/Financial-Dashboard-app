"""Backward-compatible entrypoint and module alias for backend.validate."""
import sys

from backend import validate as _module


if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
