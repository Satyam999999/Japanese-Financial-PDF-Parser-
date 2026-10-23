"""
jp-fin-parser root conftest.py

Adds src/ to sys.path so that pytest and tools run without requiring PYTHONPATH=src.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
