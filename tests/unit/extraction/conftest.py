"""
tests/unit/extraction/conftest.py

Make the real PDF path available to extraction tests regardless of
the working directory pytest is invoked from.
"""
from pathlib import Path
import pytest

# The PDF sits two levels up from jp-fin-parser/
_REPO_ROOT = Path(__file__).parents[4]   # …/UR AI/
REAL_PDF_PATH = _REPO_ROOT / "balance_sheet_after_correction.pdf"
