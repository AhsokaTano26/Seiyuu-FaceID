"""Streamlit launcher for the local seiyuu-faceid web interface."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seiyuu_faceid.web.pages import main, query_image_paths  # noqa: E402, F401, I001


if __name__ == "__main__":
    main()
