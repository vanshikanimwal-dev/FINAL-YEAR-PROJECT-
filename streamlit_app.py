"""Entry point for Streamlit Community Cloud.

The cloud runner starts in the repository root. The study package lives in
``cicd_bottleneck`` and resolves its data files from that folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "cicd_bottleneck"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dashboard import main

main()
