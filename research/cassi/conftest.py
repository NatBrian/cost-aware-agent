"""Make `import cassi.*` work when running pytest from research/cassi/."""

import sys
from pathlib import Path

_RESEARCH_DIR = str(Path(__file__).resolve().parent.parent)
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)
