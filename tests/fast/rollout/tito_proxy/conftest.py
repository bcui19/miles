"""Make the shared stub modules (stub_stack, _stub_heavy_deps) importable when
pytest collects this directory as a package (miles' default --pyargs mode does
not put package test dirs on sys.path)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
