"""Project paths. Set SIMP_DATA_DIR to keep data outside the source tree."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get('SIMP_DATA_DIR', ROOT)).resolve()
STATE = DATA_ROOT / 'state'
RESULTS = DATA_ROOT / 'results'
RUNS = RESULTS / 'runs'
BATCHES = RESULTS / 'batches'
SCRIPTS = Path(os.environ.get('SIMP_SCRIPTS_DIR', ROOT / 'scripts')).resolve()
ANALYSES = SCRIPTS  # Backwards-compatible internal name.
ASSETS = ROOT / 'app' / 'assets'

def ensure_dirs():
    for p in (STATE, RESULTS, RUNS, BATCHES):
        p.mkdir(parents=True, exist_ok=True)
