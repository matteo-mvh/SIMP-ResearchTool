"""SIMP 1D continuation.

Metadata and model inputs are supplied by analysis.toml's continuation form.
The worker provides --config and --output. Equilibrium discovery and the full
Jacobian use model.shared_eqn; pyoomph traces branches and corrects events.
Output: CSV branches/events, JSON diagnostics, app PNGs and thesis PDFs.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.continuation_runner import main

if __name__ == '__main__':
    raise SystemExit(main(1))
