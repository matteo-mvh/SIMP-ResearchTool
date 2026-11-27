"""SIMP 2D bifurcation continuation.

Metadata and model inputs are supplied by analysis.toml's continuation form.
The worker provides --config and --output. Scalar equilibria seed native
pyoomph fold/Hopf/invasion tracking. Region checks use the shared Jacobian.
Output: CSV curves/events, JSON regions/diagnostics, app PNGs and thesis PDFs.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.continuation_runner import main

if __name__ == '__main__':
    raise SystemExit(main(2))
