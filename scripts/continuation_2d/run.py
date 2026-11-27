
"""SIMP 2D continuation with automatic bifurcation classification."""

from pathlib import Path
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)

from scripts.bifurcation_classification import classified_main


if __name__ == '__main__':
    raise SystemExit(classified_main(2))
