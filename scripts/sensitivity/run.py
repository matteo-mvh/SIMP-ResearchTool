"""LHS sensitivity using only the shared equilibrium solver and Jacobian."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.sensitivity_runner import main
if __name__=='__main__':raise SystemExit(main())
