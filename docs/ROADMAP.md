# Roadmap to the SIMP thesis research environment

1. **Scientific foundation:** Transfer the existing, tested SIMP equations into `model/`, with parameter defaults (15°C), unit tests, and a single parameter schema.
2. **Equilibrium solver:** Implement the eight presence–absence classes using the five reduced 1D residuals in the equilibrium-solver appendix. Validate root residuals in the original equations, physical bounds, and the full Jacobian.
3. **Sensitivity:** Replace `sensitivity_demo/run.py` with Latin hypercube sampling plus systematic equilibrium searches, convergence checks, and Seed-ID/parameter export. Integrate uncertainty estimates.
4. **Continuation:** Replace the placeholder 1D/2D analysis scripts with adapters to the real PyOOMPH scripts. Ensure each has a standalone CLI entry point.
5. **Long-running safety:** Add checkpointing/resume for individual sensitivity samples and continuation steps. Enable safe termination of solver subprocess trees and log restart policies.
6. **Scientific results:** Add standardized result schema, branch diagrams, parameter heatmaps, job comparisons and CSV/Parquet output where justified.
7. **Deployment:** Pin computational environment dependencies; add a `conda` environment file if PyOOMPH requires special installation, and package Windows GUI when model backend is stable.

**Avoid coupling scientific work to the GUI.** Scripts should remain directly callable and independently testable. Track Git commit and dependency versions for every run when the real model is integrated.
