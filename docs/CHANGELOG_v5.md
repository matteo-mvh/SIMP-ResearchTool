# Version 5 — Windows high-resolution display update

- Declare Per-Monitor-V2 DPI awareness **before importing Tkinter** in `start_app.py`. Fall back to system-aware mode for older Tk/Windows configurations.
- Read native display DPI and scale the main window, navigation, job-status bar and styled table controls accordingly, while respecting available screen dimensions.
- Render the sidebar logo at the appropriate pixel dimensions from the original 256-pixel icon rather than magnifying a small thumbnail.
- Display plots using Pillow's Lanczos resampling instead of Tkinter's `PhotoImage.subsample`, which discarded pixels and made scientific labels/lines jagged.
- Raise default demo-plot export resolution from 130 to 200 DPI. This affects only demonstration plots, not scientific calculations or user-authored analyses.
- Preserve the v4 GUI layout, seed library, editable parameter forms, job queue and job records unchanged.

## Testing

11 backend unit tests passed. Each major GUI page opened under a Linux/Xvfb display. Native Windows DPI behavior cannot be verified in this environment; please test at your actual display scale.

## Upgrading

Shut down ongoing jobs. Install in a separate folder and copy your old `state/jobs.sqlite` and `results/runs/` directories (or point `SIMP_DATA_DIR` to the existing local data folder). Use `Start_SIMPCenter.bat` normally. If per-monitor awareness causes problems with an older Windows/Tk version, add `set SIMP_DPI_MODE=system` near the beginning of that launcher.
