"""Launch the SIMP desktop application with an independent Windows taskbar icon."""
from __future__ import annotations
import sys

# Must run before importing tkinter (including via app.ui).
from app.core.dpi import configure_process_dpi_awareness
DPI_MODE = configure_process_dpi_awareness()

# On Windows, a stable AppUserModelID helps separate the SIMP window from
# generic Python consoles in taskbar grouping. Tk will also set PNG/ICO icons.
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'SIMP.ResearchControlCenter.Desktop.v2')
    except (AttributeError, OSError):
        pass

from app.ui import Application

if __name__ == '__main__':
    Application().mainloop()
