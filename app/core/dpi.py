"""Windows high-DPI setup for SIMP Control Center.

Call configure_process_dpi_awareness() BEFORE importing tkinter or creating a
window. Otherwise Windows can bitmap-stretch the whole interface, making text,
lines and icons blurry on displays using 125–250% scaling.

Windows API details:
https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-setprocessdpiawarenesscontext
"""
from __future__ import annotations

import os
import sys


def configure_process_dpi_awareness() -> str:
    """Enable sharp native DPI rendering, preferring Per-Monitor v2.

    The system-aware fallback is helpful for older Windows/Tk installations.
    This method changes only how the GUI is rendered, not how jobs are run.
    Set SIMP_DPI_MODE=system to skip per-monitor mode for troubleshooting.
    """
    if sys.platform != 'win32':
        return 'not-windows'

    import ctypes
    from ctypes import wintypes
    prefer_system = os.environ.get('SIMP_DPI_MODE', '').lower() == 'system'
    if not prefer_system:
        try:
            user32 = ctypes.WinDLL('user32', use_last_error=True)
            func = user32.SetProcessDpiAwarenessContext
            func.argtypes = [wintypes.HANDLE]
            func.restype = wintypes.BOOL
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = (HANDLE) -4.
            if func(ctypes.c_void_p(-4)):
                return 'per-monitor-v2'
        except (AttributeError, OSError):
            pass

    try:
        shcore = ctypes.WinDLL('shcore', use_last_error=True)
        func = shcore.SetProcessDpiAwareness
        func.argtypes = [ctypes.c_int]
        func.restype = ctypes.c_long
        # PROCESS_SYSTEM_DPI_AWARE = 1 (does not bitmap-stretch on primary screen).
        if func(1) == 0:
            return 'system-aware'
    except (AttributeError, OSError):
        pass

    try:
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        if user32.SetProcessDPIAware():
            return 'system-aware-legacy'
    except (AttributeError, OSError):
        pass
    # This may mean Python already configured awareness in its manifest.
    return 'already-configured-or-unavailable'


def get_dpi_for_window(window_id: int | None = None) -> int | None:
    """Return the physical Windows DPI for the Tk top-level, if available."""
    if sys.platform != 'win32':
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        if window_id is not None and hasattr(user32, 'GetDpiForWindow'):
            fn = user32.GetDpiForWindow
            fn.argtypes = [wintypes.HWND]
            fn.restype = wintypes.UINT
            n = int(fn(window_id))
            if 72 <= n <= 768:
                return n
        if hasattr(user32, 'GetDpiForSystem'):
            fn = user32.GetDpiForSystem
            fn.argtypes = []
            fn.restype = wintypes.UINT
            n = int(fn())
            if 72 <= n <= 768:
                return n
    except (AttributeError, OSError, ValueError, TypeError):
        pass
    return None
