"""Cooperative pause/cancel control for participating analysis scripts."""
from __future__ import annotations
import json
import time
from pathlib import Path

class StopRequested(Exception): pass

class RunContext:
    def __init__(self, output):
        self.folder = Path(output)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.last_write = 0.0
        self.stage_seconds = {'compute':0.,'plot':0.}
        self.timing_stage = 'compute'
        self.timing_clock = time.monotonic()
        self.plot_start = 85.

    def checkpoint(self, progress: float, phase: str='', force=False):
        if (self.folder/'stop.request').exists():
            self.progress(progress, 'Stopping', force=True)
            raise StopRequested()
        paused = False
        while (self.folder/'pause.request').exists():
            if not paused:
                self.progress(progress, 'Paused', force=True)
                paused = True
            if (self.folder/'stop.request').exists(): raise StopRequested()
            time.sleep(0.2)
        self.progress(progress, phase, force=force or paused)

    def progress(self, progress: float, phase: str='', force=False):
        t = time.monotonic()
        if self.timing_stage in self.stage_seconds:
            self.stage_seconds[self.timing_stage] += t-self.timing_clock
        self.timing_clock = t
        if phase == 'Paused':self.timing_stage = 'paused'
        elif 'plotting' in phase.lower() or 'rendering' in phase.lower():
            if self.timing_stage != 'plot':self.plot_start = progress
            self.timing_stage = 'plot'
        elif progress < self.plot_start:self.timing_stage = 'compute'
        if not force and (t-self.last_write) < 0.35: return
        self.last_write = t
        target = self.folder/'progress.json'
        temp = self.folder/'progress.tmp'
        stage_progress = ((progress-self.plot_start)/(100-self.plot_start) if self.timing_stage=='plot'
                          else progress/self.plot_start)
        payload={'progress':max(0.0,min(100.0,float(progress))), 'phase':phase,
                 'stage':self.timing_stage,'stage_progress':max(0.,min(1.,stage_progress)),
                 'stage_seconds':self.stage_seconds,'updated':time.time()}
        temp.write_text(json.dumps(payload),encoding='utf-8')
        # Readers/OneDrive can briefly hold progress.json without delete sharing
        # on Windows. A transient progress-file lock must not abort the science.
        for attempt in range(5):
            try:
                temp.replace(target)
                break
            except PermissionError:
                if attempt == 4:return
                time.sleep(.02 * 2**attempt)


def load_args():
    import argparse
    parser = argparse.ArgumentParser(description='SIMP plugin analysis')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    a = parser.parse_args()
    data = json.loads(a.config.read_text(encoding='utf-8'))
    return data['parameters'], RunContext(a.output), data
