"""Independent sequential scheduler. Designed to survive GUI shutdown."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .paths import ROOT, STATE
from .storage import Repository, dump_json, now
from .discovery import discover
from .debug_export import export_debug, UPDATE_INTERVAL


def alive(pid):
    if pid <= 0: return False
    if sys.platform == 'win32':
        # os.kill(pid, 0) calls TerminateProcess on Windows;
        # never use it as a probe.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied is not proof of death.
            return ctypes.get_last_error() == 5

        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)

    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def worker_python():
    """Console Python provides real log streams,
    while CREATE_NO_WINDOW hides it."""
    executable = Path(sys.executable)
    if sys.platform == 'win32' and executable.name.lower() == 'pythonw.exe':
        console = executable.with_name('python.exe')
        if console.is_file():
            return str(console)

    return str(executable)


def heartbeat():
    try:
        dump_json(STATE / 'worker_heartbeat.json', {'pid': os.getpid(), 'updated': time.time(),},)
    except OSError:
        # A transient file-sharing lock must not orphan
        # a running computation.
        pass


def acquire_lock(path: Path):
    for _ in range(2):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,)
            with os.fdopen(fd, 'w') as fp:
                fp.write(str(os.getpid()))

            return True

        except FileExistsError:
            try:
                oldpid = int(path.read_text().strip())
                if oldpid != os.getpid() and alive(oldpid):
                    return False

                path.unlink(missing_ok=True)

            except (ValueError, OSError):
                return False

    return False


def request(ident, kind):
    job = Repository().get_job(ident)
    if not job or job['status'] not in ('running', 'paused', 'queued', 'draft'):
        return False

    folder = Path(job['output_dir'])
    if kind == 'pause' and job['status'] in ('running', 'paused'):
        (folder / 'pause.request').write_text('', encoding='utf-8')

    elif kind == 'resume':
        (folder / 'pause.request').unlink(missing_ok=True)

    elif kind == 'stop':
        if job['status'] in ('draft', 'queued'):
            Repository().update_job(
                ident,
                status='stopped',
                ended_at=now(),
                phase='Stopped before execution',
            )
        else:
            (folder / 'stop.request').write_text('', encoding='utf-8')

    elif kind == 'force':
        (folder / 'force_stop.request').write_text('', encoding='utf-8')

    else:
        return False

    return True


def force_terminate(proc):
    if proc.poll() is not None:
        return

    try:
        if sys.platform == 'win32':
            subprocess.run(
                ['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)

        try:
            proc.wait(timeout=3)

        except subprocess.TimeoutExpired:
            if sys.platform == 'win32':
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)

    except (ProcessLookupError, OSError):
        pass


def _run_job(repo, job):
    """Execute one queued analysis and update its runtime state."""
    ident = job['id']
    out = Path(job['output_dir'])
    out.mkdir(parents=True, exist_ok=True)

    # Remove control flags left from previous executions.
    for flag in ('pause.request', 'stop.request', 'force_stop.request',):
        (out / flag).unlink(missing_ok=True)

    spec = discover().get(job['analysis_id'])
    if not spec:
        repo.update_job(
            ident,
            status='failed',
            ended_at=now(),
            error='Analysis manifest missing',
            phase='Analysis unavailable',
        )
        return

    # Configure the analysis subprocess.
    env = os.environ.copy()
    env['PYTHONPATH'] = (str(ROOT) + os.pathsep + env.get('PYTHONPATH', ''))
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'

    # Capture additional Python tracebacks during native crashes.
    # Inherited by continuation and periodic-orbit subprocesses.
    env['PYTHONFAULTHANDLER'] = '1'
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
        if sys.platform == 'win32'
        else 0
    )
    cmd = [
        worker_python(),
        '-u',
        str(spec.script),
        '--config',
        str(out / 'config.json'),
        '--output',
        str(out),
    ]
    dump_json(
        out / 'metadata.json',
        {
            'run_id': ident,
            'analysis_id': spec.id,
            'created_at': job['created_at'],
            'started_at': now(),
            'python': sys.version.split()[0],
            'command': cmd,
            'status': 'running',
        },
    )

    # Save the initial configuration and launch metadata.
    export_debug(repo, job)

    # Redirect subprocess output into persistent run logs.
    with (
        (out / 'stdout.log').open('a', encoding='utf-8') as stdout,
        (out / 'stderr.log').open('a', encoding='utf-8') as stderr,
    ):
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=flags,
                start_new_session=(sys.platform != 'win32'),
            )

        except OSError as exc:
            repo.update_job(
                ident,
                status='failed',
                ended_at=now(),
                error=str(exc),
                phase='Launch failed',
            )
            return

        repo.update_job(ident, pid=proc.pid, phase='Running',)
        last_db_update = 0
        forced = False
        last_progress = 0.0
        last_phase = 'Running'

        # Schedule periodic debugging snapshots.
        next_debug_export = time.monotonic() + UPDATE_INTERVAL
        while proc.poll() is None:
            heartbeat()

            # Handle immediate termination requests.
            if (out / 'force_stop.request').exists():
                forced = True
                force_terminate(proc)
                break

            t = time.monotonic()

            # Read analysis progress and update the job database.
            if t - last_db_update >= 0.5:
                try:
                    info = json.loads((out / 'progress.json').read_text(encoding='utf-8'))
                    last_progress = float(info.get('progress', 0.0))
                    last_phase = str(info.get('phase', 'Running'))
                    repo.update_job(ident, progress=last_progress, phase=last_phase,)

                except (OSError, ValueError, TypeError):
                    pass

                last_db_update = t

            # Export current diagnostics, including the latest progress.
            # The full simulation output remains in results/.
            if t >= next_debug_export:
                export_debug(repo, job)
                next_debug_export = (time.monotonic() + UPDATE_INTERVAL)

            time.sleep(0.15)

        code = proc.wait()

    # Determine the final analysis status.
    stopped = (forced or (out / 'stop.request').exists() or code == 130)
    status = ('stopped' if stopped else ('completed' if code == 0 else 'failed'))
    err = ('' if status != 'failed' else f'Exit code {code}; see stderr.log')
    registered = 0

    # Register generated parameter seeds.
    if status == 'completed' and (out / 'seeds.csv').exists():
        repo.update_job(ident, phase='Indexing generated Seed-IDs', progress=98,)
        try:
            registered = repo.register_seeds(ident, out / 'seeds.csv',)

        except (OSError, ValueError, TypeError) as exc:
            status = 'failed'
            err = f'Could not register generated seeds: {exc}'

    # Every completed analysis is reusable as a seed,
    # even without seeds.csv.
    repo.update_job(
        ident,
        status=status,
        ended_at=now(),
        exit_code=code,
        progress=100 if status == 'completed' else last_progress,
        phase=status.capitalize(),
        error=err,
    )
    if status == 'completed':
        repo.learn_completed_runtimes()
        try:
            repo.register_run_seed(ident)

        except (ValueError, TypeError, OSError) as exc:
            repo.update_job(ident, phase='Completed; could not index run seed', error=str(exc),)

    # Update persistent run metadata.
    try:
        meta = json.loads((out / 'metadata.json').read_text(encoding='utf-8'))
        meta.update({
            'status': status,
            'ended_at': now(),
            'exit_code': code,
            'registered_seeds': registered,
        })
        dump_json(out / 'metadata.json', meta)

    except (OSError, ValueError):
        pass


def run_job(repo, job):
    """Execute an analysis with automatic debugging snapshots.

    The final export is attempted even when an unexpected exception
    occurs inside the worker.
    """
    failure = None

    # Initial snapshot, including the original job configuration.
    export_debug(repo, job)
    try:
        return _run_job(repo, job)

    except BaseException:
        failure = traceback.format_exc()
        raise

    finally:
        # Export final status, logs and numerical diagnostics.
        export_debug(repo, job, final=True, failure=failure,)


def work_loop(once=False):
    """Sequential background scheduler."""
    repo = Repository()
    lock = STATE / 'worker.lock'
    if not acquire_lock(lock):
        print('Worker already running', file=sys.stderr)
        return 2

    try:
        # Jobs orphaned by a previous worker crash
        # cannot be safely claimed again.
        with repo.connect() as db:
            db.execute(
                "UPDATE jobs "
                "SET status='interrupted', "
                "phase='Worker unexpectedly stopped' "
                "WHERE status='running'"
            )

        while True:
            heartbeat()
            job = repo.claim_next()
            if job:
                run_job(repo, job)

            elif once:
                break

            else:
                time.sleep(0.6)

            if once:
                break

    except KeyboardInterrupt:
        pass

    finally:
        lock.unlink(missing_ok=True)
        (STATE / 'worker_heartbeat.json').unlink(missing_ok=True)

    return 0


def worker_is_alive():
    try:
        data = json.loads((STATE / 'worker_heartbeat.json').read_text(encoding='utf-8'))
        return (time.time() - data['updated'] < 5 and alive(int(data['pid'])))

    except (OSError, ValueError, KeyError):
        return False


def start_worker():
    """Start the independent worker if it is not already running."""
    if worker_is_alive():
        return True

    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
        if sys.platform == 'win32'
        else 0
    )
    subprocess.Popen(
        [worker_python(), str(ROOT / 'start_worker.py')],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        start_new_session=(sys.platform != 'win32'),
        close_fds=True,
        env=os.environ.copy(),
    )
    return True
