"""Export compact run diagnostics for remote debugging via GitHub."""

from __future__ import annotations

import json
import shutil
import time
import sys
from pathlib import Path

from .paths import ROOT

DEBUG_ROOT = ROOT / "debug" / "runs"

# Debugging settings
KEEP_RUNS = 10
UPDATE_INTERVAL = 30.0
MAX_FILE_SIZE = 1024 * 1024       # 1 MiB
MAX_LOG_SIZE = 128 * 1024         # 128 KiB
MAX_DIAGNOSTIC_FILES = 64
MAX_PERIODIC_WORKERS = 8

CORE_FILES = (
    "config.json",
    "metadata.json",
    "progress.json",
    "summary.json",
    "stdout.log",
    "stderr.log",
)

DIAGNOSTIC_PATTERNS = (
    "diagnostics_*.json",
    "performance_*.json",
    "geometry_*.json",
    "refinement_*.json",
    "regions_*.json",
    "periodic_H*.json",
    "events_*.csv",
    "branches_*.csv",
    "stable_orbit_scan_*.csv",
    "figures_revised/hopf_audit_*.json",
    "figures_revised/bifurcation*.csv",
    "figures_revised/bifurcation*.json",
    "bifurcations_*.csv",
    "bifurcation_summary*.json",
    "bifurcation_audit_failures_*.csv",
    "hopf_audit_H*.json",
    "stable_orbit_summary_H*.json",
    "periodic_error_*.txt",
    "equilibrium_preview_error_*.txt",
)


def log_export_warning(message):
    """Persist export warnings even when the worker has no console."""
    message = f'{time.strftime("%Y-%m-%d %H:%M:%S")} | {message}'
    try:
        path = DEBUG_ROOT.parent / 'export_errors.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(message + '\n')
    except OSError:
        pass
    if sys.stderr is not None:
        print(message, file=sys.stderr)


def scientific_job(job):
    """Keep test-fixture jobs out of scientific diagnostic storage."""
    from .paths import SCRIPTS
    fixtures = ROOT / 'tests' / 'fixtures'
    return (not str(job.get('analysis_id', '')).endswith('_demo')
            and not SCRIPTS.resolve().is_relative_to(fixtures.resolve()))


def remove_debug_tree(path):
    """Retry read-only entries; defer locked Windows directories."""
    import os
    import stat

    def recover(func, name, info):
        error = info[1]
        if not isinstance(error, PermissionError):
            raise error
        # Never follow a symlink when changing file attributes.
        if Path(name).is_symlink():
            raise error
        os.chmod(name, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        func(name)

    try:
        shutil.rmtree(path, onerror=recover)
        return True
    except FileNotFoundError:
        if not Path(path).exists():
            return True
        log_export_warning(f'Cleanup deferred: {path}: directory changed during removal')
        return False
    except OSError as exc:
        log_export_warning(f'Cleanup deferred: {path}: {exc}')
        return False


def atomic_write(path, data):
    """Use unique temporary files and retry transient sharing locks."""
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.debugtmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def copy_diagnostic(source, target):
    """Copy small diagnostic files or the tail of large logs."""
    if source.suffix in (".log", ".jsonl"):
        with source.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - MAX_LOG_SIZE))
            data = stream.read(MAX_LOG_SIZE)

        if size > MAX_LOG_SIZE:
            data = b"[Earlier log output omitted]\n" + data

    else:
        with source.open("rb") as stream:
            data = stream.read(MAX_FILE_SIZE + 1)

        if len(data) > MAX_FILE_SIZE:
            target.unlink(missing_ok=True)
            return False

    atomic_write(target, data)
    return True


def cleanup_old_runs():
    """Retain recent scientific runs; never delete active snapshots."""
    folders = []
    for path in DEBUG_ROOT.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            marker = path / 'job_status.json'
            if not marker.is_file():
                continue
            status = json.loads(marker.read_text(encoding='utf-8'))
            if status.get('status') in ('running', 'paused'):
                continue
            if str(status.get('analysis_id', '')).endswith('_demo'):
                remove_debug_tree(path)
                continue
            folders.append((str(status.get('started_at') or ''), marker.stat().st_mtime, path))
        except (OSError, ValueError) as exc:
            log_export_warning(f'Cannot inspect {path.name}: {exc}')
    folders.sort(key=lambda item: item[:2], reverse=True)
    for _, _, folder in folders[KEEP_RUNS:]:
        remove_debug_tree(folder)


def export_debug(repo, job, final=False, failure=None):
    """Export diagnostics without interrupting the actual analysis."""
    if not scientific_job(job):
        return False
    try:
        ident = str(job["id"])
        source = Path(job["output_dir"])
        target = DEBUG_ROOT / ident
        target.mkdir(parents=True, exist_ok=True)
        current = repo.get_job(ident) or job
        status = {
            key: current.get(key)
            for key in (
                "id", "name", "analysis_id", "status",
                "progress", "phase", "error", "exit_code",
                "created_at", "started_at", "ended_at",
            )
        }
        status.update(
            exported_at=time.time(),
            final_snapshot=final,
            python=sys.version,
            worker_exception=failure,
        )
        files = [source / name for name in CORE_FILES]
        extra = sorted({path for pattern in DIAGNOSTIC_PATTERNS for path in source.glob(pattern)})
        files.extend(extra[:MAX_DIAGNOSTIC_FILES])

        # Export recent native workers during AND after the run.
        requests = sorted(
            source.glob("periodic_workers_H*/*/request_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for request in requests[:MAX_PERIODIC_WORKERS]:
            files.extend((request, request.with_suffix(".jsonl"), request.with_suffix(".log"),))

        # Keep only a bounded number of worker snapshots per job.
        keep = {
            str(request.parent.relative_to(source))
            for request in requests[:MAX_PERIODIC_WORKERS]
        }
        for directory in target.glob("periodic_workers_H*/*"):
            if directory.is_dir() and not directory.is_symlink():
                relative = str(directory.relative_to(target))
                if relative not in keep:
                    remove_debug_tree(directory)

        skipped = [str(path.relative_to(source)) for path in extra[MAX_DIAGNOSTIC_FILES:]]
        for path in files:
            if not path.is_file():
                continue

            relative = path.relative_to(source)
            try:
                copied = copy_diagnostic(path, target / relative)
                if not copied:
                    skipped.append(str(relative))
            except OSError as exc:
                skipped.append(str(relative))
                log_export_warning(f"Copy failed: {relative}: {exc}")

        summarize_periodic(source, target)
        status["skipped_files"] = skipped
        atomic_write(target / "job_status.json", json.dumps(status, indent=2).encode("utf-8"),)
        try:
            cleanup_old_runs()
        except OSError as exc:
            log_export_warning(f"Cleanup deferred: {exc}")
        return True

    except Exception as exc:
        # Debugging must never cause a scientific run to fail.
        log_export_warning(f"Debug export failed: {exc}")
        return False


def summarize_periodic(source, target):
    """Retain useful diagnostics even when raw output exceeds 1 MB."""
    from collections import Counter
    from statistics import median
    for path in source.glob('periodic_H*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            attempts = data.get('attempts', [])
            locations = data.get('locations', [])
            durations = [float(a['seconds']) for a in attempts if 'seconds' in a]
            coverage = dict(data.get('coverage', {}))
            for key in ('unvisited_locations', 'incomplete_locations',):
                if isinstance(coverage.get(key), list):
                    values = coverage.pop(key)
                    coverage[key+'_count'] = len(values)
                    coverage[key+'_examples'] = values[:10]

            branches = []
            for branch in data.get('branches', []):
                points = branch.get('points', [])
                branches.append(dict(
                    id=branch['id'],
                    points=len(points),
                    stability=dict(Counter(
                        row['stability'] for row in points
                    )),
                    mesh_sizes=dict(Counter(
                        row.get('nodes') for row in points
                    )),
                    first_point=points[0] if points else None,
                    last_point=points[-1] if points else None,
                    arms=[
                        dict(
                            axis=arm['axis'],
                            direction=arm['direction'],
                            termination=arm['termination'],
                            points=len(arm['records']),
                            notes=arm.get('notes', []),
                        )
                        for arm in branch.get('arms', [])
                    ],
                ))

            summary = dict(
                status=data.get('status'),
                settings=data.get('settings'),
                algorithm=data.get('algorithm'),
                timing=data.get('timing'),
                active_attempt=data.get('active_attempt'),
                slowest_searches=sorted(attempts, key=lambda a: a.get('seconds', 0), reverse=True)[:5],
                coverage=coverage,
                attempts=len(attempts),
                attempt_types=dict(Counter(
                    a.get('kind') for a in attempts
                )),
                attempt_outcomes=dict(Counter(
                    a.get('reason') for a in attempts
                )),
                total_worker_seconds=sum(durations),
                median_attempt_seconds=(
                    median(durations) if durations else None
                ),
                location_statuses=dict(Counter(
                    x.get('status') for x in locations
                )),
                branches=branches,
                hopf=data.get('hopf', []),
                recent_attempts=attempts[-20:],
            )
            name = path.name.replace('periodic_', 'periodic_summary_', 1)
            atomic_write(target/name, json.dumps(summary, indent=2, default=str).encode('utf-8'),)

        except (OSError, ValueError, TypeError, KeyError) as exc:
            log_export_warning(f'Periodic diagnostic summary failed: {exc}')


def backfill_debug():
    """Recover recent scientific runs from existing results."""
    from .storage import Repository
    repo = Repository()
    query = ('SELECT * FROM jobs WHERE started_at IS NOT NULL '
             'ORDER BY started_at DESC')
    with repo.connect() as db:
        jobs = [dict(row) for row in db.execute(query)]
    selected = [job for job in jobs
                if scientific_job(job) and Path(job['output_dir']).is_dir()][:KEEP_RUNS]
    successful = 0
    for job in reversed(selected):
        exported = export_debug(repo, job, final=job['status'] not in ('running', 'paused'))
        if exported:
            successful += 1
            print('Snapshot written:', job['id'], job['name'])
        else:
            print('Export failed:', job['id'], job['name'])
    print(f'Exported {successful}/{len(selected)} scientific runs')
    print('Diagnostic directory:', DEBUG_ROOT)
    print('Warnings:', DEBUG_ROOT.parent / 'export_errors.log')

if __name__ == '__main__':
    backfill_debug()
