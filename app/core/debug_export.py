
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
KEEP_RUNS = 5
UPDATE_INTERVAL = 30.0
MAX_FILE_SIZE = 1024 * 1024       # 1 MiB
MAX_LOG_SIZE = 128 * 1024         # 128 KiB
MAX_DIAGNOSTIC_FILES = 24
MAX_PERIODIC_WORKERS = 3

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
)


def atomic_write(path, data):
    """Avoid exposing partially written debugging files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".debugtmp")
    temp.write_bytes(data)
    temp.replace(path)


def copy_diagnostic(source, target):
    """Copy small diagnostic files or the tail of large logs."""
    if source.suffix == ".log":
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
    """Keep only the most recent automatically exported runs."""
    folders = [
        path for path in DEBUG_ROOT.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and (path / "job_status.json").is_file()
    ]

    folders.sort(
        key=lambda path: (
            path / "job_status.json"
        ).stat().st_mtime,
        reverse=True,
    )

    for folder in folders[KEEP_RUNS:]:
        shutil.rmtree(folder)


def export_debug(repo, job, final=False, failure=None):
    """Export diagnostics without interrupting the actual analysis."""
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

        files = [
            source / name for name in CORE_FILES
        ]

        extra = sorted({
            path
            for pattern in DIAGNOSTIC_PATTERNS
            for path in source.glob(pattern)
        })

        files.extend(extra[:MAX_DIAGNOSTIC_FILES])

        # Include recent isolated periodic-orbit worker diagnostics.
        if final:
            requests = sorted(
                source.glob(
                    "periodic_workers_H*/*/request_*.json"
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )

            for request in requests[:MAX_PERIODIC_WORKERS]:
                files.extend((
                    request,
                    request.with_suffix(".jsonl"),
                    request.with_suffix(".log"),
                ))

        skipped = [
            str(path.relative_to(source))
            for path in extra[MAX_DIAGNOSTIC_FILES:]
        ]

        for path in files:
            if not path.is_file():
                continue

            relative = path.relative_to(source)

            try:
                copied = copy_diagnostic(
                    path, target / relative
                )
                if not copied:
                    skipped.append(str(relative))
            except OSError:
                skipped.append(str(relative))

        status["skipped_files"] = skipped

        atomic_write(
            target / "job_status.json",
            json.dumps(status, indent=2).encode("utf-8"),
        )

        cleanup_old_runs()

    except Exception as exc:
        # Debugging must never cause a scientific run to fail.
        print(
            f"Debug export failed: {exc}",
            file=sys.stderr,
        )
