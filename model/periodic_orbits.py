
"""MSP periodic-orbit discovery, continuation and process isolation."""

from dataclasses import asdict, replace
from pathlib import Path

import csv
import itertools
import json
import os
import subprocess
import sys
import time

from uuid import uuid4

import numpy as np

from model.shared_eqn import solve_equilibria
from model.periodic_config import (
    periodic_options,
    PERIODIC_DEPTH_BUDGET,
    PERIODIC_MAX_ATTEMPTS,
)
from model.periodic_math import (
    admissible_msp,
    absence_certificate,
    deterministic_guesses,
    same_orbit,
    orbit_signature,
)


class PeriodicBudgetReached(Exception):
    """The periodic search exhausted its computational budget."""


def parameters_at(base, axes, uv):
    return replace(
        base,
        **{
            axis.name: float(axis.physical(value))
            for axis, value in zip(axes, uv)
        },
    )


def equilibrium_evidence(p, known=None):
    """Distinguish certified exclusion from unsuccessful discovery."""

    if known is not None:
        msp = [
            row for row in known
            if admissible_msp(row, p)
        ]

        if msp:
            return dict(
                status='msp_present',
                roots=known,
                msp=msp,
                source='reused_equilibria',
            )

    certificate = absence_certificate(p)

    if certificate:
        return dict(
            status='msp_absent_certified',
            roots=known or [],
            msp=[],
            reason=certificate,
        )

    try:
        roots = [
            dict(
                community=e.community,
                stability=e.stability.classification,
                **dict(zip('MSP', e.biomass)),
            )
            for e in solve_equilibria(p, grid_points=513)
        ]

        msp = [
            row for row in roots
            if admissible_msp(row, p)
        ]

        return dict(
            status='msp_present' if msp else 'msp_not_found',
            roots=roots,
            msp=msp,
            source='reduced_solver',
            reason=(
                '' if msp
                else 'Finite equilibrium search is inconclusive'
            ),
        )

    except (
        ValueError,
        RuntimeError,
        ArithmeticError,
        np.linalg.LinAlgError,
    ) as exc:
        return dict(
            status='equilibrium_discovery_failed',
            roots=known or [],
            msp=[],
            reason=str(exc),
        )


def run_isolated(
    task, base, axes, options, folder,
    checkpoint, on_event=lambda _: None,
):
    """Execute one native attempt in a disposable process."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    request = folder / f'request_{uuid4().hex}.json'

    request.write_text(
        json.dumps(dict(
            task=task,
            base=asdict(base),
            axes=[asdict(a) for a in axes],
            options=options,
        )),
        encoding='utf-8',
    )

    env = os.environ.copy()

    env.update({
        key: '1'
        for key in (
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS',
            'BLIS_NUM_THREADS',
            'PYTHONFAULTHANDLER',
        )
    })

    cwd = Path(__file__).resolve().parents[1]

    events = []
    position = 0
    pending = ''
    started = time.monotonic()
    process = None
    reason = 'completed'
    covered = False

    def read_events():
        """Read only new, complete JSONL records."""

        nonlocal position, pending, covered

        path = request.with_suffix('.jsonl')

        if not path.exists():
            return

        with path.open(encoding='utf-8') as stream:
            stream.seek(position)
            pending += stream.read()
            position = stream.tell()

        lines = pending.split('\n')
        pending = lines.pop()

        for line in lines:
            if not line:
                continue

            item = json.loads(line)
            events.append(item)

            if not covered and on_event(item) is False:
                covered = True

    with request.with_suffix('.log').open(
        'w', encoding='utf-8'
    ) as log:

        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    '-m',
                    'model.periodic_worker',
                    str(request.resolve()),
                ],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(
                    subprocess, 'CREATE_NO_WINDOW', 0
                ),
            )

            while process.poll() is None:
                read_events()

                if covered:
                    reason = 'covered_branch'

                    if process.poll() is None:
                        process.kill()

                    break

                elapsed = time.monotonic() - started

                checkpoint(
                    f'Periodic {task["kind"]} '
                    f'· {elapsed:.0f} s '
                    f'· {len(events)} records'
                )

                if elapsed > options['periodic_timeout']:
                    reason = 'worker_timeout'

                    if process.poll() is None:
                        process.kill()

                    break

                time.sleep(.2)

            process.wait()
            read_events()

            if reason == 'completed':
                if any(e['type'] == 'failure' for e in events):
                    reason = 'solver_failed'

                elif (
                    process.returncode != 0
                    or not any(
                        e['type'] == 'complete'
                        for e in events
                    )
                ):
                    reason = 'native_worker_crashed'

        except OSError as exc:
            if process is not None:
                raise

            reason = 'launch_failed'

            events.append(dict(
                type='failure',
                reason=str(exc),
            ))

        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    return dict(
        reason=reason,
        exit_code=(
            process.returncode
            if process is not None else None
        ),
        events=events,
        seconds=time.monotonic() - started,
    )


def _key(uv):
    return tuple(round(float(v), 11) for v in uv)


def matching_branch(
    row, branches, dimensions, tolerance=.003
):
    """Identify matching orbits without merging distinct branches."""

    keys = 'uv'[:dimensions]

    uv = np.array([row[key] for key in keys])
    signature = orbit_signature(row)

    for branch in branches:

        for old in branch['points']:
            distance = max(
                abs(row[key] - old[key])
                for key in keys
            )

            if distance < 1e-5 and same_orbit(row, old):
                return branch['id']

        for arm in branch['arms']:
            path = [
                branch['points'][0],
                *arm['records'],
            ]

            for a, b in zip(path, path[1:]):
                start = np.array([a[k] for k in keys])
                delta = np.array([
                    b[k] - a[k] for k in keys
                ])

                norm = float(delta @ delta)

                if norm < 1e-16 or norm > .03**2:
                    continue

                fraction = float(
                    (uv - start) @ delta / norm
                )

                if not 0 <= fraction <= 1:
                    continue

                if np.linalg.norm(
                    uv - start - fraction * delta
                ) > 1e-5:
                    continue

                expected = (
                    (1 - fraction) * orbit_signature(a)
                    + fraction * orbit_signature(b)
                )

                if np.max(
                    abs(signature - expected)
                ) < tolerance:
                    return branch['id']

    return None


def cached_equilibria(data, dimensions):
    """Reuse equilibria already computed by continuation."""

    cache = {}

    for row in data.get('scan', []):
        uv = _key([
            row[key] for key in 'uv'[:dimensions]
        ])

        cache.setdefault(uv, []).append(row)

    for path in data.get('paths', []):
        for row in path:
            if row['community'] != 'MSP':
                continue

            uv = _key([
                row[key] for key in 'uv'[:dimensions]
            ])

            roots = cache.setdefault(uv, [])

            if not any(
                old['community'] == 'MSP'
                and np.allclose(
                    [old[c] for c in 'MSP'],
                    [row[c] for c in 'MSP'],
                )
                for old in roots
            ):
                roots.append(row)

    for region in data.get('regions', []):
        for sample in region['samples']:
            if 'roots' in sample:
                uv = _key((sample['u'], sample['v']))
                cache[uv] = sample['roots']

    return cache


def _hopf_seeds(data, dimensions, count):
    """Collect native Hopf points and representative curve samples."""

    found = [
        row for row in data.get('events', [])
        if row.get('kind') == 'hopf'
        and row['community'] == 'MSP'
    ]

    for curve in data.get('curves', []):
        if (
            curve
            and curve[0]['kind'] == 'hopf'
            and curve[0]['community'] == 'MSP'
        ):
            indices = np.unique(np.linspace(
                0,
                len(curve) - 1,
                min(count, len(curve)),
                dtype=int,
            ))

            found.extend(curve[i] for i in indices)

    unique = {}

    for row in found:
        uv = _key([
            row[k] for k in 'uv'[:dimensions]
        ])

        entries = unique.setdefault(uv, [])

        if not any(
            np.allclose(
                [row[c] for c in 'MSP'],
                [old[c] for c in 'MSP'],
                rtol=1e-7,
                atol=1e-9,
            )
            for old in entries
        ):
            entries.append(row)

    return unique


def export_periodic(data, folder, depth):
    """Save verified solutions and incomplete-search information."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    payload = data['periodic_analysis']
    path = folder / f'periodic_H{depth:g}.json'
    temporary = path.with_suffix('.tmp')

    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding='utf-8',
    )

    # Retry transient Windows and OneDrive file-sharing locks.
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            break

        except PermissionError:
            if attempt == 4:
                raise

            time.sleep(.05 * 2**attempt)

    rows = [
        dict(row, branch=branch['id'], point=i)
        for branch in payload['branches']
        for i, row in enumerate(branch['points'])
    ]

    data['oscillations'] = rows

    fields = [
        'branch', 'point', 'status', 'community',
        'stability', 'period', 'H', 'u', 'v', 'x', 'y',
    ]

    fields += list(dict.fromkeys(
        key
        for row in rows
        for key in row
        if key not in fields
    ))

    with (
        folder / f'periodic_branches_H{depth:g}.csv'
    ).open('w', newline='', encoding='utf-8') as stream:

        writer = csv.DictWriter(
            stream, fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            {
                key: (
                    json.dumps(value)
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


def analyze_periodic(
    data, base, axes, values, checkpoint, folder
):
    """Search for and continue verified MSP periodic solutions."""

    options = periodic_options(values)
    dimensions = len(axes)
    cache = cached_equilibria(data, dimensions)

    started = time.monotonic()

    payload = dict(
        method='pyoomph periodic boundary-value problem',
        settings=options,
        excluded_communities=[
            '0', 'M', 'S', 'P', 'MS', 'MP', 'SP',
        ],
        exclusion_reason=(
            'Bendixson–Dulac exclusion assumed for '
            'the current SIMP formulation'
        ),
        locations=[],
        branches=[],
        hopf=[],
        attempts=[],
        coverage={},
        status='running',
    )

    data['periodic_analysis'] = payload
    data['oscillations'] = []

    # Only retained for direct low-level testing.
    # Ordinary continuation jobs always supply 'on'.
    if options['periodic_analysis'] == 'off':
        payload['status'] = 'disabled'
        export_periodic(data, folder, base.H)
        return

    hopf = _hopf_seeds(
        data, dimensions, options['periodic_samples']
    )

    grid = [
        _key(uv)
        for uv in itertools.product(
            np.linspace(
                0, 1, options['periodic_samples']
            ),
            repeat=dimensions,
        )
    ]

    reused = [
        uv for uv, roots in sorted(cache.items())
        if any(
            row['community'] == 'MSP'
            for row in roots
        )
    ]

    # Select representative existing MSP locations.
    if reused:
        indices = np.unique(np.linspace(
            0,
            len(reused) - 1,
            min(
                len(reused),
                options['periodic_samples']**dimensions,
            ),
            dtype=int,
        ))

        reused = [reused[i] for i in indices]

    queue = list(dict.fromkeys([
        *hopf,
        *reused,
        *grid,
    ]))

    scheduled = set(queue)
    attempted = set()
    completed = set()

    locations = {}
    evidence_cache = {}

    bank = []
    pending_traces = []

    counter = 0

    def update_coverage():
        payload['coverage'].update(
            checked_locations=len(attempted),
            completed_locations=len(completed),
            max_locations=options['periodic_max_locations'],
            native_attempts=counter,
            max_native_attempts=PERIODIC_MAX_ATTEMPTS,
            elapsed_seconds=time.monotonic() - started,
            budget_seconds=PERIODIC_DEPTH_BUDGET,
            unvisited_locations=[
                list(uv)
                for uv in sorted(scheduled - attempted)
            ],
            incomplete_locations=[
                list(uv)
                for uv in sorted(attempted - completed)
            ],
            limit_reached=bool(
                scheduled - completed
                or pending_traces
            ),
        )

    def save():
        update_coverage()
        export_periodic(data, folder, base.H)

    def check_budget():
        remaining = (
            PERIODIC_DEPTH_BUDGET
            - (time.monotonic() - started)
        )

        if counter >= PERIODIC_MAX_ATTEMPTS:
            raise PeriodicBudgetReached(
                'Maximum native attempt count reached'
            )

        if remaining < 5:
            raise PeriodicBudgetReached(
                'Periodic wall-time budget reached'
            )

        return remaining

    def attempt(task, on_event=lambda _: None):
        """Apply a shared budget to native periodic workers."""

        nonlocal counter

        remaining = check_budget()
        counter += 1

        task_options = dict(options)

        task_options['periodic_timeout'] = min(
            options['periodic_timeout'],
            remaining,
        )

        directory = (
            Path(folder)
            / f'periodic_workers_H{base.H:g}'
            / f'{counter:06d}'
        )

        def report(message):
            checkpoint(
                f'{message} '
                f'· attempt {counter}/{PERIODIC_MAX_ATTEMPTS}'
            )

        report(f'Periodic {task["kind"]}')

        result = run_isolated(
            task,
            base,
            axes,
            task_options,
            directory,
            report,
            on_event,
        )

        payload['attempts'].append(dict(
            task=counter,
            kind=task['kind'],
            uv=task.get(
                'uv',
                task.get('seed', {}).get('uv'),
            ),
            mode=task.get('mode'),
            amplitude=task.get('amplitude'),
            period=task.get('period'),
            reason=result['reason'],
            exit_code=result['exit_code'],
            seconds=result['seconds'],
            failures=[
                event['reason']
                for event in result['events']
                if event['type'] == 'failure'
            ],
        ))

        return result

    def get_location(uv, round_index=0):
        """Calculate the local MSP evidence at most once."""

        if uv in locations:
            return locations[uv], evidence_cache[uv]

        check_budget()
        attempted.add(uv)

        checkpoint(
            f'MSP periodic discovery '
            f'· location {len(attempted)} '
            f'· refinement {round_index}'
        )

        evidence = equilibrium_evidence(
            parameters_at(base, axes, uv),
            cache.get(uv),
        )

        location = dict(
            uv=list(uv),
            **{
                key: float(axis.physical(value))
                for key, axis, value in zip('xy', axes, uv)
            },
            equilibrium_status=evidence['status'],
            reason=evidence.get('reason', ''),
            msp_count=len(evidence['msp']),
            round=round_index,
            branches=[],
            status='search_pending',
        )

        locations[uv] = location
        evidence_cache[uv] = evidence

        payload['locations'].append(location)

        if evidence['status'] != 'msp_present':
            location['status'] = (
                'excluded_no_msp'
                if evidence['status'] == 'msp_absent_certified'
                else 'equilibrium_unresolved'
            )

            completed.add(uv)

        return location, evidence

    def accept(item, location):
        """Register a verified orbit and queue its continuation."""

        row = item['row']
        seed = item['seed']

        duplicate = matching_branch(
            row, payload['branches'], dimensions
        )

        if duplicate is not None:
            return duplicate

        branch = dict(
            id=len(payload['branches']) + 1,
            origin=location,
            points=[row],
            arms=[],
        )

        payload['branches'].append(branch)

        bank.append(dict(
            row=row,
            seed=seed,
            branch=branch['id'],
        ))

        # Queue tracing instead of launching it immediately.
        # This allows other Hopf seeds to be tested first.
        pending_traces.append((branch, seed))

        return branch['id']

    def process_events(result, location):
        for item in result['events']:

            if item['type'] == 'hopf':
                payload['hopf'].append(item)

            elif item['type'] == 'orbit':
                ident = accept(item, location['uv'])

                if ident not in location['branches']:
                    location['branches'].append(ident)

                location['status'] = 'verified_periodic'

        save()

    def trace_pending():
        """Continue discovered periodic branches in both directions."""

        while pending_traces:

            check_budget()

            branch, seed = pending_traces[0]

            for axis in range(dimensions):
                for direction in (-1, 1):

                    check_budget()

                    arm = []

                    arm_info = dict(
                        axis=axis,
                        direction=direction,
                        records=arm,
                        termination='running',
                    )

                    branch['arms'].append(arm_info)

                    def on_event(event):
                        if event['type'] != 'orbit':
                            return

                        value = event['row']

                        old = matching_branch(
                            value,
                            [
                                other
                                for other in payload['branches']
                                if other['id'] != branch['id']
                            ],
                            dimensions,
                        )

                        if old is not None:
                            arm_info['joins_branch'] = old
                            return False

                        arm.append(value)
                        branch['points'].append(value)

                        bank.append(dict(
                            row=value,
                            seed=event['seed'],
                            branch=branch['id'],
                        ))

                        # Save every verified point, even if the next
                        # native arclength step causes a crash.
                        save()

                    result = attempt(
                        dict(
                            kind='trace',
                            seed=seed,
                            axis=axis,
                            direction=direction,
                        ),
                        on_event,
                    )

                    termination = result['reason']

                    if termination == 'completed':
                        termination = next(
                            (
                                event['reason']
                                for event in reversed(
                                    result['events']
                                )
                                if event['type'] == 'termination'
                            ),
                            'completed',
                        )

                    arm_info.update(
                        termination=termination,
                        notes=[
                            event
                            for event in result['events']
                            if event['type'] in (
                                'termination', 'failure'
                            )
                        ],
                    )

                    save()

            pending_traces.pop(0)
            save()

    def discover_at(uv, round_index):
        """Search independently for additional periodic solutions."""

        location, evidence = get_location(
            uv, round_index
        )

        if uv in completed:
            return location

        location['status'] = (
            'verified_periodic'
            if location['branches']
            else 'orbit_discovery_unsuccessful'
        )

        tasks = []

        # Reuse at most one nearby guess per existing branch,
        # limited to two branches per search location.
        nearby = sorted(
            bank,
            key=lambda entry: np.linalg.norm(
                np.asarray(entry['seed']['uv']) - uv
            ),
        )

        used = set()

        for entry in nearby:
            if entry['branch'] in used:
                continue

            distance = np.linalg.norm(
                np.asarray(entry['seed']['uv']) - uv
            )

            if distance > .25:
                continue

            used.add(entry['branch'])

            tasks.append(dict(
                kind='reuse',
                seed=entry['seed'],
                equilibrium=evidence['msp'][0],
            ))

            if len(used) >= 2:
                break

        # Independent guesses are still needed because disconnected
        # periodic branches may not originate at a detected Hopf.
        for root in evidence['msp']:
            tasks.extend(deterministic_guesses(
                root,
                parameters_at(base, axes, uv),
                options,
            ))

        for task in tasks:
            task['uv'] = list(uv)

            result = attempt(task)
            process_events(result, location)

        completed.add(uv)
        save()

        return location

    try:
        save()

        # Stage 1: prioritize all available native Hopf seeds.
        for uv, roots in hopf.items():

            if (
                uv not in attempted
                and len(attempted) >= options['periodic_max_locations']
            ):
                break

            location, evidence = get_location(uv)

            if evidence['status'] != 'msp_present':
                continue

            for row in roots:

                task = dict(
                    kind='hopf',
                    equilibrium=row,
                    axis=int(row.get('solved_axis', 0)),
                    uv=list(uv),
                )

                result = attempt(task)
                process_events(result, location)

        # Stage 2: continue the discovered Hopf branches.
        trace_pending()

        # Stage 3: independent periodic discovery.
        rounds = options['periodic_refinement_rounds']

        for round_index in range(rounds + 1):

            this_round = []

            for uv in queue:
                uv = _key(uv)

                if uv in completed:
                    continue

                if (
                    uv not in attempted
                    and len(attempted)
                    >= options['periodic_max_locations']
                ):
                    break

                check_budget()

                location = discover_at(uv, round_index)
                this_round.append(location)

                # Continue newly discovered non-Hopf branches.
                trace_pending()

            # Stage 4: refine selected parameter locations.
            if round_index == rounds:
                break

            refinement = []

            step = (
                1
                / (options['periodic_samples'] - 1)
                / 2**(round_index + 1)
            )

            for location in this_round:
                if (
                    location['equilibrium_status']
                    != 'msp_present'
                    and location['status']
                    != 'equilibrium_unresolved'
                ):
                    continue

                for axis in range(dimensions):
                    for sign in (-1, 1):

                        uv = list(location['uv'])
                        uv[axis] += sign * step

                        if (
                            0 <= uv[axis] <= 1
                            and _key(uv) not in attempted
                        ):
                            refinement.append(_key(uv))

            queue = list(dict.fromkeys(refinement))
            scheduled.update(queue)

        update_coverage()

        unresolved = any(
            location['status'] in (
                'equilibrium_unresolved',
                'orbit_discovery_unsuccessful',
                'search_pending',
            )
            for location in payload['locations']
        )

        failed_attempts = any(
            result['reason'] in (
                'worker_timeout',
                'native_worker_crashed',
                'solver_failed',
                'launch_failed',
            )
            for result in payload['attempts']
        )

        incomplete_arms = any(
            arm['termination'] not in (
                'covered_branch',
                'closed_loop',
                'range_boundary',
                'parameter_boundary',
            )
            for branch in payload['branches']
            for arm in branch['arms']
        )

        payload['status'] = (
            'completed_with_unresolved_searches'
            if (
                unresolved
                or failed_attempts
                or incomplete_arms
                or payload['coverage']['limit_reached']
            )
            else 'completed'
        )

    except PeriodicBudgetReached as exc:

        payload['status'] = 'budget_exhausted'
        payload['reason'] = str(exc)

        for branch in payload['branches']:
            for arm in branch['arms']:
                if arm['termination'] == 'running':
                    arm['termination'] = 'budget_exhausted'

        for uv in attempted - completed:
            if locations[uv]['status'] == 'search_pending':
                locations[uv]['status'] = 'search_incomplete'

        checkpoint(
            'Periodic search budget reached '
            '· retaining verified results'
        )

    finally:
        save()

        if payload['status'] == 'budget_exhausted':
            payload['coverage']['limit_reached'] = True
            export_periodic(data, folder, base.H)
