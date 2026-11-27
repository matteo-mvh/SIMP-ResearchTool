
"""Mathematically selective discovery and continuation of MSP periodic orbits."""

from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import csv
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

from model.shared_eqn import solve_equilibria
from model.periodic_config import periodic_options
from model.periodic_math import (
    admissible_msp,
    absence_certificate,
    deterministic_guesses,
    same_orbit,
    orbit_signature,
)


def parameters_at(base, axes, uv):
    return replace(base, **{
        axis.name: float(axis.physical(value))
        for axis, value in zip(axes, uv)
    })


def orbit_exclusion(p, margin=1e-10):
    """Sufficient conditions excluding strictly positive MSP cycles.

    A positive periodic orbit requires zero average per-capita growth
    for every resident species over a complete period.
    """

    if p.N_wc <= 0:
        return 'No nitrogen inventory available for M and P'

    # Upper bounds over the entire physically admissible state space.
    N = p.N_wc / p.H
    I_M = p.I_0 * np.exp(-p.H * p.kw)
    I_S = I_M
    I_P = p.I_0 * np.exp(-p.H * p.kw / 2)

    growth = {
        'M': (
            p.mu_M * I_M / (p.K_I_M + I_M)
            * N / (p.K_N_M + N)
            - p.m_M
        ),
        'S': (
            p.mu_S * I_S / (p.K_I_S + I_S)
            - p.m_S
        ),
        'P': (
            p.mu_P * I_P / (p.K_I_P + I_P)
            * N / (p.K_N_P + N)
            - p.m_P - p.g_P
        ),
    }

    for species, maximum in growth.items():
        if maximum < -margin:
            return (
                f'{species}: maximum possible per-capita '
                f'growth is negative ({maximum:.8g} d^-1)'
            )

    # Without feedback from S to M, the M-P subsystem is autonomous.
    # It is competitive and cannot support nonconstant periodic orbits.
    # Once M and P are constant, S follows a scalar autonomous ODE.
    if p.g_M == 0:
        return 'No S-to-M feedback; autonomous competitive M-P subsystem'

    # Without incoming feedback to S, S is an autonomous scalar ODE.
    if p.k_M == 0 and p.k_P == 0 and p.m_SM == 0:
        return 'S is independent; remaining subsystem is competitive'

    return None


def equilibrium_evidence(p, known=None):
    """Distinguish rigorous orbit exclusion from missing equilibrium seeds."""

    if known is not None:
        roots = [row for row in known if admissible_msp(row, p)]
        if roots:
            return dict(
                status='msp_present', roots=known,
                msp=roots, source='reused_equilibria',
            )

    exclusion = orbit_exclusion(p)

    if exclusion:
        return dict(
            status='msp_absent_certified',
            roots=known or [],
            msp=[],
            reason=exclusion,
        )

    # The reduced equilibrium certificate is useful information,
    # but is not used as an unconditional global orbit exclusion.
    hint = absence_certificate(p)

    if hint:
        return dict(
            status='msp_not_found',
            roots=known or [],
            msp=[],
            reason='No MSP equilibrium under the reduced bounds: ' + hint,
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

        msp = [row for row in roots if admissible_msp(row, p)]

        return dict(
            status='msp_present' if msp else 'msp_not_found',
            roots=roots,
            msp=msp,
            source='reduced_solver',
            reason='' if msp else 'No admissible MSP equilibrium found',
        )

    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
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
    """Run one native attempt in a disposable process.

    Timeout is based on the last completed native record.
    A productive continuation arm may run for longer than the timeout.
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    request = folder / f'request_{uuid4().hex}.json'

    request.write_text(json.dumps(dict(
        task=task,
        base=asdict(base),
        axes=[asdict(a) for a in axes],
        options=options,
    )), encoding='utf-8')

    env = os.environ.copy()
    env.update({
        key: '1' for key in (
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
    started = last_record = time.monotonic()
    process = None
    reason = 'completed'
    covered = False

    def read_events():
        nonlocal position, pending, covered, last_record

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

            # A completed verified orbit resets the inactivity timer.
            if item['type'] in ('orbit', 'hopf', 'trace_progress'):
                last_record = time.monotonic()

            if not covered and on_event(item) is False:
                covered = True

    with request.with_suffix('.log').open('w', encoding='utf-8') as log:
        try:
            process = subprocess.Popen(
                [
                    sys.executable, '-m', 'model.periodic_worker',
                    str(request.resolve()),
                ],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )

            while process.poll() is None:
                read_events()

                if covered:
                    reason = 'covered_branch'
                    if process.poll() is None:
                        process.kill()
                    break

                now = time.monotonic()

                checkpoint(
                    f'Periodic {task["kind"]} · '
                    f'{now-started:.0f} s · '
                    f'{sum(e["type"] == "orbit" for e in events)} orbits'
                )

                if now - last_record > options['periodic_timeout']:
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
                elif process.returncode != 0 or not any(
                    e['type'] == 'complete' for e in events
                ):
                    reason = 'native_worker_crashed'

        except OSError as exc:
            if process is not None:
                raise
            reason = 'launch_failed'
            events.append(dict(type='failure', reason=str(exc)))

        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    return dict(
        reason=reason,
        exit_code=process.returncode if process is not None else None,
        events=events,
        seconds=time.monotonic() - started,
    )


def _key(uv):
    return tuple(round(float(value), 11) for value in uv)


def matching_branch(row, branches, dimensions, tolerance=.003):
    """Numerically identify previously covered periodic solutions."""

    keys = 'uv'[:dimensions]
    uv = np.array([row[k] for k in keys])
    signature = orbit_signature(row)

    def compatible(old):
        a, b = row.get('stability'), old.get('stability')
        return {a, b} != {'stable', 'unstable'}

    for branch in branches:
        for old in branch['points']:
            if (
                compatible(old)
                and max(abs(row[k]-old[k]) for k in keys) < 1e-5
                and same_orbit(row, old)
            ):
                return branch['id']

        for arm in branch['arms']:
            path = [branch['points'][0], *arm['records']]

            for a, b in zip(path, path[1:]):
                if not compatible(a) or not compatible(b):
                    continue

                start = np.array([a[k] for k in keys])
                delta = np.array([b[k]-a[k] for k in keys])
                norm = float(delta @ delta)

                if norm < 1e-16 or norm > .03**2:
                    continue

                fraction = float((uv-start) @ delta / norm)

                if not 0 <= fraction <= 1:
                    continue

                if np.linalg.norm(uv-start-fraction*delta) > 1e-5:
                    continue

                expected = (
                    (1-fraction)*orbit_signature(a)
                    + fraction*orbit_signature(b)
                )

                if np.max(abs(signature-expected)) < tolerance:
                    return branch['id']

    return None


def cached_equilibria(data, dimensions):
    """Collect distinct equilibria from existing continuation results."""

    cache = {}

    def add(row, uv):
        roots = cache.setdefault(_key(uv), [])
        if not any(
            row['community'] == old['community']
            and np.allclose(
                [row[c] for c in 'MSP'],
                [old[c] for c in 'MSP'],
                rtol=1e-7, atol=1e-9,
            )
            for old in roots
        ):
            roots.append(row)

    for row in data.get('scan', []):
        add(row, [row[k] for k in 'uv'[:dimensions]])

    for path in data.get('paths', []):
        for row in path:
            if row['community'] == 'MSP':
                add(row, [row[k] for k in 'uv'[:dimensions]])

    for region in data.get('regions', []):
        for sample in region['samples']:
            for row in sample.get('roots', []):
                add(row, (sample['u'], sample['v']))

    return cache


def _hopf_seeds(data, dimensions, count):
    """Collect native Hopf events and samples along 2D Hopf curves."""

    found = [
        row for row in data.get('events', [])
        if row.get('kind') == 'hopf' and row['community'] == 'MSP'
    ]

    for curve in data.get('curves', []):
        if not curve or curve[0]['kind'] != 'hopf':
            continue

        if curve[0]['community'] != 'MSP':
            continue

        indices = np.unique(np.linspace(
            0, len(curve)-1, min(count, len(curve)), dtype=int
        ))

        for i in indices:
            row = dict(curve[i])

            # Choose a correction axis transverse to the Hopf curve.
            if dimensions == 2 and len(curve) > 1:
                a = curve[max(0, i-1)]
                b = curve[min(len(curve)-1, i+1)]
                tangent = [abs(b[k]-a[k]) for k in 'uv']
                row['solved_axis'] = int(np.argmin(tangent))

            found.append(row)

    unique = {}

    for row in found:
        uv = _key([row[k] for k in 'uv'[:dimensions]])
        entries = unique.setdefault(uv, [])

        if not any(
            np.allclose(
                [row[c] for c in 'MSP'],
                [old[c] for c in 'MSP'],
                rtol=1e-7, atol=1e-9,
            )
            for old in entries
        ):
            entries.append(row)

    return unique


def export_periodic(data, folder, depth):
    """Save verified periodic points and search diagnostics."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    payload = data['periodic_analysis']
    path = folder / f'periodic_H{depth:g}.json'
    temporary = path.with_suffix('.tmp')

    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding='utf-8',
    )

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
        'branch', 'point', 'status', 'community', 'stability',
        'period', 'H', 'u', 'v', 'x', 'y',
    ]

    fields += list(dict.fromkeys(
        key for row in rows for key in row if key not in fields
    ))

    with (folder/f'periodic_branches_H{depth:g}.csv').open(
        'w', newline='', encoding='utf-8'
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        writer.writerows({
            k: json.dumps(v) if isinstance(v, (list, dict)) else v
            for k, v in row.items()
        } for row in rows)


def analyze_periodic(data, base, axes, values, checkpoint, folder):
    """Search without a global runtime, location or attempt budget."""

    options = periodic_options(values)
    dimensions = len(axes)

    # Higher parameter resolution permits longer native traces.
    options['periodic_max_steps'] = max(
        options['periodic_max_steps'],
        8 * max(axis.resolution for axis in axes),
    )

    cache = cached_equilibria(data, dimensions)
    started = time.monotonic()

    payload = dict(
        method='Native PyOOMPH periodic boundary-value problems',
        settings=options,
        excluded_communities=['0', 'M', 'S', 'P', 'MS', 'MP', 'SP'],
        exclusion_reason='Competitive lower-dimensional SIMP subsystems',
        locations=[],
        branches=[],
        hopf=[],
        attempts=[],
        coverage={},
        status='running',
    )

    data['periodic_analysis'] = payload
    data['oscillations'] = []

    if options['periodic_analysis'] == 'off':
        payload['status'] = 'disabled'
        export_periodic(data, folder, base.H)
        return

    samples = options['periodic_samples']

    hopf = _hopf_seeds(data, dimensions, samples)

    # Hopf points are already verified equilibrium seeds.
    for uv, roots in hopf.items():
        cache.setdefault(uv, []).extend(roots)

    grid = [
        _key(uv) for uv in itertools.product(
            np.linspace(0, 1, samples), repeat=dimensions
        )
    ]

    # Reuse spatially distributed equilibrium locations.
    reused = [
        uv for uv, roots in sorted(cache.items())
        if any(row['community'] == 'MSP' for row in roots)
    ]

    if reused:
        indices = np.unique(np.linspace(
            0, len(reused)-1,
            min(len(reused), samples**dimensions),
            dtype=int,
        ))
        reused = [reused[i] for i in indices]

    queue = list(dict.fromkeys([*hopf, *reused, *grid]))
    scheduled = set(queue)
    completed = set()

    locations = {}
    evidence_cache = {}
    bank = []
    pending_traces = []

    counter = 0

    def save():
        incomplete_arms = [
            dict(branch=branch['id'], axis=arm['axis'],
                 direction=arm['direction'], reason=arm['termination'])
            for branch in payload['branches']
            for arm in branch['arms']
            if arm['termination'] not in (
                'covered_branch', 'closed_loop', 'range_boundary',
                'approached_verified_hopf' 
            )
        ]

        payload['coverage'].update(
            scheduled_locations=len(scheduled),
            checked_locations=len(locations),
            completed_locations=len(completed),
            native_attempts=counter,
            elapsed_seconds=time.monotonic()-started,
            unvisited_locations=[
                list(uv) for uv in sorted(scheduled-set(locations))
            ],
            incomplete_locations=[
                list(uv) for uv in sorted(set(locations)-completed)
            ],
            pending_branches=[
                branch['id'] for branch, _ in pending_traces
            ],
            incomplete_arms=incomplete_arms,
            limit_reached=bool(
                scheduled-completed or pending_traces or incomplete_arms
            ),
        )

        export_periodic(data, folder, base.H)

    def attempt(task, on_event=lambda _: None):
        nonlocal counter
        counter += 1

        directory = (
            Path(folder)
            / f'periodic_workers_H{base.H:g}'
            / f'{counter:06d}'
        )

        def report(message):
            checkpoint(f'{message} · native attempt {counter}')

        report(f'Periodic {task["kind"]}')

        result = run_isolated(
            task, base, axes, options, directory, report, on_event
        )

        payload['attempts'].append(dict(
            task=counter,
            kind=task['kind'],
            uv=task.get('uv', task.get('seed', {}).get('uv')),
            mode=task.get('mode'),
            amplitude=task.get('amplitude'),
            period=task.get('period'),
            reason=result['reason'],
            exit_code=result['exit_code'],
            seconds=result['seconds'],
            trace_statistics=next((
                event['statistics']
                for event in reversed(result['events'])
                if 'statistics' in event
            ), {}),
            failures=[
                e['reason'] for e in result['events']
                if e['type'] == 'failure'
            ],
        ))

        return result

    def get_location(uv, round_index=0):
        if uv in locations:
            return locations[uv], evidence_cache[uv]

        evidence = equilibrium_evidence(
            parameters_at(base, axes, uv), cache.get(uv)
        )

        location = dict(
            uv=list(uv),
            **{
                k: float(a.physical(v))
                for k, a, v in zip('xy', axes, uv)
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

        if evidence['status'] == 'msp_absent_certified':
            location['status'] = 'excluded_no_msp'
            completed.add(uv)

        return location, evidence

    def accept(item, origin):
        row, seed = item['row'], item['seed']

        duplicate = matching_branch(
            row, payload['branches'], dimensions
        )

        if duplicate is not None:
            return duplicate

        branch = dict(
            id=len(payload['branches'])+1,
            origin=origin,
            points=[row],
            arms=[],
        )

        payload['branches'].append(branch)
        bank.append(dict(row=row, seed=seed, branch=branch['id']))
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
        """Trace every discovered branch in both directions."""

        while pending_traces:
            branch, seed = pending_traces[0]

            for axis in range(dimensions):
                for direction in (-1, 1):
                    arm = []

                    arm_info = dict(
                        axis=axis,
                        direction=direction,
                        records=arm,
                        termination='running',
                    )

                    branch['arms'].append(arm_info)

                    last_trace_save = 0.

                    def on_event(event):
                        nonlocal last_trace_save
                        if event['type'] != 'orbit':
                            return

                        row = event['row']

                        old = matching_branch(
                            row,
                            [
                                b for b in payload['branches']
                                if b['id'] != branch['id']
                            ],
                            dimensions,
                        )

                        if old is not None:
                            arm_info['joins_branch'] = old
                            return False

                        arm.append(row)
                        branch['points'].append(row)

                        bank.append(dict(
                            row=row,
                            seed=event['seed'],
                            branch=branch['id'],
                        ))

                        now = time.monotonic()

                        # Avoid rewriting the growing JSON and CSV
                        # files unnecessarily at every native point.
                        if now-last_trace_save >= 2.:
                            save()
                            last_trace_save = now

                    result = attempt(dict(
                        kind='trace',
                        hopf_points=[
                            row
                            for roots in hopf.values()
                            for row in roots
                            if row.get('classification_status')
                            == 'verified'
                        ],
                        seed=seed,
                        axis=axis,
                        direction=direction,
                    ), on_event)

                    termination = result['reason']

                    if termination == 'completed':
                        termination = next((
                            e['reason']
                            for e in reversed(result['events'])
                            if e['type'] == 'termination'
                        ), 'completed')

                    arm_info.update(
                        termination=termination,
                        notes=[
                            e for e in result['events']
                            if e['type'] in ('termination', 'failure')
                        ],
                    )

                    save()

            pending_traces.pop(0)
            save()

    def discover_at(uv, round_index):
        location, evidence = get_location(uv, round_index)

        if uv in completed:
            return location

        p = parameters_at(base, axes, uv)
        tasks = []

        # Reuse the closest verified orbit from each nearby branch.
        nearby = sorted(
            bank,
            key=lambda b: np.linalg.norm(
                np.asarray(b['seed']['uv'])-uv
            ),
        )

        used = set()

        for entry in nearby:
            if entry['branch'] in used:
                continue

            if np.linalg.norm(np.asarray(entry['seed']['uv'])-uv) > .25:
                continue

            used.add(entry['branch'])

            tasks.append(dict(
                kind='reuse',
                seed=entry['seed'],
                uv=list(uv),
            ))

        # Search every known admissible MSP root, regardless of stability.
        for root in evidence['msp']:
            tasks.extend(deterministic_guesses(root, p, options))

        if not tasks and not location['branches']:
            location['status'] = 'equilibrium_unresolved'
            completed.add(uv)
            save()
            return location

        if not location['branches']:
            location['status'] = 'orbit_discovery_unsuccessful'

        # Avoid identical guesses generated by clipped period estimates
        # or duplicate equilibrium records.
        seen = set()

        for task in tasks:
            task['uv'] = list(uv)
            signature = json.dumps(task, sort_keys=True)

            if signature in seen:
                continue
            seen.add(signature)

            process_events(attempt(task), location)

        completed.add(uv)
        save()

        return location

    try:
        save()

        # Stage 1: start periodic branches from known Hopf points.
        hopf_items = list(hopf.items())

        for index, (uv, roots) in enumerate(hopf_items, 1):
            checkpoint(
                f'Periodic Hopf seeds {index}/{len(hopf_items)}'
            )

            location, evidence = get_location(uv)

            if evidence['status'] == 'msp_absent_certified':
                continue

            for row in roots:
                task = dict(
                    kind='hopf',
                    equilibrium=row,
                    axis=int(row.get('solved_axis', 0)),
                    uv=list(uv),
                )

                process_events(attempt(task), location)

        # Stage 2: continue the emerging periodic branches.
        trace_pending()

        # Stage 3: search for additional, disconnected branches.
        rounds = options['periodic_refinement_rounds']

        for round_index in range(rounds+1):
            this_round = []

            for index, uv in enumerate(queue, 1):
                uv = _key(uv)

                if uv in completed:
                    continue

                checkpoint(
                    f'Periodic discovery {index}/{len(queue)} '
                    f'· round {round_index+1}/{rounds+1}'
                )

                location = discover_at(uv, round_index)
                this_round.append(location)

                trace_pending()

            if round_index == rounds:
                break

            # Refine around possible coexistence and verified cycles.
            step = 1/(samples-1)/2**(round_index+1)
            refinement = []

            for location in this_round:
                if location['status'] == 'excluded_no_msp':
                    continue

                for axis in range(dimensions):
                    for sign in (-1, 1):
                        uv = list(location['uv'])
                        uv[axis] += sign*step
                        key = _key(uv)

                        if 0 <= uv[axis] <= 1 and key not in completed:
                            refinement.append(key)

            queue = list(dict.fromkeys(refinement))
            scheduled.update(queue)

        unresolved = any(
            location['status'] != 'excluded_no_msp'
            and location['status'] != 'verified_periodic'
            for location in payload['locations']
        )

        failed = any(
            result['reason'] in (
                'worker_timeout',
                'native_worker_crashed',
                'solver_failed',
                'launch_failed',
            )
            for result in payload['attempts']
        )

        save()

        payload['status'] = (
            'completed_with_unresolved_searches'
            if unresolved or failed or payload['coverage']['limit_reached']
            else 'completed'
        )

    except BaseException:
        payload['status'] = 'interrupted'
        raise

    finally:
        save()
