"""Simple detection of stable periodic attractors for 1D continuation results.

Purpose
-------
This module answers the practical thesis question:

    Does a stable periodic attractor exist at this parameter value,
    and can it coexist with a stable equilibrium?

It does NOT try to continue unstable periodic orbits or trace
periodic folds exactly. It is therefore much cheaper than the
current collocation-based periodic continuation.

Main outputs
------------
For each 1D branch point, this module can classify:
- no stable periodic attractor found
- stable periodic attractor found
- coexistence: stable equilibrium + stable periodic attractor

It also estimates:
- period
- relative amplitude per state variable
- minima / maxima over one cycle

Assumptions
-----------
- This is intended for the 1D continuation workflow.
- Branch points come from the existing continuation result structure.
- The SIMP model has three state variables in the order M, S, P.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import json
import math

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.shared_eqn import rhs, jacobian


STATE_NAMES = ("M", "S", "P")


# ============================================================
# Basic helpers
# ============================================================

def state_from_row(row):
    return np.array([float(row[k]) for k in STATE_NAMES], dtype=float)


def parameter_at_row(base, axis, row):
    """Construct parameter set for a given 1D branch/event row."""
    return replace(base, **{axis.name: float(row["x"])})


def dominant_complex_mode(J):
    """Return dominant complex eigenvalue and eigenvector if present."""
    values, vectors = np.linalg.eig(J)

    complex_indices = [
        i for i, value in enumerate(values)
        if abs(value.imag) > 1e-10
    ]

    if not complex_indices:
        return None, None

    index = max(complex_indices, key=lambda i: values[i].real)
    return values[index], vectors[:, index]


def equilibrium_stability(state, p):
    """Classify local stability of the equilibrium using Jacobian eigenvalues."""
    J = jacobian(state, p)
    eig = np.linalg.eigvals(J)

    max_real = float(np.max(eig.real))
    dominant_complex_real = None

    complex_real_parts = [
        value.real for value in eig
        if abs(value.imag) > 1e-10
    ]
    if complex_real_parts:
        dominant_complex_real = float(max(complex_real_parts))

    return {
        "max_real_part": max_real,
        "dominant_complex_real_part": dominant_complex_real,
        "stable_equilibrium": bool(max_real < 0.0),
        "oscillatory_instability": bool(
            dominant_complex_real is not None
            and dominant_complex_real > 0.0
        ),
    }


def relative_amplitude(window):
    """Return relative peak-to-trough amplitude for each variable."""
    minima = np.min(window, axis=1)
    maxima = np.max(window, axis=1)
    mean = np.mean(window, axis=1)
    return (maxima - minima) / np.maximum(np.abs(mean), 1e-12)


def find_local_maxima(series):
    """Simple local maxima finder; avoids extra dependency."""
    if len(series) < 3:
        return np.array([], dtype=int)

    peaks = []
    for i in range(1, len(series) - 1):
        if series[i - 1] < series[i] and series[i] >= series[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def choose_observable(window):
    """Choose the variable with the strongest relative oscillation."""
    amps = relative_amplitude(window)
    return int(np.argmax(amps))


# ============================================================
# Stable-cycle detection from forward simulation
# ============================================================

def detect_stable_cycle(
    t,
    y,
    transient_fraction=0.5,
    min_relative_amplitude=0.02,
    min_peaks=6,
    period_cv_tol=0.05,
    amplitude_cv_tol=0.08,
):
    """
    Detect whether the trajectory converged to a stable periodic orbit.

    Strategy:
    - discard the transient half
    - choose the most oscillatory variable
    - detect successive maxima
    - check whether peak heights and peak spacing stabilize
    """
    n = len(t)
    if n < 200:
        return None

    cut = int(transient_fraction * n)
    t2 = t[cut:]
    y2 = y[:, cut:]

    observable = choose_observable(y2)
    series = y2[observable]

    peaks = find_local_maxima(series)
    if len(peaks) < min_peaks:
        return None

    # use only the last peaks
    peaks = peaks[-min_peaks:]
    peak_times = t2[peaks]
    peak_values = series[peaks]

    periods = np.diff(peak_times)
    if len(periods) < min_peaks - 1:
        return None

    mean_period = float(np.mean(periods))
    if mean_period <= 0:
        return None

    period_cv = float(np.std(periods) / mean_period)

    # approximate troughs in between peaks
    trough_values = []
    for i in range(len(peaks) - 1):
        a = peaks[i]
        b = peaks[i + 1]
        trough_values.append(np.min(series[a:b + 1]))
    trough_values = np.asarray(trough_values, dtype=float)

    if len(trough_values) == 0:
        return None

    amplitudes = peak_values[1:] - trough_values
    mean_amp = float(np.mean(amplitudes))
    amp_cv = float(np.std(amplitudes) / max(abs(mean_amp), 1e-12))

    # full-window relative amplitudes
    rel_amp = relative_amplitude(y2)
    max_rel_amp = float(np.max(rel_amp))

    if max_rel_amp < min_relative_amplitude:
        return None

    if period_cv > period_cv_tol:
        return None

    if amp_cv > amplitude_cv_tol:
        return None

    cycle_window_start = peaks[-2]
    cycle_window_end = peaks[-1]
    cycle_slice = y2[:, cycle_window_start:cycle_window_end + 1]

    minima = np.min(cycle_slice, axis=1)
    maxima = np.max(cycle_slice, axis=1)
    means = np.mean(cycle_slice, axis=1)

    return {
        "period": mean_period,
        "observable_index": observable,
        "observable": STATE_NAMES[observable],
        "period_cv": period_cv,
        "amplitude_cv": amp_cv,
        "relative_amplitude_M": float(rel_amp[0]),
        "relative_amplitude_S": float(rel_amp[1]),
        "relative_amplitude_P": float(rel_amp[2]),
        "M_min": float(minima[0]),
        "M_max": float(maxima[0]),
        "S_min": float(minima[1]),
        "S_max": float(maxima[1]),
        "P_min": float(minima[2]),
        "P_max": float(maxima[2]),
        "M_mean": float(means[0]),
        "S_mean": float(means[1]),
        "P_mean": float(means[2]),
        # a representative point on the cycle to warm-start neighbors
        "seed_state": cycle_slice[:, 0].tolist(),
    }


def integrate_trajectory(
    p,
    y0,
    duration,
    max_step,
    rtol=1e-7,
    atol=1e-9,
):
    """Integrate the SIMP ODE."""
    solution = solve_ivp(
        fun=lambda t, y: rhs(t, y, p),
        t_span=(0.0, duration),
        y0=np.asarray(y0, dtype=float),
        method="LSODA",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )

    if not solution.success:
        return None

    if solution.y.shape[1] < 50:
        return None

    if not np.all(np.isfinite(solution.y)):
        return None

    if np.min(solution.y) < -1e-8:
        return None

    return solution.t, solution.y


def perturbations_from_equilibrium(eq_state, p, size=0.05):
    """
    Generate simple perturbation seeds.

    First try the dominant complex eigenvector if available,
    otherwise use component-wise perturbations.
    """
    seeds = []

    J = jacobian(eq_state, p)
    value, vector = dominant_complex_mode(J)

    if vector is not None:
        direction = np.real(vector)
        scale = np.maximum(np.abs(eq_state), 1.0)
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        seed = eq_state + size * scale * direction
        seed = np.maximum(seed, 1e-10)
        seeds.append(seed)

        seed2 = eq_state - size * scale * direction
        seed2 = np.maximum(seed2, 1e-10)
        seeds.append(seed2)

    scale = np.maximum(np.abs(eq_state), 1.0)

    for i in range(3):
        plus = eq_state.copy()
        plus[i] += size * scale[i]
        seeds.append(np.maximum(plus, 1e-10))

        minus = eq_state.copy()
        minus[i] = max(1e-10, minus[i] - size * scale[i])
        seeds.append(minus)

    return seeds


def find_stable_cycle_at_point(
    eq_state,
    p,
    warm_seed=None,
    duration=4000.0,
    max_step=5.0,
    perturbation_size=0.05,
):
    """
    Try to detect a stable periodic attractor at one parameter value.
    """
    tried = []

    if warm_seed is not None:
        tried.append(np.asarray(warm_seed, dtype=float))

    tried.extend(
        perturbations_from_equilibrium(
            eq_state,
            p,
            size=perturbation_size,
        )
    )

    for y0 in tried:
        result = integrate_trajectory(
            p=p,
            y0=y0,
            duration=duration,
            max_step=max_step,
        )

        if result is None:
            continue

        t, y = result

        # If the trajectory simply converges back to equilibrium, skip.
        terminal = y[:, -1]
        if np.max(np.abs(terminal - eq_state) / np.maximum(np.abs(eq_state), 1.0)) < 0.01:
            continue

        cycle = detect_stable_cycle(t, y)
        if cycle is not None:
            cycle["initial_condition"] = y0.tolist()
            return cycle

    return None


# ============================================================
# Main branch scan
# ============================================================

def scan_stable_periodic_attractors(
    branch_rows,
    event_rows,
    base,
    axis,
    output_folder,
    integration_duration=4000.0,
    integration_max_step=5.0,
):
    """
    Scan a 1D continuation branch for stable periodic attractors.

    The practical interpretation is:
    - stable equilibrium only
    - stable periodic attractor only
    - coexistence (stable equilibrium + stable periodic attractor)
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # focus on the interior three-species branch by default
    rows = [
        dict(row)
        for row in branch_rows
        if row.get("community") == "MSP"
    ]

    rows.sort(key=lambda row: float(row["x"]))

    # keep verified Hopf points for context only
    hopf_points = [
        row for row in event_rows
        if str(row.get("classification", "")).startswith("hopf_")
        and row.get("classification_status") == "verified"
    ]

    results = []
    warm_seed = None

    for i, row in enumerate(rows):
        eq_state = state_from_row(row)
        p = parameter_at_row(base, axis, row)

        eq_info = equilibrium_stability(eq_state, p)

        # Only attempt cycle detection when it is plausible:
        # - equilibrium oscillatory unstable, or
        # - warm-start from previously found cycle, or
        # - close to a Hopf point, or
        # - equilibrium stable but we want to test for coexistence
        x = float(row["x"])
        near_hopf = any(
            abs(x - float(h["x"])) <= 10.0 / max(axis.resolution, 1)
            for h in hopf_points
        )

        should_try = (
            eq_info["oscillatory_instability"]
            or warm_seed is not None
            or near_hopf
            or eq_info["stable_equilibrium"]
        )

        cycle = None
        if should_try:
            cycle = find_stable_cycle_at_point(
                eq_state=eq_state,
                p=p,
                warm_seed=warm_seed,
                duration=integration_duration,
                max_step=integration_max_step,
            )

        stable_cycle = cycle is not None
        stable_eq = eq_info["stable_equilibrium"]
        coexistence = stable_eq and stable_cycle

        result = dict(row)
        result.update(eq_info)
        result.update({
            "stable_cycle_found": stable_cycle,
            "coexistence_eq_and_cycle": coexistence,
            "attractor_class": (
                "coexistence"
                if coexistence else
                "stable_cycle_only"
                if stable_cycle else
                "stable_equilibrium_only"
                if stable_eq else
                "no_stable_attractor_classified"
            ),
        })

        if cycle is not None:
            result.update({
                key: value
                for key, value in cycle.items()
                if key != "seed_state"
            })
            warm_seed = cycle["seed_state"]
        else:
            warm_seed = None

        results.append(result)

    write_csv(
        output_folder / f"stable_orbit_scan_H{base.H:g}.csv",
        results,
    )

    write_summary(
        output_folder / f"stable_orbit_summary_H{base.H:g}.json",
        results,
    )

    plot_stable_orbit_scan(
        output_folder / f"stable_orbit_scan_H{base.H:g}",
        results,
        axis_name=axis.name,
    )

    return results


# ============================================================
# Output
# ============================================================

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    summary = {
        "points": len(rows),
        "stable_cycle_found": sum(bool(row.get("stable_cycle_found")) for row in rows),
        "coexistence": sum(bool(row.get("coexistence_eq_and_cycle")) for row in rows),
        "stable_equilibrium_only": sum(
            row.get("attractor_class") == "stable_equilibrium_only"
            for row in rows
        ),
        "stable_cycle_only": sum(
            row.get("attractor_class") == "stable_cycle_only"
            for row in rows
        ),
        "x_range_with_cycle": None,
        "x_range_with_coexistence": None,
    }

    xs_cycle = [float(row["x"]) for row in rows if row.get("stable_cycle_found")]
    xs_co = [float(row["x"]) for row in rows if row.get("coexistence_eq_and_cycle")]

    if xs_cycle:
        summary["x_range_with_cycle"] = [min(xs_cycle), max(xs_cycle)]

    if xs_co:
        summary["x_range_with_coexistence"] = [min(xs_co), max(xs_co)]

    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def contiguous_regions(x, mask):
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    regions = []
    start = None

    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            regions.append((x[start], x[i - 1]))
            start = None

    if start is not None:
        regions.append((x[start], x[-1]))

    return regions


def plot_stable_orbit_scan(path_root, rows, axis_name):
    if not rows:
        return

    x = np.array([float(row["x"]) for row in rows], dtype=float)

    stable_eq = np.array([
        bool(row.get("stable_equilibrium"))
        for row in rows
    ])

    stable_cycle = np.array([
        bool(row.get("stable_cycle_found"))
        for row in rows
    ])

    coexistence = np.array([
        bool(row.get("coexistence_eq_and_cycle"))
        for row in rows
    ])

    fig, ax = plt.subplots(figsize=(9, 3.8))

    ax.plot(x, stable_eq.astype(float), lw=1.8, label="Stable equilibrium present")
    ax.plot(x, stable_cycle.astype(float), lw=1.8, label="Stable periodic attractor found")

    for a, b in contiguous_regions(x, stable_cycle):
        ax.axvspan(a, b, alpha=0.12)

    for a, b in contiguous_regions(x, coexistence):
        ax.axvspan(a, b, alpha=0.22)

    ax.set_xlabel(axis_name)
    ax.set_ylabel("Presence (0/1)")
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 1])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.savefig(Path(str(path_root) + ".png"), dpi=250, bbox_inches="tight")
    fig.savefig(Path(str(path_root) + ".pdf"), bbox_inches="tight")
    plt.close(fig)