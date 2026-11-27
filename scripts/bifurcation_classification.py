
"""Independent classification of SIMP equilibrium bifurcations.

SN: generic saddle-node.
TC: generic boundary transcritical.
H: generic Hopf, including first Lyapunov coefficient.

Native continuation remains unchanged. Additional 1D events are
independently corrected using the augmented equilibrium equations.
"""

from dataclasses import replace
from pathlib import Path
from collections import Counter

import argparse
import csv
import json
import sys

import numpy as np
from scipy.optimize import least_squares

from model.shared_eqn import (
    per_capita, rhs, jacobian, resources,
)
from model.continuation_config import parse_job


SPECIES = 'MSP'

EIG_TOL = 1e-7
FREQ_TOL = 1e-6
COEFF_TOL = 1e-9

COLORS = {
    'saddle_node': '#222222',
    'transcritical_forward': '#2475B0',
    'transcritical_backward': '#D88A22',
    'hopf_supercritical': '#28865C',
    'hopf_subcritical': '#B52952',
    'unresolved': '#999999',
}

MARKERS = {
    'saddle_node': '^',
    'transcritical_forward': 'o',
    'transcritical_backward': 'v',
    'hopf_supercritical': 'D',
    'hopf_subcritical': 's',
    'unresolved': 'x',
}

LABELS = {
    'saddle_node': 'Saddle-node (SN)',
    'transcritical_forward': 'Forward TC',
    'transcritical_backward': 'Backward TC',
    'hopf_supercritical': 'Supercritical Hopf',
    'hopf_subcritical': 'Subcritical Hopf',
    'unresolved': 'Unresolved / degenerate',
}


def parameters_at(base, axes, uv):
    """Convert normalized coordinates to physical parameters."""

    return replace(base, **{
        axis.name: float(axis.physical(t))
        for axis, t in zip(axes, uv)
    })


def state_of(row):
    return np.array([float(row[c]) for c in SPECIES])


def coordinates_of(row, axes):
    return np.array([
        float(row[k]) for k in 'uv'[:len(axes)]
    ])


def resident_indices(community):
    return [i for i, c in enumerate(SPECIES) if c in community]


def solve_linear(A, b):
    """Reject severely ill-conditioned linear systems."""

    if np.linalg.cond(A) > 1e10:
        raise ValueError('Ill-conditioned critical linear system')

    return np.linalg.solve(A, b)


def parameter_derivative(fun, uv, axis, h):
    """Second-order derivative in a normalized parameter direction.

    Use centered differences inside the domain and one-sided
    differences near its boundaries.
    """

    uv = np.asarray(uv, float)
    e = np.zeros_like(uv)
    e[axis] = h

    if uv[axis] >= h and uv[axis] <= 1-h:
        return (fun(uv+e)-fun(uv-e))/(2*h)

    if uv[axis] < h:
        return (
            -3*fun(uv)+4*fun(uv+e)-fun(uv+2*e)
        )/(2*h)

    return (
        3*fun(uv)-4*fun(uv-e)+fun(uv-2*e)
    )/(2*h)


def state_tensors(j, h, third=False):
    """Obtain D²f and optionally D³f from the analytical Jacobian.

    j(z) is evaluated at displacements from the equilibrium.
    """

    zero = np.zeros(3)
    A = j(zero)
    directions = h*np.eye(3)

    B = np.empty((3, 3, 3))

    for k, e in enumerate(directions):
        B[:, :, k] = (j(e)-j(-e))/(2*h)

    # The Hessian is symmetric in its two input directions.
    B = (B+B.swapaxes(1, 2))/2

    if not third:
        return B, None

    C = np.empty((3, 3, 3, 3))

    for k, e in enumerate(directions):
        C[:, :, k, k] = (
            j(e)-2*A+j(-e)
        )/h**2

        for l in range(k+1, 3):
            d = directions[l]

            mixed = (
                j(e+d)-j(e-d)-j(-e+d)+j(-e-d)
            )/(4*h**2)

            C[:, :, k, l] = mixed
            C[:, :, l, k] = mixed

    # Symmetrize all three input directions.
    C = sum(
        C.transpose(0, *order)
        for order in (
            (1, 2, 3), (1, 3, 2),
            (2, 1, 3), (2, 3, 1),
            (3, 1, 2), (3, 2, 1),
        )
    )/6

    return B, C


def B_apply(B, a, b):
    return np.einsum('ijk,j,k->i', B, a, b)


def C_apply(C, a, b, c):
    return np.einsum('ijkl,j,k,l->i', C, a, b, c)


def critical_eigenvectors(A, target):
    """Compute normalized right and adjoint eigenvectors."""

    values, vectors = np.linalg.eig(A)

    index = int(np.argmin(abs(values-target)))
    q = vectors[:, index].astype(complex)

    left_values, left_vectors = np.linalg.eig(A.T)

    index = int(
        np.argmin(abs(left_values-np.conj(target)))
    )

    w = left_vectors[:, index].astype(complex)

    # Normalize the phase of the right eigenvector.
    k = int(np.argmax(abs(q)))
    q *= np.exp(-1j*np.angle(q[k]))
    q /= np.linalg.norm(q)

    inner = np.vdot(w, q)

    if abs(inner) < 1e-8:
        raise ValueError('Nearly singular critical eigenvectors')

    # Hermitian normalization: <w,q> = 1.
    w /= np.conj(inner)

    return q, w


def lyapunov_coefficient(A, B, C, q, w, omega):
    """First Lyapunov coefficient using Kuznetsov's formula."""

    qbar = np.conj(q)

    h11 = solve_linear(A, B_apply(B, q, qbar))

    h20 = solve_linear(
        2j*omega*np.eye(3)-A,
        B_apply(B, q, q),
    )

    G21 = (
        C_apply(C, q, q, qbar)
        - 2*B_apply(B, q, h11)
        + B_apply(B, qbar, h20)
    )

    return float(np.real(np.vdot(w, G21))/(2*omega))


def reliable_nonzero(coarse, fine):
    """Require a nonzero coefficient robust to derivative refinement."""

    if not np.isfinite(coarse) or not np.isfinite(fine):
        return False

    error = abs(coarse-fine)

    return (
        np.sign(coarse) == np.sign(fine)
        and abs(fine) > max(COEFF_TOL, 5*error)
    )


def classify_local(f, j, uv, kind, residents, invader=None,
                   state_step=1e-3):
    """Classify a local bifurcation of a smooth 3D vector field.

    f(z, uv) and j(z, uv) use scaled state displacements z.
    The candidate equilibrium is at z = 0.
    """

    uv = np.asarray(uv, float)
    zero = np.zeros(3)

    A = j(zero, uv)
    eig = np.linalg.eigvals(A)

    result = dict(
        classification='unresolved',
        classification_status='unresolved',
        classification_reason='',
        zero_eigenvalues=int(np.sum(abs(eig) <= EIG_TOL)),
        critical_eigenvalues=int(
            np.sum(abs(eig.real) <= EIG_TOL)
        ),
        unstable_directions=int(
            np.sum(eig.real > EIG_TOL)
        ),
    )

    def unresolved(reason):
        result['classification_reason'] = reason
        return result

    if not np.all(np.isfinite(A)):
        return unresolved('Nonfinite Jacobian')

    if np.max(abs(f(zero, uv))) > 1e-7:
        return unresolved('Candidate is not an equilibrium')

    center = np.flatnonzero(abs(eig.real) <= EIG_TOL)

    if kind in ('fold', 'invasion'):

        if len(center) != 1 or abs(eig[center[0]]) > EIG_TOL:
            return unresolved('Expected one simple zero eigenvalue')

        others = np.delete(eig, center[0])

        if np.any(abs(others.real) < 5*EIG_TOL):
            return unresolved('Additional nearly critical eigenvalue')

    elif kind == 'hopf':

        pair = [
            v for v in eig
            if abs(v.real) <= EIG_TOL
            and abs(v.imag) > FREQ_TOL
        ]

        if len(center) != 2 or len(pair) != 2:
            return unresolved('Expected one isolated imaginary pair')

        others = np.delete(eig, center)

        if np.any(abs(others.real) < 5*EIG_TOL):
            return unresolved('Additional nearly critical eigenvalue')

    else:
        return unresolved('Unknown candidate type')

    if kind == 'invasion':

        if invader is None or invader in residents:
            return unresolved('Invalid invading species')

        if abs(A[invader, invader]) > EIG_TOL:
            return unresolved('Invasion rate is not zero')

        # The critical direction has invading component equal to one.
        q = np.zeros(3)
        q[invader] = 1.

        if residents:
            R = np.ix_(residents, residents)
            q[residents] = -solve_linear(
                A[R], A[residents, invader]
            )

    elif kind == 'fold':

        q, w = critical_eigenvectors(A, 0.)
        q, w = q.real, w.real

    else:

        omega = float(max(v.imag for v in pair))
        q, w = critical_eigenvectors(A, 1j*omega)

        result['omega'] = omega
        result['hopf_period'] = 2*np.pi/omega

    def coefficients(h):

        B, C = state_tensors(
            lambda z: j(z, uv),
            h,
            third=(kind == 'hopf'),
        )

        parameter_coefficients = []

        for axis in range(len(uv)):

            hp = min(h, 1e-4)

            Fp = parameter_derivative(
                lambda t: f(zero, t), uv, axis, hp
            )

            Jp = parameter_derivative(
                lambda t: j(zero, t), uv, axis, hp
            )

            if kind == 'fold':

                # Parameter unfolding of the saddle-node.
                a = float(np.real(np.vdot(w, Fp)))

            elif kind == 'invasion':

                # Movement of the resident equilibrium with parameter.
                tangent = np.zeros(3)

                if residents:
                    tangent[residents] = -solve_linear(
                        A[np.ix_(residents, residents)],
                        Fp[residents],
                    )

                # Derivative of the invasion eigenvalue along
                # the resident equilibrium branch.
                a = float(
                    Jp[invader, invader]
                    + B[invader, invader, :] @ tangent
                )

            else:

                # The equilibrium moves as the parameter changes.
                tangent = -solve_linear(A, Fp)

                Jtotal = Jp + np.einsum(
                    'ijk,k->ij', B, tangent
                )

                a = float(np.real(
                    np.vdot(w, Jtotal @ q)
                ))

            parameter_coefficients.append(a)

        if kind == 'invasion':

            b = float(
                .5*B_apply(B, q, q)[invader]
            )

        elif kind == 'fold':

            b = float(np.real(
                .5*np.vdot(w, B_apply(B, q, q))
            ))

        else:

            b = lyapunov_coefficient(
                A, B, C, q, w, omega
            )

        return np.asarray(parameter_coefficients), b

    # Repeat numerical differentiation at a smaller step.
    a_coarse, b_coarse = coefficients(state_step)
    a_fine, b_fine = coefficients(state_step/2)

    # In 2D, use the parameter direction with the clearest unfolding.
    axis = int(np.argmax(abs(a_fine)))

    a = float(a_fine[axis])
    b = float(b_fine)

    result.update(
        classification_axis=axis,
        parameter_coefficient=a,
        parameter_coefficient_error=float(
            abs(a_coarse[axis]-a)
        ),
        nonlinear_coefficient=b,
        nonlinear_coefficient_error=float(
            abs(b_coarse-b)
        ),
        noncritical_unstable_directions=int(
            np.sum(others.real > EIG_TOL)
        ),
    )

    for k, value in zip('uv', a_fine):
        result[f'unfolding_{k}'] = float(value)

    if kind == 'hopf':
        result['first_lyapunov_coefficient'] = b

    elif kind == 'invasion':
        result['invasion_quadratic_coefficient'] = b

    else:
        result['fold_quadratic_coefficient'] = b

    if not reliable_nonzero(a_coarse[axis], a):
        return unresolved(
            'Vanishing or numerically uncertain parameter crossing'
        )

    if not reliable_nonzero(b_coarse, b):

        reason = (
            'Possible degenerate Hopf / generalized Hopf'
            if kind == 'hopf'
            else 'Vanishing or uncertain quadratic coefficient'
        )

        return unresolved(reason)

    if kind == 'fold':

        result['classification'] = 'saddle_node'

        # Equilibria exist where -a*dmu/b is positive.
        result['branch_side'] = (
            'increasing_parameter' if -a/b > 0
            else 'decreasing_parameter'
        )

    elif kind == 'invasion':

        result['classification'] = (
            'transcritical_forward' if b < 0
            else 'transcritical_backward'
        )

        result['branch_side'] = (
            'increasing_parameter' if -a/b > 0
            else 'decreasing_parameter'
        )

        # Stability of the new branch in the critical direction.
        result['emerging_branch_critical_stability'] = (
            'stable' if b < 0 else 'unstable'
        )

    else:

        result['classification'] = (
            'hopf_supercritical' if b < 0
            else 'hopf_subcritical'
        )

        result['branch_side'] = (
            'increasing_parameter' if -a/b > 0
            else 'decreasing_parameter'
        )

        result['emerging_orbit_stability'] = (
            'stable'
            if b < 0 and np.all(others.real < -EIG_TOL)
            else 'unstable'
        )

    result['classification_status'] = 'verified'
    result['classification_reason'] = (
        'Spectral and normal-form nondegeneracy checks passed'
    )

    return result


def classify_event(row, base, axes):
    """Apply local normal-form classification to a SIMP event."""

    output = dict(row)

    try:

        y = state_of(row)
        uv = coordinates_of(row, axes)
        p = parameters_at(base, axes, uv)

        residents = resident_indices(row['community'])
        absent = [i for i in range(3) if i not in residents]

        if np.any(y < -1e-8):
            raise ValueError('Negative equilibrium biomass')

        if any(y[i] <= 1e-8 for i in residents):
            raise ValueError('Resident biomass near presence boundary')

        if any(abs(y[i]) > 1e-8 for i in absent):
            raise ValueError('Community and biomass are inconsistent')

        if any(t < -1e-10 or t > 1+1e-10 for t in uv):
            raise ValueError('Parameter outside continuation domain')

        _, nitrogen = resources(y, p)

        if nitrogen < -1e-8:
            raise ValueError('Negative dissolved nitrogen')

        rates = per_capita(y, p)

        if residents and np.max(abs(rates[residents])) > 1e-7:
            raise ValueError('Resident stationarity check failed')

        # Scale biomass coordinates to avoid derivatives being
        # dominated by the largest biomass component.
        scale = np.maximum(abs(y), 1.)

        def f(z, t):
            state = y+scale*z
            params = parameters_at(base, axes, t)
            return rhs(0., state, params)/scale

        def j(z, t):
            state = y+scale*z
            params = parameters_at(base, axes, t)

            return (
                jacobian(state, params)
                * scale[None, :]/scale[:, None]
            )

        # Bound derivative perturbations away from Monod poles
        # and the extended grazing-function singularities.
        step = min(
            1e-3,
            .05*(nitrogen+min(p.K_N_M, p.K_N_P))
            /(scale[0]/p.H+scale[2]),
            .05*(y[1]+p.K_G_S)/scale[1],
        )

        if step < 1e-7:
            raise ValueError('Insufficient safe differentiation interval')

        invader = (
            SPECIES.index(row['invader'])
            if row['kind'] == 'invasion'
            else None
        )

        result = classify_local(
            f, j, uv, row['kind'],
            residents, invader, state_step=step,
        )

        output.update(result)

        if 'classification_axis' in result:
            output['classification_parameter'] = axes[
                result['classification_axis']
            ].name

        if invader is not None:

            new_community = ''.join(
                c for c in SPECIES
                if c in row['community'] or c == SPECIES[invader]
            )

            output['resident_community'] = row['community']
            output['emerging_community'] = new_community

        output['parameter_domain_boundary'] = bool(
            np.any(uv < 1e-8) or np.any(uv > 1-1e-8)
        )

        output['nitrogen_boundary'] = bool(nitrogen < 1e-8)

        output['state_scale'] = ','.join(map(str, scale))

    except (
        ValueError, ArithmeticError, KeyError,
        TypeError, np.linalg.LinAlgError,
    ) as exc:

        output.update(
            classification='unresolved',
            classification_status='unresolved',
            classification_reason=str(exc),
        )

    return output


# ==============================================================
# Independent audit of 1D equilibrium branches
# ==============================================================

def event_indicator(y, p, community, kind, invader=''):
    """Continuous scalar indicators for bifurcation candidates."""

    residents = resident_indices(community)

    if kind == 'invasion':
        return float(per_capita(y, p)[SPECIES.index(invader)])

    A = jacobian(y, p)[np.ix_(residents, residents)]

    if kind == 'fold':

        norms = np.maximum(
            np.linalg.norm(A, axis=1), 1e-15
        )

        return float(np.linalg.det(A/norms[:, None]))

    # Characteristic-polynomial Hopf condition.
    # This is only a candidate test; classify_local verifies the pair.
    A = A/max(np.max(abs(A)), .01)
    coefficients = np.poly(A)

    if len(residents) == 2:
        return float(coefficients[1])

    return float(
        coefficients[1]*coefficients[2]-coefficients[3]
    )


def correct_candidate(a, b, base, axis, kind, invader):
    """Correct a bracketed 1D event independently using SciPy."""

    face = a['community']
    residents = resident_indices(face)

    ua, ub = float(a['u']), float(b['u'])
    lo, hi = sorted((ua, ub))

    # A saddle-node can turn just outside the interval between
    # adjacent sampled parameter coordinates.
    if kind == 'fold':
        lo -= 1/axis.resolution
        hi += 1/axis.resolution

    lo, hi = max(0., lo-1e-9), min(1., hi+1e-9)

    if hi-lo < 1e-12:
        raise ValueError('Empty candidate bracket')

    ya, yb = state_of(a), state_of(b)

    def unpack(z):
        y = np.zeros(3)
        y[residents] = np.exp(z[:-1])
        u = float(z[-1])
        p = parameters_at(base, [axis], [u])

        return y, u, p

    def residual(z):

        y, u, p = unpack(z)
        _, N = resources(y, p)

        if N < -1e-8:
            return np.full(len(residents)+1, 1+abs(N))

        F = per_capita(y, p)[residents]

        indicator = event_indicator(
            y, p, face, kind, invader
        )

        return np.r_[F/.1, indicator]

    start = np.r_[
        .5*(
            np.log(np.maximum(ya[residents], 1e-8))
            + np.log(np.maximum(yb[residents], 1e-8))
        ),
        .5*(lo+hi),
    ]

    upper = max(
        30.,
        float(np.max(start[:-1]))+5 if residents else 30.,
    )

    solved = least_squares(
        residual,
        start,
        bounds=(
            np.r_[np.full(len(residents), -18.42), lo],
            np.r_[np.full(len(residents), upper), hi],
        ),
        x_scale='jac',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )

    if np.max(abs(residual(solved.x))) > 1e-7:
        raise ValueError('Independent event correction failed')

    y, u, p = unpack(solved.x)
    _, N = resources(y, p)

    return dict(
        H=p.H,
        u=u,
        x=float(axis.physical(u)),
        community=face,
        kind=kind,
        invader=invader,
        source='independent_scipy_correction',
        M=float(y[0]),
        S=float(y[1]),
        P=float(y[2]),
        N=float(N),
        P_area=float(y[2]*p.H),
        N_area=float(N*p.H),
    )


def duplicate_event(row, events):

    for old in events:

        if any(
            row[k] != old[k]
            for k in ('kind', 'community', 'invader')
        ):
            continue

        if abs(float(row['u'])-float(old['u'])) > 1e-5:
            continue

        distance = np.max(abs(
            np.log1p(state_of(row))
            - np.log1p(state_of(old))
        ))

        if distance < 1e-4:
            return True

    return False


def audit_1d(branches, events, base, axis):
    """Look for additional bifurcations along computed 1D branches."""

    groups = {}

    for row in branches:
        groups.setdefault(row['branch'], []).append(row)

    added = []
    failures = []

    for branch_id, path in groups.items():

        path.sort(key=lambda row: int(row['point']))

        if len(path) < 2:
            continue

        face = path[0]['community']

        kinds = [
            ('invasion', c) for c in SPECIES if c not in face
        ]

        if face != '0':
            kinds.append(('fold', ''))

        if len(resident_indices(face)) >= 2:
            kinds.append(('hopf', ''))

        for kind, invader in kinds:

            values = []

            for row in path:

                try:
                    p = parameters_at(
                        base, [axis], [float(row['u'])]
                    )

                    values.append(event_indicator(
                        state_of(row), p, face, kind, invader
                    ))

                except (
                    ValueError, ArithmeticError,
                    np.linalg.LinAlgError,
                ):
                    values.append(np.nan)

            for i, (va, vb) in enumerate(
                zip(values[:-1], values[1:])
            ):

                if not np.isfinite(va+vb):
                    continue

                if va*vb > 0 or abs(va)+abs(vb) < 1e-12:
                    continue

                a, b = path[i], path[i+1]

                try:

                    candidate = correct_candidate(
                        a, b, base, axis, kind, invader
                    )

                    if duplicate_event(candidate, events+added):
                        continue

                    result = classify_event(
                        candidate, base, [axis]
                    )

                    added.append(result)

                except (
                    ValueError, ArithmeticError,
                    np.linalg.LinAlgError,
                ) as exc:

                    failures.append(dict(
                        branch=branch_id,
                        community=face,
                        kind=kind,
                        invader=invader,
                        u_before=float(a['u']),
                        u_after=float(b['u']),
                        reason=str(exc),
                    ))

    return added, failures


# ==============================================================
# CSV output and classification figures
# ==============================================================

def read_csv(path):

    if not path.is_file():
        return []

    with path.open(
        newline='', encoding='utf-8-sig'
    ) as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):

    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    if not fields:
        fields = ['classification', 'classification_status']

    with path.open(
        'w', newline='', encoding='utf-8'
    ) as stream:

        writer = csv.DictWriter(
            stream, fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_classification(
    events, curves, branches, axes, depth, folder
):
    """Create separate figures without altering the original plots."""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)

    if len(axes) == 1:

        fig, panels = plt.subplots(
            3, 1, figsize=(9, 8), sharex=True
        )

        groups = {}

        for row in branches:
            groups.setdefault(row['branch'], []).append(row)

        for ax, species in zip(panels, SPECIES):

            for path in groups.values():

                path.sort(key=lambda row: int(row['point']))

                x = [float(r['x']) for r in path]
                y = [float(r[species]) for r in path]

                if species == 'P':
                    y = [value*depth for value in y]

                ax.plot(x, y, color='.65', lw=1.)

            for category, color in COLORS.items():

                rows = [
                    r for r in events
                    if r['classification'] == category
                ]

                if not rows:
                    continue

                y = [float(r[species]) for r in rows]

                if species == 'P':
                    y = [value*depth for value in y]

                ax.scatter(
                    [float(r['x']) for r in rows],
                    y,
                    color=color,
                    marker=MARKERS[category],
                    s=35,
                    zorder=5,
                    label=LABELS[category],
                )

            ax.set_ylabel(f'{species} [mmol N/m²]')
            ax.grid(alpha=.15)
            ax.set_xlim(axes[0].low, axes[0].high)

        panels[-1].set_xlabel(axes[0].name)

        handles, labels = panels[0].get_legend_handles_labels()

        if handles:
            fig.legend(handles, labels, loc='upper center',
                       ncol=3, frameon=False)

        fig.subplots_adjust(top=.86, hspace=.2)

    else:

        fig, ax = plt.subplots(figsize=(8, 5.5))
        groups = {}

        for row in curves:
            groups.setdefault(row['curve'], []).append(row)

        for path in groups.values():

            path.sort(key=lambda row: int(row['point']))

            ax.plot(
                [float(r['x']) for r in path],
                [float(r['y']) for r in path],
                color='.8',
                lw=1.,
                zorder=1,
            )

            # Color only segments whose endpoint
            # classifications agree.
            for a, b in zip(path[:-1], path[1:]):

                category = a['classification']

                if category != b['classification']:
                    continue

                ax.plot(
                    [float(a['x']), float(b['x'])],
                    [float(a['y']), float(b['y'])],
                    color=COLORS[category],
                    lw=1.7,
                    zorder=2,
                )

        for category, color in COLORS.items():

            rows = [
                r for r in events+curves
                if r['classification'] == category
            ]

            if not rows:
                continue

            ax.scatter(
                [float(r['x']) for r in rows],
                [float(r['y']) for r in rows],
                color=color,
                marker=MARKERS[category],
                s=10,
                label=LABELS[category],
                zorder=3,
            )

        ax.set_xlabel(axes[0].name)
        ax.set_ylabel(axes[1].name)

        ax.set_xlim(axes[0].low, axes[0].high)
        ax.set_ylim(axes[1].low, axes[1].high)

        ax.grid(alpha=.12)

        handles, labels = ax.get_legend_handles_labels()

        if handles:
            ax.legend(fontsize=8, frameon=False)

        ax.set_title(f'Bifurcation classification · H={depth:g} m')

    name = folder/f'bifurcation_classification_H{depth:g}'

    fig.savefig(Path(str(name)+'.png'), dpi=270,
                bbox_inches='tight')

    fig.savefig(Path(str(name)+'.pdf'),
                bbox_inches='tight')

    plt.close(fig)


def classify_output(folder, dimensions=None, report=print,
                    reuse_preflight=False):
    """Classify a continuation result directory."""

    folder = Path(folder)

    config = json.loads(
        (folder/'config.json').read_text(encoding='utf-8-sig')
    )

    if dimensions is None:

        analysis_id = config.get('analysis_id', '')

        if '2d' in analysis_id:
            dimensions = 2
        elif '1d' in analysis_id:
            dimensions = 1
        else:
            raise ValueError('Cannot determine continuation dimension')

    base, depths, axes, options = parse_job(
        config['parameters'], dimensions
    )

    summaries = {}

    for depth in depths:

        p = replace(base, H=depth)
        suffix = f'H{depth:g}'

        marker = folder/f'bifurcation_summary_{suffix}.json'

        if reuse_preflight and marker.is_file():
            summaries[suffix] = json.loads(
                marker.read_text(encoding='utf-8')
            )
            continue

        report(f'Classifying bifurcations · H={depth:g} m')

        raw_events = read_csv(
            folder/f'events_{suffix}.csv'
        )

        raw_curves = read_csv(
            folder/f'curves_{suffix}.csv'
        )

        branches = read_csv(
            folder/f'branches_{suffix}.csv'
        )

        events = [
            classify_event(
                dict(row, source='native_event'),
                p, axes,
            )
            for row in raw_events
        ]

        curves = []

        for i, row in enumerate(raw_curves):

            curves.append(classify_event(
                dict(row, source='native_curve'),
                p, axes,
            ))

            if i and i % 100 == 0:
                report(
                    f'Classifying curve points '
                    f'{i}/{len(raw_curves)}'
                )

        added, failures = [], []

        if dimensions == 1:

            added, failures = audit_1d(
                branches, events, p, axes[0]
            )

            events.extend(added)

            events.sort(
                key=lambda row: float(row['u'])
            )

        write_csv(
            folder/f'bifurcations_{suffix}.csv',
            events,
        )

        write_csv(
            folder/f'bifurcation_curves_{suffix}.csv',
            curves,
        )

        write_csv(
            folder/f'bifurcation_audit_failures_{suffix}.csv',
            failures,
        )

        summaries[suffix] = dict(
            native_event_candidates=len(raw_events),
            independent_candidates=len(added),
            classified_curve_points=len(curves),
            event_classifications=dict(Counter(
                row['classification'] for row in events
            )),
            curve_classifications=dict(Counter(
                row['classification'] for row in curves
            )),
            audit_correction_failures=len(failures),
            verified_events=sum(
                r['classification_status'] == 'verified'
                for r in events
            ),
        )

        plot_classification(
            events, curves, branches,
            axes, depth, folder/'figures',
        )

    (folder/'bifurcation_summary.json').write_text(
        json.dumps(summaries, indent=2),
        encoding='utf-8',
    )

    return summaries


# ==============================================================
# Verification against known analytical normal forms
# ==============================================================

def self_test():

    uv = np.array([.5])

    # Saddle-node: x' = mu + x².
    def fold_f(z, t):
        x, y, w = z
        return np.array([
            t[0]-.5+x*x, -y, -w
        ])

    def fold_j(z, t):
        return np.diag([2*z[0], -1., -1.])

    result = classify_local(
        fold_f, fold_j, uv, 'fold', [0]
    )

    assert result['classification'] == 'saddle_node', result

    # Forward transcritical with resident feedback.
    #
    # x' = -x + 2y
    # y' = y(mu - 3y + x)
    #
    # Effective quadratic coefficient = -3 + 2 = -1.
    def tc_f(z, t):
        x, y, w = z
        mu = t[0]-.5

        return np.array([
            -x+2*y,
            y*(mu-3*y+x),
            -w,
        ])

    def tc_j(z, t):
        x, y, w = z
        mu = t[0]-.5

        return np.array([
            [-1., 2., 0.],
            [y, mu-6*y+x, 0.],
            [0., 0., -1.],
        ])

    result = classify_local(
        tc_f, tc_j, uv, 'invasion', [0], invader=1
    )

    assert result['classification'] == (
        'transcritical_forward'
    ), result

    np.testing.assert_allclose(
        result['nonlinear_coefficient'], -1., atol=1e-7
    )

    # Backward transcritical: x' = x(mu + x).
    def backward_f(z, t):
        x, y, w = z

        return np.array([
            x*(t[0]-.5+x), -y, -w
        ])

    def backward_j(z, t):
        return np.diag([
            t[0]-.5+2*z[0], -1., -1.
        ])

    result = classify_local(
        backward_f, backward_j, uv,
        'invasion', [], invader=0,
    )

    assert result['classification'] == (
        'transcritical_backward'
    ), result

    # Degenerate boundary event: x' = x(mu - x²).
    def degenerate_f(z, t):
        x, y, w = z
        return np.array([
            x*(t[0]-.5-x*x), -y, -w
        ])

    def degenerate_j(z, t):
        return np.diag([
            t[0]-.5-3*z[0]**2, -1., -1.
        ])

    result = classify_local(
        degenerate_f, degenerate_j, uv,
        'invasion', [], invader=0,
    )

    assert result['classification'] == 'unresolved', result

    # Both supercritical and subcritical Hopf normal forms.
    for sign, expected in (
        (-1., 'hopf_supercritical'),
        (1., 'hopf_subcritical'),
    ):

        def hopf_f(z, t):
            x, y, w = z
            mu = t[0]-.5
            r2 = x*x+y*y

            return np.array([
                mu*x-y+sign*x*r2,
                x+mu*y+sign*y*r2,
                -w,
            ])

        def hopf_j(z, t):
            x, y, w = z
            mu = t[0]-.5

            return np.array([
                [
                    mu+sign*(3*x*x+y*y),
                    -1+2*sign*x*y,
                    0.,
                ],
                [
                    1+2*sign*x*y,
                    mu+sign*(x*x+3*y*y),
                    0.,
                ],
                [0., 0., -1.],
            ])

        result = classify_local(
            hopf_f, hopf_j, uv, 'hopf', [0, 1]
        )

        assert result['classification'] == expected, result

        np.testing.assert_allclose(
            result['omega'], 1., atol=1e-7
        )

    print('All analytical bifurcation classification tests passed.')


def classified_main(dimensions):
    """Run native continuation followed by independent classification."""

    import multiprocessing as mp

    from app.core.control import load_args, StopRequested
    from scripts.continuation_runner import run

    mp.freeze_support()

    values, ctx, job = load_args()

    try:

        # Preserve the existing scientific workflow.
        run(values, ctx, dimensions)

        classify_output(
            ctx.folder,
            dimensions,
            reuse_preflight=True,
            report=lambda message: ctx.checkpoint(
                100., message, force=True
            ),
        )

        ctx.checkpoint(
            100.,
            'Continuation and bifurcation classification completed',
            force=True,
        )

    except StopRequested:
        return 130

    return 0


def main():

    parser = argparse.ArgumentParser(
        description='Independent SIMP bifurcation classification'
    )

    parser.add_argument(
        'folder', nargs='?', type=Path
    )

    parser.add_argument(
        '--dimensions', type=int, choices=(1, 2)
    )

    parser.add_argument(
        '--self-test', action='store_true'
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.folder is None:
        parser.error('Provide a result folder or --self-test')

    summary = classify_output(
        args.folder, args.dimensions
    )

    print(json.dumps(summary, indent=2))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
