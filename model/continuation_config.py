
"""Job schema and validation shared by GUI and continuation workers."""

from dataclasses import asdict, dataclass, replace
import math

from model.shared_eqn import Parameters
from model.parameter_metadata import (
    CONTINUATION_PARAMETERS,
    parameter_schema,
)
from model.periodic_config import standard_periodic_options


def job_schema(dimensions):
    schema = parameter_schema()

    schema.update({
        'depth_count': dict(
            type='choice', default='1',
            options=['1', '2', '4'],
            description='Depth panels',
        ),
        'depths': dict(
            type='string', default='4',
            description='Depths in metres, separated by commas',
        ),
        'x_parameter': dict(
            type='choice', default='N_wc',
            options=list(CONTINUATION_PARAMETERS),
            description='Continuation parameter',
        ),
        'x_min': dict(
            type='float', default=1., min=0,
            description='Range minimum',
        ),
        'x_max': dict(
            type='float', default=200., min=0,
            description='Range maximum',
        ),
        'x_resolution': dict(
            type='integer', default=100, min=8, max=2000,
            description='Resolution and maximum continuation step',
        ),
        'discovery_points': dict(
            type='integer', default=17, min=3, max=200,
            description='Equilibrium discovery samples per transect',
        ),
        'max_steps': dict(
            type='integer', default=1200, min=20, max=20000,
            description='Maximum native steps per direction',
        ),
    })

    # Periodic settings are deliberately absent from the GUI schema.
    # They are automatically added in parse_job().

    if dimensions == 2:
        schema.update({
            'y_parameter': dict(
                type='choice', default='kw',
                options=list(CONTINUATION_PARAMETERS),
                description='Second continuation parameter',
            ),
            'y_min': dict(
                type='float', default=.1, min=0,
                description='Second range minimum',
            ),
            'y_max': dict(
                type='float', default=1., min=0,
                description='Second range maximum',
            ),
            'y_resolution': dict(
                type='integer', default=100, min=8, max=2000,
                description='Second axis resolution',
            ),
            'depth_workers': dict(
                type='choice', default='2',
                options=['1', '2', '4'],
                description='Independent depth worker processes',
            ),
            'reuse_native': dict(
                type='choice', default='off',
                options=['off', 'on'],
                description='Experimental compiled native problem reuse',
            ),
            'adaptive_step_factor': dict(
                type='choice', default='2',
                options=['1', '2', '3'],
                description='Maximum adaptive step multiplier',
            ),
            'transects': dict(
                type='integer', default=5, min=2, max=15,
                description='Seed-discovery transects per direction',
            ),
            'discovery_refinement': dict(
                type='integer', default=5, min=0, max=8,
                description='Adaptive discovery bisection rounds',
            ),
            'region_samples': dict(
                type='integer', default=7, min=3, max=50,
                description='Independent equilibrium checks per region',
            ),
            'refinement_rounds': dict(
                type='integer', default=3, min=0, max=8,
                description='Conflict-driven refinement rounds',
            ),
            'refinement_searches': dict(
                type='integer', default=12, min=1, max=100,
                description='Maximum searches per refinement round',
            ),
        })

    return schema


@dataclass(frozen=True)
class Axis:
    name: str
    low: float
    high: float
    resolution: int

    def physical(self, value):
        return self.low + (self.high - self.low) * value

    def normalized(self, value):
        return (value - self.low) / (self.high - self.low)


def parse_job(values, dimensions):
    defaults = {
        key: spec['default']
        for key, spec in job_schema(dimensions).items()
    }

    # Ignore obsolete periodic options inherited from older jobs.
    defaults.update({
        key: value
        for key, value in values.items()
        if not key.startswith('periodic_')
        and key != 'oscillation_checks'
    })

    v = defaults

    # Periodic analysis is always enabled.
    v.update(standard_periodic_options())

    depths = tuple(
        float(x.strip())
        for x in str(v['depths']).split(',')
        if x.strip()
    )

    if (
        int(v['depth_count']) not in (1, 2, 4)
        or len(depths) != int(v['depth_count'])
    ):
        raise ValueError(
            'Enter exactly 1, 2, or 4 depths '
            'to match the selected layout'
        )

    if (
        any(not math.isfinite(x) or x <= 0 for x in depths)
        or len(set(depths)) != len(depths)
    ):
        raise ValueError(
            'Depths must be distinct, finite, positive numbers'
        )

    axes = []

    for prefix in ('x', 'y')[:dimensions]:
        name = v[prefix + '_parameter']

        if name not in CONTINUATION_PARAMETERS:
            raise ValueError(
                'Choose a continuous model parameter other than depth'
            )

        lo = float(v[prefix + '_min'])
        hi = float(v[prefix + '_max'])
        resolution = int(v[prefix + '_resolution'])

        if (
            not all(math.isfinite(x) for x in (lo, hi))
            or not 0 <= lo < hi
        ):
            raise ValueError(
                f'{prefix.upper()} range must satisfy '
                '0 <= minimum < maximum'
            )

        if not 8 <= resolution <= 2000:
            raise ValueError(
                'Axis resolution must be 8–2000 intervals'
            )

        axes.append(Axis(name, lo, hi, resolution))

    if len({a.name for a in axes}) != len(axes):
        raise ValueError(
            'Choose different continuation parameters for X and Y'
        )

    # Disabled model inputs are replaced by continuation midpoints.
    model = {
        key: float(v[key])
        for key in asdict(Parameters())
    }

    model.update({
        axis.name: axis.physical(.5)
        for axis in axes
    })

    model['H'] = depths[0]
    p = Parameters(**model)

    # Reject degenerate reduced-solver endpoints.
    for axis in axes:
        for endpoint in (axis.low, axis.high):
            trial = replace(
                p, **{axis.name: endpoint}
            )

            if min(
                trial.k_S,
                trial.m_S,
                trial.m_M,
                trial.m_P + trial.g_P,
            ) <= 0:
                raise ValueError(
                    f'{axis.name}: this range includes a '
                    'degenerate reduced-solver endpoint; '
                    'choose a positive minimum'
                )

    limits = (
        ('discovery_points', 3, 200),
        ('max_steps', 20, 20000),
        ('transects', 2, 15),
        ('discovery_refinement', 0, 8),
        ('region_samples', 3, 50),
        ('refinement_rounds', 0, 8),
        ('refinement_searches', 1, 100),
    )

    for key, lo, hi in limits:
        if key in v and not lo <= int(v[key]) <= hi:
            raise ValueError(
                f'{key} must be between {lo} and {hi}'
            )

    if dimensions == 2:
        choices = (
            ('depth_workers', ('1', '2', '4')),
            ('reuse_native', ('off', 'on')),
            ('adaptive_step_factor', ('1', '2', '3')),
        )

        for key, allowed in choices:
            if str(v[key]) not in allowed:
                raise ValueError(f'Invalid {key}: {v[key]}')

    return p, depths, axes, v
