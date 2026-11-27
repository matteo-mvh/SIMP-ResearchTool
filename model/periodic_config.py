
"""Internal numerical configuration for MSP periodic orbit analysis."""

import math


# Computational budgets per depth.
PERIODIC_DEPTH_BUDGET = 300.0
PERIODIC_MAX_ATTEMPTS = 64


def periodic_schema():
    """Internal settings, deliberately hidden from the GUI."""

    def integer(default, low, high, description):
        return dict(type='integer', default=default,
                    min=low, max=high, description=description)

    def real(default, low, high, description):
        return dict(type='float', default=default,
                    min=low, max=high, description=description)

    def choice(default, options, description):
        return dict(type='choice', default=default,
                    options=options, description=description)

    def string(default, description):
        return dict(type='string', default=default,
                    description=description)

    return {
        # General configuration.
        'periodic_analysis': choice('on', ['on', 'off'], 'Periodic analysis'),
        'periodic_mode': choice('collocation', ['collocation', 'bspline'], 'Discretization'),

        # Periodic orbit discovery.
        'periodic_samples': integer(5, 3, 301, 'Discovery locations per axis'),
        'periodic_amplitudes': string('0.08,0.35', 'Initial amplitudes'),
        'periodic_period_factors': string('1,2', 'Period multipliers'),
        'periodic_real_modes': choice('fallback', ['fallback', 'always'], 'Real-mode guesses'),
        'periodic_max_locations': integer(36, 3, 5000, 'Maximum discovery locations'),
        'periodic_refinement_rounds': integer(1, 0, 4, 'Parameter refinement rounds'),

        # Temporal discretization.
        'periodic_nodes': integer(30, 12, 600, 'Initial temporal nodes'),
        'periodic_order': integer(3, 2, 5, 'Collocation order'),
        'periodic_refinement_levels': integer(3, 1, 5, 'Maximum mesh doublings'),

        # Native pseudo-arclength continuation.
        'periodic_max_steps': integer(75, 0, 10000, 'Steps per direction'),
        'periodic_max_step': real(.015, 1e-6, .2, 'Maximum normalized step'),
        'periodic_min_step': real(1e-7, 1e-12, .01, 'Minimum normalized step'),

        # Newton correction.
        'periodic_global_newton': choice('on', ['on', 'off'], 'Globalized Newton correction'),
        'periodic_newton_tolerance': real(1e-9, 1e-13, 1e-4, 'Newton tolerance'),
        'periodic_newton_iterations': integer(25, 5, 100, 'Maximum Newton iterations'),

        # Numerical verification.
        'periodic_validation_tolerance': real(.005, 1e-6, .05, 'Mesh convergence tolerance'),
        'periodic_defect_tolerance': real(.005, 1e-6, .05, 'Periodic balance tolerance'),
        'periodic_floquet_tolerance': real(1e-4, 1e-8, .02, 'Floquet tolerance'),
        'periodic_min_amplitude': real(1e-4, 1e-8, .1, 'Minimum relative amplitude'),
        'periodic_min_period': real(.1, 1e-5, 1e5, 'Minimum period in days'),
        'periodic_max_period': real(1e5, 1, 1e8, 'Maximum period in days'),

        # Hopf switching and native worker isolation.
        'periodic_hopf_offset': real(.0004, 1e-8, .02, 'Hopf branch-switch offset'),
        'periodic_timeout': real(45, 5, 86400, 'Native worker timeout'),
    }


def periodic_options(values):
    """Validate and normalize internal numerical settings."""

    schema = periodic_schema()

    result = {
        key: values.get(key, spec['default'])
        for key, spec in schema.items()
    }

    # Compatibility for direct low-level calls.
    # Ordinary continuation jobs always enable periodic analysis.
    if ('periodic_analysis' not in values
            and values.get('oscillation_checks') == 'off'):
        result['periodic_analysis'] = 'off'

    for key, spec in schema.items():
        value = result[key]

        if spec['type'] == 'choice':
            if value not in spec['options']:
                raise ValueError(f'Invalid {key}: {value}')

        elif spec['type'] in ('float', 'integer'):
            number = float(value)

            if (not math.isfinite(number)
                    or not spec['min'] <= number <= spec['max']):
                raise ValueError(
                    f'{key} must be between '
                    f'{spec["min"]} and {spec["max"]}'
                )

            if spec['type'] == 'integer':
                if number != int(number):
                    raise ValueError(f'{key} must be an integer')

                result[key] = int(number)

            else:
                result[key] = number

        else:
            if isinstance(value, (tuple, list)):
                numbers = tuple(float(v) for v in value)
            else:
                numbers = tuple(
                    float(v.strip()) for v in str(value).split(',')
                )

            if (not numbers
                    or len(numbers) > 12
                    or any(not math.isfinite(n) or n <= 0 for n in numbers)):
                raise ValueError(
                    f'{key}: enter 1–12 positive finite numbers'
                )

            result[key] = numbers

    if result['periodic_min_step'] >= result['periodic_max_step']:
        raise ValueError('Minimum step must be below maximum step')

    if result['periodic_min_period'] >= result['periodic_max_period']:
        raise ValueError('Minimum period must be below maximum period')

    return result


def standard_periodic_options():
    """Automatic numerical policy for all normal continuation jobs."""

    return periodic_options({
        'periodic_analysis': 'on',
    })


def node_count(options, level=0):
    """Return a node count compatible with the collocation order."""

    n = options['periodic_nodes'] * 2**level
    multiple = math.lcm(2, options['periodic_order'])

    return ((n + multiple - 1) // multiple) * multiple
