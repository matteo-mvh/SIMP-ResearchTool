"""Automatic numerical settings for MSP periodic-orbit analysis."""

import math


def periodic_schema():
    def integer(default, low, high):
        return dict(type='integer', default=default, min=low, max=high)

    def real(default, low, high):
        return dict(type='float', default=default, min=low, max=high)

    def choice(default, options):
        return dict(type='choice', default=default, options=options)

    def string(default):
        return dict(type='string', default=default)

    return {
        # Always enabled in ordinary continuation jobs.
        'periodic_analysis': choice('on', ['on', 'off']),

        # Initial discovery and local refinement.
        'periodic_samples': integer(9, 3, 301),
        'periodic_amplitudes': string('0.03,0.15,0.4,0.8'),
        'periodic_period_factors': string('0.5,1,2'),
        'periodic_real_modes': choice('fallback', ['fallback', 'always']),
        'periodic_refinement_rounds': integer(1, 0, 8),

        # Temporal discretization.
        'periodic_mode': choice('collocation', ['collocation', 'bspline']),
        'periodic_nodes': integer(30, 12, 600),
        'periodic_order': integer(3, 2, 5),
        'periodic_refinement_levels': integer(4, 1, 5),

        # Native arclength continuation.
        # Increased automatically for high-resolution axes.
        'periodic_max_steps': integer(1200, 1, 20000),
        'periodic_max_step': real(.015, 1e-6, .2),
        'periodic_min_step': real(1e-7, 1e-12, .01),

        # Newton correction.
        'periodic_global_newton': choice('on', ['on', 'off']),
        'periodic_newton_tolerance': real(1e-9, 1e-13, 1e-4),
        'periodic_newton_iterations': integer(25, 5, 100),

        # Numerical verification.
        'periodic_validation_tolerance': real(.005, 1e-6, .05),
        'periodic_defect_tolerance': real(.005, 1e-6, .05),
        'periodic_floquet_tolerance': real(1e-4, 1e-8, .02),
        'periodic_min_amplitude': real(1e-4, 1e-8, .1),
        'periodic_min_period': real(.1, 1e-5, 1e5),
        'periodic_max_period': real(1e5, 1, 1e8),

        # Native Hopf switching.
        'periodic_hopf_offset': real(.0004, 1e-8, .02),

        # Maximum time without new native solution records.
        # NOT a total runtime limit for the periodic analysis.
        'periodic_timeout': real(180, 5, 86400),
    }


def periodic_options(values):
    """Validate and normalize the internal solver configuration."""
    schema = periodic_schema()
    result = {key: values.get(key, spec['default']) for key, spec in schema.items()}
    for key, spec in schema.items():
        value = result[key]
        if spec['type'] == 'choice':
            if value not in spec['options']:
                raise ValueError(f'Invalid {key}: {value}')

        elif spec['type'] in ('float', 'integer'):
            number = float(value)
            if (not math.isfinite(number) or not spec['min'] <= number <= spec['max']):
                raise ValueError(f'{key} must be between ' f'{spec["min"]} and {spec["max"]}')

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
                numbers = tuple(float(v.strip()) for v in str(value).split(','))

            if (
                not numbers
                or len(numbers) > 12
                or any(not math.isfinite(n) or n <= 0 for n in numbers)
            ):
                raise ValueError(f'{key}: enter 1–12 positive numbers')

            result[key] = numbers

    if result['periodic_min_step'] >= result['periodic_max_step']:
        raise ValueError('Minimum step must be below maximum step')

    if result['periodic_min_period'] >= result['periodic_max_period']:
        raise ValueError('Minimum period must be below maximum period')

    return result


def standard_periodic_options():
    """Fixed numerical policy for all ordinary continuation jobs."""
    return periodic_options({'periodic_analysis': 'on'})


def node_count(options, level=0):
    """Return a node count compatible with the collocation order."""
    n = options['periodic_nodes'] * 2**level
    multiple = math.lcm(2, options['periodic_order'])
    return ((n + multiple - 1) // multiple) * multiple
