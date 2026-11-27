"""Configuration for MSP periodic boundary-value problems (no time stepping)."""
import math


def periodic_schema():
    def integer(default, low, high, description):
        return dict(type='integer',default=default,min=low,max=high,description=description)
    def real(default, low, high, description):
        return dict(type='float',default=default,min=low,max=high,description=description)
    return {
        'periodic_analysis': dict(type='choice',default='on',options=['on','off'],description='MSP periodic orbit analysis'),
        'periodic_mode': dict(type='choice',default='collocation',options=['collocation','bspline'],description='Periodic discretization'),
        'periodic_samples': integer(9,3,301,'Coarse discovery locations per axis'),
        'periodic_amplitudes': dict(type='string',default='0.03,0.15,0.4,0.8',description='Relative initial amplitudes'),
        'periodic_global_newton': dict(type='choice',default='on',options=['on','off'],description='Globalized Newton correction'),
        'periodic_real_modes': dict(type='choice',default='fallback',options=['fallback','always'],description='Real-mode planes when complex modes exist'),
        'periodic_period_factors': dict(type='string',default='0.5,1,2',description='Characteristic period multipliers'),
        'periodic_nodes': integer(30,12,600,'Coarse periodic nodes (rounded for collocation)'),
        'periodic_order': integer(3,2,5,'Collocation / B-spline order'),
        'periodic_refinement_levels': integer(4,1,5,'Maximum temporal mesh doublings'),
        'periodic_refinement_rounds': integer(1,0,4,'Targeted parameter refinement rounds'),
        'periodic_max_steps': integer(100,0,10000,'Arclength steps per direction'),
        'periodic_max_locations': integer(150,3,5000,'Maximum discovery locations (excess reported)'),
        'periodic_max_step': real(.015,1e-6,.2,'Maximum normalized arclength step'),
        'periodic_min_step': real(1e-7,1e-12,.01,'Minimum arclength step'),
        'periodic_newton_tolerance': real(1e-9,1e-13,1e-4,'Newton residual tolerance'),
        'periodic_newton_iterations': integer(25,5,100,'Maximum Newton iterations'),
        'periodic_validation_tolerance': real(.005,1e-6,.05,'Relative mesh convergence tolerance'),
        'periodic_defect_tolerance': real(.005,1e-6,.05,'Relative periodic balance defect tolerance'),
        'periodic_floquet_tolerance': real(1e-4,1e-8,.02,'Floquet unit-circle tolerance'),
        'periodic_min_amplitude': real(1e-4,1e-8,.1,'Minimum relative orbit amplitude'),
        'periodic_min_period': real(.1,1e-5,1e5,'Minimum admissible period (days)'),
        'periodic_max_period': real(1e5,1,1e8,'Maximum admissible period (days)'),
        'periodic_hopf_offset': real(.0004,1e-8,.02,'Normalized Hopf branch-switch offset magnitude'),
        'periodic_timeout': real(180,5,86400,'Native worker wall-time limit (seconds)'),
    }


def periodic_options(values):
    schema=periodic_schema()
    result={k:values.get(k,s['default']) for k,s in schema.items()}
    # Honor old saved jobs that explicitly disabled oscillation work.
    if 'periodic_analysis' not in values and values.get('oscillation_checks')=='off':
        result['periodic_analysis']='off'
    for key,spec in schema.items():
        value=result[key]
        if spec['type']=='choice':
            if value not in spec['options']:raise ValueError(f'Invalid {key}')
        elif spec['type'] in ('float','integer'):
            number=float(value)
            if not math.isfinite(number) or not spec['min']<=number<=spec['max']:
                raise ValueError(f'{key} must be between {spec["min"]} and {spec["max"]}')
            if spec['type']=='integer' and number!=int(number):raise ValueError(f'{key} must be an integer')
            result[key]=int(number) if spec['type']=='integer' else number
        else:
            numbers=tuple(float(v) for v in value) if isinstance(value,(tuple,list)) else tuple(float(v.strip()) for v in str(value).split(','))
            if not numbers or len(numbers)>12 or any(not math.isfinite(n) or n<=0 for n in numbers):
                raise ValueError(f'{key}: enter 1–12 positive finite numbers')
            result[key]=numbers
    if result['periodic_min_step']>=result['periodic_max_step']:raise ValueError('Periodic minimum step must be below maximum')
    if result['periodic_min_period']>=result['periodic_max_period']:raise ValueError('Periodic minimum period must be below maximum')
    return result


def node_count(options, level=0):
    n=options['periodic_nodes']*2**level
    multiple=math.lcm(2,options['periodic_order'])
    return ((n+multiple-1)//multiple)*multiple
