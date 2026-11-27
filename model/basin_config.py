"""Fixed-parameter basin-analysis settings."""
from model.parameter_metadata import parameter_schema


def job_schema():
    schema = parameter_schema()
    schema.update({
        'basin_sampling_mode': dict(type='choice', default='adaptive', options=['adaptive','uniform'], category='Basin sampling',
                                    description='Adaptive boundary pilot and independent volume-weighted audit, or original uniform Monte Carlo.'),
        'basin_boundary_budget': dict(type='integer', default=128, min=18, max=100000, category='Basin sampling',
                                      description='Maximum extra pilot integrations for adaptive octree refinement.'),
        'basin_boundary_depth': dict(type='integer', default=5, min=1, max=12, category='Basin sampling',
                                     description='Maximum octree refinement depth; no assumption of a homogeneous interior.'),
        'basin_backend': dict(type='choice', default='fast', options=['fast','reference'], category='Basin sampling',
                              description='Exact scalar/optional Numba RHS and analytical Jacobian, or original solver.'),
        'basin_compare_reference': dict(type='boolean', default=False, category='Basin sampling',
                                        description='Run identical audit samples through the original solver to measure speed and assignment agreement.'),
        'basin_samples': dict(type='integer', default=256, min=16, max=100000, category='Basin sampling',
                              description='Independent uniform initial conditions; increase for precision.'),
        'basin_S_max': dict(type='float', default=0., min=0., unit='mmol N/m²', category='Basin sampling',
                           description='Upper initial seagrass biomass; 0 uses 1.25 times the unshaded S equilibrium (at least 1).'),
        'basin_days': dict(type='float', default=20000., min=500., category='Basin sampling', unit='days',
                           description='Maximum integration per seed; stop early after confirmed convergence.'),
        'basin_seed': dict(type='integer', default=42, min=0, category='Basin sampling', description='Reproducible random seed.'),
        'basin_workers': dict(type='integer', default=4, min=1, max=32, category='Basin sampling', description='Independent trajectory worker processes.'),
        'basin_cycles': dict(type='boolean', default=True, category='Basin sampling', description='Search for and verify a stable MSP cycle before sampling.'),
    })
    return schema
