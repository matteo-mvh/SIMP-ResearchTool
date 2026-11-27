"""Shared display metadata; defaults live in shared_eqn.Parameters."""
from dataclasses import asdict
from model.shared_eqn import Parameters

# Scientific defaults are read from Parameters, never duplicated here.
GROUPS = (
    ('Environment', (
        ('N_wc', 'Water-column nitrogen', 'mmol N/m²'),
        ('H', 'Water depth', 'm'), ('I_0', 'Surface irradiance', 'µmol photons/m²/s'),
        ('kw', 'Background attenuation', '1/m'))),
    ('Maximum growth', tuple((f'mu_{s}', n, '1/day') for s, n in
                            [('M', 'Macroalgae'), ('S', 'Seagrass'), ('P', 'Phytoplankton')])),
    ('Light half-saturation', tuple((f'K_I_{s}', n, 'µmol photons/m²/s') for s, n in
                                   [('M', 'Macroalgae'), ('S', 'Seagrass'), ('P', 'Phytoplankton')])),
    ('Nitrogen half-saturation', (('K_N_M', 'Macroalgae', 'mmol N/m³'),
                                 ('K_N_P', 'Phytoplankton', 'mmol N/m³'))),
    ('Biomass shading', tuple((f'k_{s}', n, 'm²/mmol N') for s, n in
                             [('M', 'Macroalgae'), ('S', 'Seagrass'), ('P', 'Phytoplankton')])),
    ('Background losses', tuple((f'm_{s}', n, '1/day') for s, n in
                               [('M', 'Macroalgae'), ('S', 'Seagrass'), ('P', 'Phytoplankton')])),
    ('Grazing and interference', (
        ('g_P', 'Phytoplankton grazing', '1/day'), ('g_M', 'Macroalgal grazing', '1/day'),
        ('K_G_S', 'Reference seagrass biomass', 'mmol N/m²'),
        ('gamma', 'Grazing saturation γ', '0 or 1'), ('beta', 'Grazing shape β', '1 or 2'),
        ('m_SM', 'Macroalgal interference', 'm²/mmol N/day'))),
)

CONTINUATION_PARAMETERS = tuple(k for k in asdict(Parameters()) if k not in ('H', 'gamma', 'beta'))

def parameter_schema():
    defaults = asdict(Parameters())
    return {name: dict(type='integer' if name in ('gamma','beta') else 'float',
                      default=defaults[name], min=0, category=group, unit=unit, description=description)
            for group, items in GROUPS for name, description, unit in items}
