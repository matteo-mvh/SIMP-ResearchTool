from dataclasses import asdict
import unittest

from app.parameter_inspection import GROUPS, inspect_parameters, seed_parameters
from model.shared_eqn import Parameters, per_capita


class InspectionTests(unittest.TestCase):
    def test_seed_import_defaults_aliases_and_canonical_precedence(self):
        import json
        source = {'kw': .7, 'depth': 6, 'nwc': 80, 'H': 3, 'sample_count': 100}
        record = {'values_json': json.dumps(source)}
        values, applied, ignored = seed_parameters(record)
        self.assertEqual(values['H'], 3)
        self.assertEqual(values['N_wc'], 80)
        self.assertEqual(values['kw'], .7)
        self.assertEqual(values['mu_M'], Parameters().mu_M)
        self.assertEqual(applied, {'kw', 'H', 'N_wc'})
        self.assertEqual(ignored, ['sample_count'])
        self.assertEqual(json.loads(record['values_json']), source)
        with self.assertRaises(ValueError):
            seed_parameters({'values_json': '{"sample_count": 100}'})

    def test_every_parameter_is_editable_exactly_once(self):
        names = [name for _, group in GROUPS for name, _, _ in group]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(asdict(Parameters())))

    def test_invasion_rates_only_for_absent_species(self):
        p = Parameters()
        rows = inspect_parameters({k: str(v) for k, v in asdict(p).items()})
        self.assertTrue(any(e.stability.classification == 'stable' for e, _ in rows))
        self.assertTrue(any(e.stability.classification == 'unstable' for e, _ in rows))
        for eq, rates in rows:
            expected = per_capita(eq.biomass, p)
            for i, biomass in enumerate(eq.biomass):
                if biomass > 0:
                    self.assertIsNone(rates[i])
                else:
                    self.assertAlmostEqual(rates[i], expected[i])

    def test_changed_parameters_and_invalid_edits(self):
        values = asdict(Parameters())
        values['I_0'] = '0'
        rows = inspect_parameters(values)
        self.assertEqual([e.community for e, _ in rows], ['0'])
        self.assertEqual(rows[0][0].stability.classification, 'stable')
        for key, value in [('H', ''), ('H', '0'), ('kw', 'nan'), ('gamma', '0')]:
            changed = asdict(Parameters())
            changed[key] = value
            with self.assertRaises(ValueError):
                inspect_parameters(changed)
