"""Manifest-only discovery; scientific code is never imported in the GUI."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tomllib
import json
from .paths import ANALYSES

@dataclass(frozen=True)
class Analysis:
    id: str
    name: str
    group: str
    description: str
    path: Path
    script: Path
    parameters: dict
    form: str = ''


def discover(directory=None) -> dict[str, Analysis]:
    found = {}
    for manifest in sorted(Path(directory or ANALYSES).glob('*/analysis.toml')):
        try:
            with manifest.open('rb') as fp:
                spec = tomllib.load(fp)
            ident = str(spec['id'])
            if not ident.isidentifier():
                continue
            relative = Path(spec.get('entrypoint', 'run.py'))
            if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.py':
                continue
            script = (manifest.parent / relative).resolve()
            if not script.is_file() or not script.is_relative_to(manifest.parent.resolve()):
                continue
            if ident in found:
                continue
            pars = spec.get('parameters', {})
            form = str(spec.get('form', ''))
            if form == 'sensitivity':
                from model.sensitivity_config import job_schema
                pars = {**job_schema(), **pars}
            if form == 'timeseries':
                from model.timeseries_config import job_schema
                pars = {**job_schema(), **pars}
            if form in ('continuation_1d', 'continuation_2d'):
                from model.continuation_config import job_schema
                pars = {**job_schema(1 if form == 'continuation_1d' else 2), **pars}
            if not isinstance(pars, dict):
                continue
            # Optional shared parameter schema. Import data only: never import the run script.
            # File must live inside this analysis directory; manifest entries override it.
            config_file = spec.get('parameter_file')
            if config_file:
                rel_params = Path(config_file)
                if rel_params.is_absolute() or '..' in rel_params.parts or rel_params.suffix.lower() not in ('.json', '.toml'):
                    continue
                parameters_path = (manifest.parent / rel_params).resolve()
                if not parameters_path.is_file() or not parameters_path.is_relative_to(manifest.parent.resolve()):
                    continue
                if rel_params.suffix.lower() == '.json':
                    import json
                    loaded = json.loads(parameters_path.read_text(encoding='utf-8'))
                else:
                    with parameters_path.open('rb') as f:
                        loaded = tomllib.load(f)
                file_parameters = loaded.get('parameters', loaded)
                if not isinstance(file_parameters, dict):
                    continue
                pars = {**file_parameters, **pars}
            if any(not isinstance(v,dict) or not isinstance(k,str) or not k.isidentifier() for k,v in pars.items()):
                continue
            found[ident] = Analysis(
                ident, str(spec.get('name', ident)), str(spec.get('group', 'Other')),
                str(spec.get('description', '')), manifest.parent, script, pars, form
            )
        except (OSError, KeyError, ValueError, TypeError, tomllib.TOMLDecodeError, json.JSONDecodeError):
            continue
    return found


def cast_value(spec: dict, value):
    typ = spec.get('type', 'float')
    if typ == 'integer':
        if isinstance(value, bool): raise ValueError('Expected integer')
        v = int(value)
        if str(value).strip() not in (str(v), f'+{v}') and isinstance(value, str):
            raise ValueError('Expected an integer')
    elif typ == 'float':
        v = float(value)
        import math
        if not math.isfinite(v): raise ValueError('Value must be finite')
    elif typ == 'boolean':
        if isinstance(value, bool): v = value
        elif str(value).strip().lower() in ('true','1','yes'): v = True
        elif str(value).strip().lower() in ('false','0','no'): v = False
        else: raise ValueError('Expected true/false')
    elif typ in ('choice','seed','string'):
        v = str(value)
        if typ == 'choice' and v not in [str(x) for x in spec.get('options', [])]:
            raise ValueError('Invalid choice')
    else:
        raise ValueError(f'Unknown parameter type: {typ}')
    if isinstance(v, (int,float)) and not isinstance(v, bool):
        if 'min' in spec and v < spec['min']: raise ValueError('Below minimum')
        if 'max' in spec and v > spec['max']: raise ValueError('Above maximum')
    return v


def validate(analysis: Analysis, values: dict) -> dict:
    unknown = set(values) - set(analysis.parameters)
    if unknown: raise ValueError(f'Unknown parameters: {", ".join(sorted(unknown))}')
    out = {}
    for key, spec in analysis.parameters.items():
        if not isinstance(spec, dict): raise ValueError(f'Invalid manifest parameter: {key}')
        raw = values.get(key, spec.get('default', ''))
        try: out[key] = cast_value(spec, raw)
        except (ValueError, TypeError) as exc: raise ValueError(f'{key}: {exc}') from exc
    return out
