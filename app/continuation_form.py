"""Split continuation job editor, with model parameters and task controls."""
from dataclasses import asdict
import tkinter as tk
from tkinter import ttk, messagebox
from app.parameter_inspection import seed_parameters
from model.shared_eqn import Parameters
from model.parameter_metadata import GROUPS
from model.continuation_config import parse_job
from app.core.job_eta import (
    bind_creation_estimate,
    continuation_values,
)


# Short display labels only; backend keys and validation remain unchanged.
CONTINUATION_LABELS = {
    'depth_count': 'Depth count',
    'depths': 'Depths (m)',
    'depth_workers': 'Workers (parallel depths)',
    'x_parameter': 'X parameter',
    'x_min': 'X min',
    'x_max': 'X max',
    'x_resolution': 'X resolution (step limit)',
    'y_parameter': 'Y parameter',
    'y_min': 'Y min',
    'y_max': 'Y max',
    'y_resolution': 'Y resolution (step limit)',
    'discovery_points': 'Discovery points (per transect)',
    'transects': 'Transects (per axis)',
    'discovery_refinement': 'Discovery refinement (local seed searches)',
    'max_steps': 'Max steps (per direction)',
    'adaptive_step_factor': 'Adaptive step factor (max step multiplier)',
    'reuse_native': 'Native reuse (experimental)',
    'region_samples': 'Region samples (stability checks)',
    'refinement_rounds': 'Refinement rounds',
    'refinement_searches': 'Refinement searches (per round)',
}


class ContinuationForm(tk.Frame):
    def __init__(self, parent, app, spec, seed_id=None):
        from app.ui import C, label, button, px
        super().__init__(parent, bg=C['bg'])
        self.app, self.spec = app, spec
        self.is_timeseries = spec.form == 'timeseries'
        self.dimensions = 0 if self.is_timeseries else 1 if spec.form == 'continuation_1d' else 2
        self.controls, self.widgets = {}, {}
        self.seed = tk.StringVar(value='')
        self.name = tk.StringVar(value=spec.name)
        top = tk.Frame(self, bg=C['bg']); top.pack(fill='x', pady=(0, px(6)))
        button(top, '← Analyses', lambda: app.navigate('Create job')).pack(side='left')
        label(top, 'Seed', bg=C['bg']).pack(side='left', padx=px(8))
        entry = ttk.Entry(top, textvariable=self.seed, width=25)
        entry.pack(side='left', fill='x', expand=True)
        entry.bind('<Return>', lambda e: self.load_seed())
        button(top, 'Load', self.load_seed).pack(side='left', padx=px(6))
        button(top, 'Browse seed library', self.browse).pack(side='left')
        self.note = tk.StringVar(value='Choose a seed or edit the model. Continuation axes and depth are set on the right.')
        if self.is_timeseries:self.note.set('Choose a seed or edit the model. Time, initial biomasses, plots and depths are set on the right.')
        label(self, '', 9, muted=True, bg=C['bg'], textvariable=self.note,
              wraplength=px(900), justify='left').pack(anchor='w', pady=(0, px(8)))
        bottom = tk.Frame(self, bg='white'); bottom.pack(side='bottom', fill='x', pady=(px(8),0))
        label(bottom, 'Run name', 10, True).pack(side='left', padx=px(8))
        ttk.Entry(bottom, textvariable=self.name, width=25).pack(side='left', fill='x', expand=True)
        draft_button = button(bottom, 'Create draft', lambda: self.save(False))
        draft_button.pack(side='left', padx=px(6))
        queue_button = button(bottom, 'Create & queue', lambda: self.save(True), primary=True)
        queue_button.pack(side='left')
        split = tk.PanedWindow(self, orient='horizontal', bg=C['line'], sashwidth=px(7), bd=0)
        split.pack(fill='both', expand=True)
        left, right = (tk.Frame(split,bg='white') for _ in range(2))
        split.add(left, minsize=px(280), width=px(400))
        split.add(right, minsize=px(300))
        header = tk.Frame(left,bg='white'); header.pack(fill='x', padx=px(10), pady=px(8))
        label(header,'Model parameters',12,True).pack(side='left')
        button(header,'Reset defaults',self.reset).pack(side='right')
        form = self.scroll_panel(left)
        for group, items in GROUPS:
            label(form,group,11,True).pack(anchor='w',padx=px(10),pady=(px(12),px(4)))
            for key, description, unit in items:
                row=tk.Frame(form,bg='white');row.pack(fill='x',padx=px(10),pady=px(4))
                caption=tk.Frame(row,bg='white');caption.pack(side='left',fill='x',expand=True)
                label(caption,f'{key} · {description}',9).pack(anchor='w')
                label(caption,unit,8,muted=True).pack(anchor='w')
                self.add_input(row,key,width=10).pack(side='right')
        label(right,'Time-series settings' if self.is_timeseries else 'Continuation settings',12,True).pack(anchor='w',padx=px(12),pady=px(12))
        tasks=self.scroll_panel(right)
        # Display task controls in logical groups, using the registered schema
        # as the single source of defaults, choices and supported settings.
        if self.is_timeseries:
            groups = [('Time series', ('depth_count','depths','duration','samples',
                                       'initial_M','initial_S','initial_P','plot_groups'))]
        else:
            groups = [
                ('Depth and workers', ('depth_count','depths','depth_workers')),
                ('Continuation axes', ('x_parameter','x_min','x_max','x_resolution',
                                       'y_parameter','y_min','y_max','y_resolution')),
                ('Equilibrium discovery', ('discovery_points','transects','discovery_refinement')),
                ('Native continuation', ('max_steps','adaptive_step_factor','reuse_native')),
                ('MSP periodic orbits', ('periodic_analysis','periodic_mode','periodic_samples',
                                        'periodic_amplitudes','periodic_period_factors','periodic_nodes',
                                        'periodic_order','periodic_max_steps')),
                ('Region refinement', ('region_samples','refinement_rounds',
                                                      'refinement_searches')),
            ]
        model_keys = set(asdict(Parameters()))
        shown = set()

        def add_group(title, keys):
            keys = [key for key in keys if key in spec.parameters
                    and key not in model_keys and key not in shown]
            if not keys:
                return
            label(tasks,title,11,True).pack(anchor='w',padx=px(12),pady=(px(12),px(4)))
            for key in keys:
                meta=spec.parameters[key]
                row=tk.Frame(tasks,bg='white');row.pack(fill='x',padx=px(12),pady=px(6))
                caption = (meta.get('description', key) if self.is_timeseries else
                           CONTINUATION_LABELS.get(key, key.replace('_', ' ').capitalize()))
                label(row,caption,9,wraplength=px(340),justify='left').pack(anchor='w')
                self.add_input(row,key,width=27).pack(fill='x',pady=(px(3),0))
                shown.add(key)

        for title,keys in groups:
            add_group(title,keys)
        # Automatically expose future task settings declared by continuation_config.
        # Do not add undeclared options: the job validator must accept every field.
        if not self.is_timeseries:
            add_group('Additional settings', spec.parameters)
        help_text=('MS, S, P creates three figures: M and S together, S alone, P alone.\n'
                   'N is available too. Initial biomasses and all plots use mmol N/m².\n'
                   'The same areal initial biomasses apply at every depth.\n'
                   '1 / 2 / 4 depths use one / two / 2×2 panels per figure.\n'
                   'Plots are saved as PNG and vector PDF.' if self.is_timeseries else
                   '1 depth → one panel · 2 → side by side · 4 → 2×2\n'
              'γ and β are discrete; not continuation axes.\n'
              'Output: PNG and PDF.')
        if self.dimensions == 2 and 'depth_workers' in self.controls:
            help_text += ('\nWorkers ≤ depth count. Native reuse: off initially. '
                          'Step factor 1: original step limit.')
        label(tasks,help_text,9,muted=True,justify='left',
              wraplength=px(340)).pack(anchor='w',padx=px(12),pady=px(12))
        for key in ('x_parameter','y_parameter'):
            if key in self.controls:self.controls[key].trace_add('write',self.update_disabled)
        self.controls['depth_count'].trace_add('write',self.depth_layout)
        self.update_disabled()
        if seed_id:self.seed.set(seed_id);self.load_seed()

        bind_creation_estimate(
            self,
            app.repo,
            spec,
            lambda: continuation_values(self),
            (
                (draft_button, 'Create draft'),
                (queue_button, 'Create & queue'),
            ),
        )

    def scroll_panel(self,parent):
        canvas=tk.Canvas(parent,bg='white',highlightthickness=0)
        bar=ttk.Scrollbar(parent,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        inner=tk.Frame(canvas,bg='white');window=canvas.create_window((0,0),window=inner,anchor='nw')
        inner.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width))
        return inner

    def add_input(self,parent,key,width):
        meta=self.spec.parameters[key]
        var=tk.StringVar(value=str(meta['default']));self.controls[key]=var
        options=meta.get('options')
        if key in ('gamma','beta'):options=[0,1] if key=='gamma' else [1,2]
        widget=(ttk.Combobox(parent,textvariable=var,values=options,state='readonly',width=width)
                if options else ttk.Entry(parent,textvariable=var,width=width))
        self.widgets[key]=widget
        return widget

    def update_disabled(self,*_):
        selected={self.controls[k].get() for k in ('x_parameter','y_parameter') if k in self.controls}
        for key in asdict(Parameters()):
            self.widgets[key].configure(state='disabled' if key in selected or key=='H' else
                                        'readonly' if key in ('gamma','beta') else 'normal')

    def depth_layout(self,*_):
        count=int(self.controls['depth_count'].get())
        old=[x.strip() for x in self.controls['depths'].get().split(',') if x.strip()]
        candidates=old+ [str(n) for n in (2,4,6,8,10,12) if str(n) not in old]
        self.controls['depths'].set(', '.join(candidates[:count]))

    def reset(self):
        for key,value in asdict(Parameters()).items():self.controls[key].set(str(value))
        self.seed.set('');self.note.set('Model defaults restored; task settings retained.')

    def load_seed(self):
        record=self.app.repo.get_seed(self.seed.get().strip())
        if record is None:
            messagebox.showerror('Seed not found','Choose a saved Seed-ID.',parent=self);return False
        try:values,applied,ignored=seed_parameters(record)
        except (ValueError,KeyError,TypeError) as exc:
            messagebox.showerror('Cannot load seed',str(exc),parent=self);return False
        for key,value in values.items():
            if key in ('gamma','beta') and isinstance(value,(int,float)) and value in (0,1,2):value=int(value)
            self.controls[key].set(str(value))
        # A saved depth is a useful starting scenario; the user can then add panels.
        self.controls['depth_count'].set('1')
        self.controls['depths'].set(str(values['H']))
        self.note.set(f'Loaded {record["seed_id"]}: {len(applied)} model inputs; missing values use defaults. '
                      + ('Time-series initial biomasses are set on the right.' if self.is_timeseries else 'Selected axes use the ranges on the right.') +
                      (' Ignored task/non-model fields: '+', '.join(ignored) if ignored else ''))
        return True

    def browse(self):
        from app.ui import SeedBrowser,px
        picker=tk.Toplevel(self);picker.title('Select parameter seed');picker.transient(self.app)
        picker.geometry(f'{min(px(1020),self.winfo_screenwidth()-50)}x{min(px(710),self.winfo_screenheight()-90)}')
        self.app._set_child_icon(picker)
        def picked(record):
            self.seed.set(record['seed_id'])
            if self.load_seed():picker.destroy()
        SeedBrowser(picker,self.app.repo,picked).pack(fill='both',expand=True,padx=15,pady=15)
        picker.grab_set()

    def save(self,queue):
        from app.core.discovery import validate
        from app.core.worker import start_worker
        try:
            values={k:v.get() for k,v in self.controls.items()}
            # Disabled entries are placeholders, never independent inputs.
            for prefix in ('x','y')[:self.dimensions]:
                values[values[prefix+'_parameter']]=(float(values[prefix+'_min'])+float(values[prefix+'_max']))/2
            values['H']=float(values['depths'].split(',')[0])
            values=validate(self.spec,values)
            if self.is_timeseries:
                from model.timeseries_config import parse_job as parse_timeseries
                parse_timeseries(values)
            else:parse_job(values,self.dimensions)
            job=self.app.repo.create_job(self.name.get(),self.spec,values,seed_id=self.seed.get().strip())
            if queue:self.app.repo.enqueue([job['id']]);start_worker()
            self.app.navigate('Job queue')
        except (ValueError,TypeError,OSError) as exc:
            messagebox.showerror('Cannot create analysis',str(exc),parent=self)
