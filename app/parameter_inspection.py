"""Live, session-local parameter exploration; never creates jobs or edits seeds."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import tkinter as tk
from tkinter import ttk, messagebox

from model.shared_eqn import Parameters, solve_equilibria, per_capita

from model.parameter_metadata import GROUPS


def inspect_parameters(values):
    """Pure worker entry point; no Tk access."""
    p = Parameters.from_mapping({k: float(v) for k, v in values.items()})
    result = []
    for eq in solve_equilibria(p):
        rates = per_capita(eq.biomass, p)
        result.append((eq, tuple(float(rates[i]) if eq.biomass[i] == 0 else None
                                 for i in range(3))))
    return result


def biomass_summary(eq, depth):
    """Display resident biomasses and dissolved nitrogen on an areal basis."""
    values = [f'{species} = {biomass * (depth if species == "P" else 1):.7g}'
              for species, biomass in zip('MSP', eq.biomass)
              if species in eq.community]
    values.append(f'N = {eq.nitrogen * depth:.7g}')
    return '   '.join(values) + ' mmol N/m²'


def seed_parameters(record):
    """Load a reproducible snapshot: missing model inputs use thesis defaults."""
    source = json.loads(record['values_json'])
    if not isinstance(source, dict):
        raise ValueError('Seed parameters must be an object')
    values = asdict(Parameters())
    aliases = {'depth': 'H', 'nwc': 'N_wc'}  # Existing demo manifest names.
    applied = set()
    ignored = []
    for name, value in source.items():
        target = aliases.get(name, name)
        if target not in values:
            ignored.append(name)
            continue
        if name in aliases and target in source:
            continue  # Canonical model names take precedence.
        values[target] = value
        applied.add(target)
    if not applied:
        raise ValueError('This seed contains no recognized SIMP parameters')
    return values, applied, sorted(ignored)


class ParameterInspection(tk.Frame):
    def __init__(self, parent, saved):
        from app.ui import C, label, button, px
        super().__init__(parent, bg=C['bg'])
        self.C, self.label, self.px = C, label, px
        self.saved = saved
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='simp-inspection')
        self.future = None
        self.revision = 0
        self.submitted = -1
        self.ready = False
        self.closed = False
        self.debounce = None
        self.poll_id = None
        self.bind('<Destroy>', self._destroy, add='+')
        self.status = tk.StringVar(value='Preparing equilibrium search…')
        seedbar = tk.Frame(self, bg=C['bg'])
        seedbar.pack(fill='x', pady=(0, px(6)))
        label(seedbar, 'Seed', 10, True, bg=C['bg']).pack(side='left', padx=(0, px(8)))
        self.seed_id = tk.StringVar()
        seed_entry = ttk.Entry(seedbar, textvariable=self.seed_id, width=35)
        seed_entry.pack(side='left', fill='x', expand=True, padx=(0, px(8)))
        seed_entry.bind('<Return>', lambda e: self.load_seed_id())
        button(seedbar, 'Load Seed-ID', self.load_seed_id).pack(side='left', padx=(0, px(8)))
        button(seedbar, 'Browse seed library', self.open_seed_picker).pack(side='left')
        self.seed_note = tk.StringVar(value='Choose a saved seed or edit parameters below.')
        label(self, '', 9, muted=True, bg=C['bg'], textvariable=self.seed_note,
              wraplength=px(900), justify='left').pack(anchor='w', pady=(0, px(6)))
        label(self, 'Edit parameters to explore equilibria. Changes apply only to this inspection.',
              10, muted=True, bg=C['bg']).pack(anchor='w', pady=(0, px(8)))
        self.status_label = label(self, '', 10, bg=C['bg'], textvariable=self.status,
                                  wraplength=px(900), justify='left')
        self.status_label.pack(anchor='w', pady=(0, px(8)))
        split = tk.PanedWindow(self, orient='horizontal', bg=C['line'], sashwidth=px(7), bd=0)
        split.pack(fill='both', expand=True)
        left = tk.Frame(split, bg='white')
        right = tk.Frame(split, bg='white')
        split.add(left, minsize=px(290), width=px(340), stretch='never')
        split.add(right, minsize=px(360), stretch='always')
        top = tk.Frame(left, bg='white'); top.pack(fill='x', padx=px(10), pady=px(8))
        label(top, 'Thesis parameters', 12, True).pack(side='left')
        button(top, 'Reset defaults', self.reset).pack(side='right')
        form = self.scroll_panel(left)
        self.variables = {}
        defaults = asdict(Parameters())
        for group, items in GROUPS:
            label(form, group, 11, True).pack(anchor='w', padx=px(10), pady=(px(14), px(4)))
            for name, description, unit in items:
                row = tk.Frame(form, bg='white'); row.pack(fill='x', padx=px(10), pady=px(4))
                text = tk.Frame(row, bg='white'); text.pack(side='left', fill='x', expand=True)
                label(text, f'{name} · {description}', 9).pack(anchor='w')
                label(text, unit, 8, muted=True).pack(anchor='w')
                var = tk.StringVar(value=saved.get(name, str(defaults[name])))
                self.variables[name] = var
                if name in ('gamma', 'beta'):
                    entry = ttk.Combobox(row, textvariable=var, width=8, state='readonly',
                                         values=('0', '1') if name == 'gamma' else ('1', '2'))
                else:
                    entry = ttk.Entry(row, textvariable=var, width=10)
                entry.pack(side='right', padx=(px(5), 0))
                var.trace_add('write', self.changed)
        label(form, 'Grazing pairs: (γ, β) = (0,1), (1,1), (1,2).\n'
              'Defaults: Table 3.4 and reference environment §3.4.',
              9, muted=True, justify='left').pack(anchor='w', padx=px(10), pady=px(12))
        self.results = self.scroll_panel(right)
        self.changed()
        self.poll_id = self.after(60, self.poll)

    def scroll_panel(self, parent):
        canvas = tk.Canvas(parent, bg='white', bd=0, highlightthickness=0)
        bar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y'); canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg='white')
        window = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window, width=e.width))
        return inner

    def changed(self, *_):
        if self.closed: return
        self.saved.update({k: v.get() for k, v in self.variables.items()})
        self.revision += 1
        self.ready = False
        if self.debounce: self.after_cancel(self.debounce)
        self.status.set('Parameters changed — recalculating…')
        self.status_label.configure(fg=self.C['muted'])
        # Do not display previous equilibria as though they match edited parameters.
        for child in self.results.winfo_children(): child.destroy()
        self.debounce = self.after(300, self._ready)

    def _ready(self):
        self.debounce = None
        self.ready = True

    def reset(self):
        self.seed_id.set('')
        self.seed_note.set('Thesis defaults restored.')
        for name, value in asdict(Parameters()).items():
            self.variables[name].set(str(value))

    def load_seed_id(self):
        record = self.winfo_toplevel().repo.get_seed(self.seed_id.get().strip())
        if record is None:
            messagebox.showerror('Seed not found', 'Enter a saved Seed-ID or browse the seed library.', parent=self)
            return
        self.apply_seed(record)

    def apply_seed(self, record):
        try:
            values, applied, ignored = seed_parameters(record)
        except (ValueError, TypeError, KeyError) as exc:
            messagebox.showerror('Cannot load seed', str(exc), parent=self)
            return False
        self.seed_id.set(record['seed_id'])
        for name, value in values.items():
            if name in ('gamma', 'beta') and isinstance(value, (int, float)) and value in (0, 1, 2):
                value = int(value)
            self.variables[name].set(str(value))
        note = (f'Loaded {record["seed_id"]}: {len(applied)} model values; '
                f'{len(values)-len(applied)} missing values use thesis defaults. Fields remain editable.')
        if ignored:
            note += ' Ignored non-model fields: ' + ', '.join(ignored) + '.'
        self.seed_note.set(note)
        return True

    def open_seed_picker(self):
        from app.ui import SeedBrowser
        app = self.winfo_toplevel()
        picker = tk.Toplevel(self)
        picker.title('Select a seed to inspect')
        picker.geometry(f'{min(self.px(1020), app.winfo_screenwidth()-self.px(50))}x'
                        f'{min(self.px(710), app.winfo_screenheight()-self.px(90))}')
        picker.configure(bg=self.C['bg'])
        picker.transient(app)
        app._set_child_icon(picker)
        def picked(record):
            if self.apply_seed(record): picker.destroy()
        SeedBrowser(picker, app.repo, on_use=picked).pack(fill='both', expand=True, padx=15, pady=15)
        picker.grab_set()

    def poll(self):
        if self.closed: return
        if self.future is not None and self.future.done():
            future, revision = self.future, self.submitted
            self.future = None
            if revision == self.revision:
                try:
                    rows = future.result()
                    self.render(rows)
                    self.status.set(f'{len(rows)} equilibria found · '
                                    f'{sum(e.stability.classification == "stable" for e, _ in rows)} stable')
                    self.status_label.configure(fg=self.C['good'])
                except Exception as exc:
                    self.status.set(f'Cannot calculate: {exc}')
                    self.status_label.configure(fg=self.C['red'])
        if self.ready and self.future is None and self.submitted != self.revision:
            self.submitted = self.revision
            values = {k: v.get() for k, v in self.variables.items()}
            self.future = self.executor.submit(inspect_parameters, values)
        self.poll_id = self.after(60, self.poll)

    def render(self, rows):
        label, px, C = self.label, self.px, self.C
        label(self.results, 'Equilibrium stability margin', 12, True).pack(anchor='w', padx=px(12), pady=(px(12), 0))
        label(self.results, 'Largest eigenvalue real part (1/day): left of zero is stable.',
              9, muted=True).pack(anchor='w', padx=px(12), pady=(0, px(5)))
        plot = tk.Canvas(self.results, bg='white', height=px(32+22*len(rows)), highlightthickness=0)
        plot.pack(fill='x', padx=px(12))
        plot.bind('<Configure>', lambda e: self.draw_plot(plot, rows))
        for classification, title, color in (
            ('stable', 'Stable equilibria', C['good']),
            ('unstable', 'Unstable equilibria', C['red']),
            ('nonhyperbolic', 'Near-zero eigenvalues — stability unresolved', C['amber'])):
            group = [(eq, rates) for eq, rates in rows
                     if eq.stability.classification == classification]
            heading = label(self.results, f'{title} ({len(group)})', 12, True)
            heading.configure(fg=color)
            heading.pack(anchor='w', padx=px(12), pady=(px(14), px(6)))
            if not group:
                label(self.results, 'None found', 9, muted=True).pack(anchor='w', padx=px(12))
            for eq, rates in group:
                block = tk.Frame(self.results, bg='#F4F9F2', highlightbackground=C['line'], highlightthickness=1)
                block.pack(fill='x', padx=px(12), pady=(0, px(7)))
                def line(text, bold=False, size=9):
                    widget = label(block, text, size, bold, bg='#F4F9F2', justify='left', anchor='w')
                    widget.pack(fill='x', padx=px(8), pady=px(2))
                    block.bind('<Configure>', lambda e, w=widget: w.configure(wraplength=max(100, e.width-px(20))), add='+')
                line(eq.community, True, size=12)
                line(biomass_summary(eq, float(self.variables['H'].get())))
                line('Invasion rates (1/day): ' + '   '.join(
                    f'{s}: {v:+.6g}' if v is not None else f'{s}: resident'
                    for s, v in zip('MSP', rates)))
                line('Eigenvalues (1/day): ' + ', '.join(
                    f'{v.real:+.5g}' + (f'{v.imag:+.5g}i' if abs(v.imag)>1e-12 else '')
                    for v in eq.stability.eigenvalues))
        label(self.results, 'Positive invasion rate: an absent species can invade. Resident species are marked explicitly.\n'
              'Finite root searches may miss unresolved folds; near-zero eigenvalues need further checks.',
              9, muted=True, wraplength=px(500), justify='left').pack(anchor='w', padx=px(12), pady=px(14))

    def draw_plot(self, canvas, rows):
        canvas.delete('all')
        px = self.px
        width = canvas.winfo_width()
        origin = (width-px(30))/2
        radius = max(10, (width-px(180))/2)
        extent = max([abs(e.stability.spectral_abscissa) for e, _ in rows]+[1e-8])
        colors = {'stable': self.C['good'], 'unstable': self.C['red'], 'nonhyperbolic': self.C['amber']}
        canvas.create_line(origin, px(22), origin, px(30+22*len(rows)), fill='#9DAAA0')
        canvas.create_text(origin, px(10), text='0', fill=self.C['muted'])
        canvas.create_text(origin-radius, px(10), text=f'{-extent:.3g}', fill=self.C['muted'])
        canvas.create_text(origin+radius, px(10), text=f'{extent:.3g}', fill=self.C['muted'])
        for i, (eq, _) in enumerate(rows, 1):
            y = px(22+22*i)
            x = origin+radius*eq.stability.spectral_abscissa/extent
            color = colors[eq.stability.classification]
            canvas.create_text(px(4), y, text=eq.community, anchor='w', fill=self.C['text'])
            canvas.create_line(origin, y, x, y, fill=color, width=px(3))
            canvas.create_oval(x-px(3), y-px(3), x+px(3), y+px(3), fill=color, outline=color)
            canvas.create_text(width-px(4), y, text=f'{eq.stability.spectral_abscissa:+.4g}',
                               anchor='e', fill=color)

    def _destroy(self, event):
        if event.widget is not self: return
        self.closed = True
        for ident in (self.debounce, self.poll_id):
            if ident: self.after_cancel(ident)
        self.executor.shutdown(wait=False, cancel_futures=True)
