"""Range/fixed-value editor for sensitivity jobs."""
import json
import tkinter as tk
from tkinter import ttk,messagebox
from dataclasses import asdict
from model.shared_eqn import Parameters
from model.parameter_metadata import GROUPS
from model.sensitivity_config import default_ranges,parse_job
from app.continuation_form import ContinuationForm
from app.core.job_eta import (bind_creation_estimate, sensitivity_values,)


class SensitivityForm(tk.Frame):
    scroll_panel=ContinuationForm.scroll_panel

    def __init__(self,parent,app,spec):
        from app.ui import C,label,button,px
        super().__init__(parent,bg=C['bg'])
        self.app,self.spec=app,spec
        self.rows={};self.settings={};self.name=tk.StringVar(value=spec.name)
        bottom=tk.Frame(self,bg='white');bottom.pack(side='bottom',fill='x',pady=8)
        ttk.Entry(bottom,textvariable=self.name).pack(side='left',fill='x',expand=True)
        draft_button = button(bottom,'Create draft',lambda:self.save(False))
        draft_button.pack(side='left',padx=6)
        queue_button = button(bottom,'Create & queue',lambda:self.save(True),primary=True)
        queue_button.pack(side='left')
        # Split the available form width equally (excluding the draggable sash).
        sash_width=px(7)
        split=tk.PanedWindow(self,orient='horizontal',sashwidth=sash_width,
                             sashpad=0,bd=0,bg=C['line'])
        split.pack(fill='both',expand=True)
        left=tk.Frame(split,bg='white');right=tk.Frame(split,bg='white')
        split.add(left,minsize=px(120),stretch='always')
        split.add(right,minsize=px(120),stretch='always')
        self._split_width=0
        def size_panels(event):
            # Centre on first display and on width changes; manual dragging
            # remains possible without a resize immediately undoing it.
            width=event.width
            if width<=sash_width or width==self._split_width:return
            self._split_width=width
            position=(width-sash_width)//2
            def centre():
                if split.winfo_exists() and split.winfo_width()==width:
                    split.sash_place(0,position,0)
            split.after_idle(centre)
        split.bind('<Configure>',size_panels)
        label(left,'Model parameters',12,True).pack(anchor='w',padx=px(10),pady=(px(10),0))
        button(left,'Reset model defaults and ranges',self.reset).pack(anchor='w',padx=px(10),pady=px(8))
        fields=self.scroll_panel(left)
        defaults=asdict(Parameters());ranges=default_ranges()
        for group,items in GROUPS:
            label(fields,group,11,True).pack(anchor='w',padx=px(10),pady=(px(12),px(4)))
            for key,description,unit in items:
                row=tk.Frame(fields,bg='white');row.pack(fill='x',padx=px(10),pady=px(5))
                # Give the parameter name and units the full row width so that
                # long descriptions never compete with the range inputs.
                caption=label(row,f'{key} · {description}',9,justify='left',anchor='w')
                caption.pack(fill='x')
                units=label(row,unit,8,muted=True,justify='left',anchor='w')
                units.pack(fill='x')
                def wrap_caption(event,widgets=(caption,units)):
                    for widget in widgets:
                        widget.configure(wraplength=max(1,event.width))
                row.bind('<Configure>',wrap_caption)
                included=tk.BooleanVar(value=key in ranges)
                fixed=tk.StringVar(value=str(defaults[key]))
                lo,hi=ranges.get(key,(0,1) if key=='gamma' else (1,2))
                low,high=tk.StringVar(value=str(lo)),tk.StringVar(value=str(hi))
                controls=tk.Frame(row,bg='white');controls.pack(fill='x',pady=(px(3),px(2)))
                toggle=button(controls,'',lambda k=key:self.toggle(k),width=7)
                toggle.pack(side='right',anchor='s',padx=(px(6),0))
                inputs=tk.Frame(controls,bg='white');inputs.pack(side='left',fill='x',expand=True)
                self.rows[key]=(included,fixed,low,high,inputs,toggle)
                self.render(key)
                tk.Frame(row,bg=C['line'],height=px(1)).pack(fill='x',pady=(px(5),0))
        label(right,'LHS settings',12,True).pack(anchor='w',padx=px(12),pady=px(12))
        # Independently scroll settings at small window heights or high DPI.
        tasks=self.scroll_panel(right)
        for key in ('sample_count','random_seed','workers','accuracy'):
            meta=spec.parameters[key];var=tk.StringVar(value=str(meta['default']));self.settings[key]=var
            label(tasks,meta['description'],10).pack(anchor='w',padx=px(12),pady=(px(10),px(3)))
            widget=ttk.Combobox(tasks,textvariable=var,values=meta['options'],state='readonly') if 'options' in meta else ttk.Entry(tasks,textvariable=var)
            widget.pack(fill='x',padx=px(12))
        help_label=label(tasks,'Uniform, independent LHS sampling.\nStarting ranges are 50–150% of model defaults; these are editable suggestions, not thesis uncertainty bounds.\n\nBistability means at least two strictly stable equilibria. ΔS is the largest seagrass difference between them (mmol N/m²).\n\nStandard / Thorough: 257 / 513 root-grid points with refinement. Finite root searches can miss narrow branches.\n\nAll samples become seeds. Figures: PNG + PDF.',9,muted=True,wraplength=px(300),justify='left',anchor='w')
        help_label.pack(fill='x',padx=px(12),pady=px(15))
        tasks.bind('<Configure>',lambda e:help_label.configure(wraplength=max(1,e.width-px(24))),add='+')
        bind_creation_estimate(
            self,
            app.repo,
            spec,
            lambda: sensitivity_values(self),
            (
                (draft_button, 'Create draft'),
                (queue_button, 'Create & queue'),
            ),
        )

    def toggle(self,key):
        self.rows[key][0].set(not self.rows[key][0].get());self.render(key)

    def render(self,key):
        enabled,fixed,low,high,frame,button=self.rows[key]
        for w in frame.winfo_children():w.destroy()
        from app.ui import label,px
        # Flexible fields use all available space, with an explicit range/fixed
        # caption. Toggling changes only the displayed inputs, not their values.
        for col in (0,1):frame.columnconfigure(col,weight=0,uniform='')
        entries=(('Min',low),('Max',high)) if enabled.get() else (('Fixed value',fixed),)
        for i,(caption,var) in enumerate(entries):
            frame.columnconfigure(i,weight=1,uniform='sensitivity_inputs')
            label(frame,caption,8,muted=True,anchor='w').grid(row=0,column=i,sticky='ew',padx=(0,px(4)))
            ttk.Entry(frame,textvariable=var,width=6).grid(row=1,column=i,sticky='ew',padx=(0,px(4)))
        button.configure(text='Exclude' if enabled.get() else 'Include')

    def reset(self):
        defaults=asdict(Parameters());ranges=default_ranges()
        for k,(enabled,fixed,low,high,_,_) in self.rows.items():
            fixed.set(str(defaults[k]));enabled.set(k in ranges)
            a,b=ranges.get(k,(0,1) if k=='gamma' else (1,2));low.set(str(a));high.set(str(b));self.render(k)

    def save(self,queue):
        from app.core.discovery import validate
        from app.core.worker import start_worker
        try:
            values={k:v.get() for k,v in self.settings.items()};ranges={}
            for k,(enabled,fixed,low,high,_,_) in self.rows.items():
                values[k]=fixed.get()
                if enabled.get():ranges[k]=[float(low.get()),float(high.get())]
            values['ranges']=json.dumps(ranges)
            values=validate(self.spec,values);parse_job(values)
            job=self.app.repo.create_job(self.name.get(),self.spec,values)
            if queue:self.app.repo.enqueue([job['id']]);start_worker()
            self.app.navigate('Job queue')
        except (ValueError,TypeError,OSError) as exc:messagebox.showerror('Cannot create sensitivity run',str(exc),parent=self)
