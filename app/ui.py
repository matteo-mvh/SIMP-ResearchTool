"""Desktop GUI; queued analyses use a worker, inspection uses a background thread."""
from __future__ import annotations
import json
import math
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime

# Pillow preserves sharpness when high-resolution PNGs are scaled for previews.
# Unlike Tk PhotoImage.subsample(), this does not discard whole pixel rows.
from PIL import Image, ImageTk
from app.core.dpi import get_dpi_for_window

from app.core.discovery import discover, validate
from app.core.storage import Repository
from app.core.worker import request, start_worker, worker_is_alive
from app.core.paths import ROOT, DATA_ROOT, RESULTS, STATE, ASSETS

# Greenpeace-inspired research palette. The GUI is not affiliated with Greenpeace.
C = {
  'nav':'#063D2D','nav2':'#116044','bg':'#F4F9F2','card':'#FFFFFF',
  'line':'#DCE9DA','text':'#163B2A','muted':'#637E68','blue':'#16803F',
  'blue2':'#E3F5E7','good':'#087A3D','red':'#B94F4F','amber':'#B27B20',
  'lime':'#77CB16','nav_text':'#D8F1DC'
}

# Pixel-based Tk paddings and sizes are expressed in logical 96-DPI units.
# Font point sizes scale separately via Tk's native dpi scaling.
GUI_SCALE = 1.0

def px(n):
    return max(1, round(n * GUI_SCALE))



def human_date(iso):
    if not iso: return '—'
    try: return datetime.fromisoformat(iso).strftime('%d %b · %H:%M')
    except (ValueError,TypeError): return str(iso)[:16]


def button(parent,text,cmd,primary=False,**kwargs):
    return tk.Button(parent,text=text,command=cmd,bd=0,relief='flat',cursor='hand2',
                     font=('Segoe UI',10,'bold' if primary else 'normal'),
                     bg=C['blue'] if primary else C['blue2'],highlightthickness=0,
                     fg='white' if primary else C['blue'],
                     activebackground='#0C6332' if primary else '#C9EDCE',
                     activeforeground='white' if primary else C['blue'],
                     padx=px(13),pady=px(9),**kwargs)


def label(parent,text,size=10,bold=False,muted=False,bg=None,**kwargs):
    return tk.Label(parent,text=text,font=('Segoe UI',size,'bold' if bold else 'normal'),
                    fg=C['muted'] if muted else C['text'],bg=bg or C['card'],**kwargs)


def card(parent,**opts):
    return tk.Frame(parent,bg=C['card'],highlightbackground=C['line'],highlightthickness=1,**opts)


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SIMP Control Center')
        # Windows may otherwise render the entire Tk window at 96 DPI and then
        # enlarge it as a blurry bitmap. start_app.py enables DPI awareness first.
        # Use the physical monitor DPI for font metrics and layout dimensions.
        native_dpi = get_dpi_for_window(self.winfo_id())
        if sys.platform == 'win32':
            if native_dpi is None:
                native_dpi = round(self.winfo_fpixels('1i'))
            self.dpi_scale = max(1.0, min(2.75, native_dpi / 96.0))
            self.tk.call('tk', 'scaling', native_dpi / 72.0)
        else:
            self.dpi_scale = 1.0
        global GUI_SCALE
        GUI_SCALE = self.dpi_scale
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        width = min(px(1330), max(px(800), sw - px(60)))
        height = min(px(860), max(px(540), sh - px(85)))
        self.geometry(f'{width}x{height}')
        self.minsize(min(px(1040), width), min(px(660), height))
        self.configure(bg=C['bg'])
        self.repo=Repository()
        self.analyses=discover()
        self.page='Overview'
        self.selected_analysis=None
        self.controls={}
        self.extra_seed_controls={}
        self.current_result_id=None
        self.img=None
        self.brand_photo=None
        self._set_window_icon()
        self.build_style()
        self.build_shell()
        if os.environ.get('SIMP_DISABLE_AUTOSTART') != '1':
            try: start_worker()
            except OSError: pass
        self.navigate('Overview')
        self.bind_all('<MouseWheel>', self._on_mousewheel, add='+')
        self.bind_all('<Button-4>', self._on_mousewheel, add='+')
        self.bind_all('<Button-5>', self._on_mousewheel, add='+')
        self.after(700,self.heartbeat)

    def _set_window_icon(self):
        """Set window, taskbar and Alt-Tab identity (Windows icon + cross-platform PNG)."""
        assets=ASSETS
        try:
            self._icon_photo=tk.PhotoImage(file=str(assets/'simp_icon.png'))
            self.iconphoto(True,self._icon_photo)
            if sys.platform=='win32':
                self.iconbitmap(str(assets/'simp_icon.ico'))
        except (OSError,tk.TclError):
            pass


    def _resize_page(self, event=None):
        """Keep content full-width, but make it scroll if its natural height overflows."""
        if not self.page_canvas.winfo_exists(): return
        width=max(1,self.page_canvas.winfo_width())
        height=(self.page_canvas.winfo_height() if self.page=='Parameter inspection' or
                (self.page=='Create job' and self.selected_analysis and self.selected_analysis.form in ('continuation_1d','continuation_2d','timeseries','sensitivity'))
                else max(self.body.winfo_reqheight(),self.page_canvas.winfo_height()))
        self.page_canvas.itemconfigure(self._body_window,width=width,height=height)
        self.page_canvas.configure(scrollregion=(0,0,width,height))

    def _on_mousewheel(self,event):
        """Scroll the page naturally on Windows; let Treeview/Text scroll themselves."""
        under=self.winfo_containing(event.x_root,event.y_root)
        if under is None: return
        # Don't intercept scrolling in pop-up windows, text fields, and native tables.
        child=under
        within=False
        while child is not None:
            if child is self.right: within=True;break
            child=getattr(child,'master',None)
        if not within: return
        current=under
        while current is not None and current is not self.right:
            if isinstance(current,(ttk.Treeview,tk.Text,tk.Listbox,tk.Scrollbar,ttk.Scrollbar,ttk.Combobox)):
                return
            current=getattr(current,'master',None)
        # Scroll the nearest inner form canvas if there is one; otherwise the page.
        current=under
        target=self.page_canvas
        while current is not None and current is not self.right:
            if isinstance(current,tk.Canvas) and current is not self.page_canvas:
                target=current;break
            current=getattr(current,'master',None)
        delta = (-1 if event.num == 4 else 1) if event.num in (4,5) else (-int(event.delta/120) if abs(event.delta)>=120 else (-1 if event.delta>0 else 1))
        if delta:
            target.yview_scroll(delta*3,'units')
        return 'break'

    def build_style(self):
        style=ttk.Style(self)
        style.theme_use('clam')
        style.configure('Treeview',background='#ffffff',fieldbackground='#ffffff',foreground=C['text'],
                        rowheight=px(32),borderwidth=0,font=('Segoe UI',10))
        style.configure('Treeview.Heading',background='#EAF5E9',foreground=C['muted'],
                        font=('Segoe UI',10,'bold'),padding=px(9),relief='flat')
        style.map('Treeview',background=[('selected','#D4F2D3')],foreground=[('selected',C['text'])])
        style.configure('TEntry',padding=px(6))
        style.configure('TCombobox',padding=px(5))
        style.configure('Horizontal.TProgressbar',background=C['blue'],troughcolor='#E1EEE1',borderwidth=0)

    def build_shell(self):
        self.sidebar=tk.Frame(self,bg=C['nav'],width=px(238))
        self.sidebar.pack(side='left',fill='y');self.sidebar.pack_propagate(False)
        brand=tk.Frame(self.sidebar,bg=C['nav'])
        brand.pack(fill='x',padx=16,pady=(20,24))
        try:
            # Generate the navigation icon from the 256px master, never stretch
            # the old 76px thumbnail when Windows uses 150–250% display scaling.
            with Image.open(ASSETS/'simp_icon.png') as master:
                icon_size=px(76)
                self.brand_photo=ImageTk.PhotoImage(master.convert('RGBA').resize(
                    (icon_size,icon_size),Image.Resampling.LANCZOS),master=self)
            tk.Label(brand,image=self.brand_photo,bg=C['nav'],bd=0).pack(side='left',padx=(0,px(8)))
        except (OSError,ValueError,tk.TclError):
            tk.Label(brand,text='SIMP',font=('Segoe UI',17,'bold'),fg='white',bg=C['nav']).pack(side='left')
        name=tk.Frame(brand,bg=C['nav']);name.pack(side='left',fill='x',expand=True)
        tk.Label(name,text='CONTROL',font=('Segoe UI',12,'bold'),fg='white',bg=C['nav']).pack(anchor='w')
        tk.Label(name,text='CENTER',font=('Segoe UI',12,'bold'),fg=C['lime'],bg=C['nav']).pack(anchor='w')
        self.nav={}
        entries=[('Overview','▦'),('Create job','＋'),('Job queue','◷'),
                 ('Results','▤'),('Seed library','◉'),('Parameter inspection','↔'),('Settings','⚙')]
        for key,mark in entries:
            b=tk.Button(self.sidebar,text=f' {mark}    {key}',font=('Segoe UI',11),anchor='w',
                      padx=20,pady=13,bd=0,relief='flat',highlightthickness=0,cursor='hand2',
                      bg=C['nav'],fg=C['nav_text'],activebackground=C['nav2'],activeforeground='white',
                      command=lambda x=key:self.navigate(x))
            b.pack(fill='x',padx=10,pady=2)
            self.nav[key]=b
        tk.Frame(self.sidebar,bg=C['nav']).pack(fill='both',expand=True)
        self.right=tk.Frame(self,bg=C['bg'])
        self.right.pack(side='right',fill='both',expand=True)
        self.header=tk.Frame(self.right,bg=C['bg'])
        self.header.pack(fill='x',padx=px(30),pady=(px(24),px(10)))
        self.head_title=label(self.header,'',23,True,bg=C['bg'])
        self.head_title.pack(side='left')
        button(self.header,'Refresh',self.refresh_page).pack(side='right')
        self.footer=card(self.right)
        self.footer.pack(side='bottom',fill='x',padx=px(30),pady=(0,px(17)))
        self.footer_eta=label(self.footer,'',9,muted=True)
        self.footer_eta.pack(side='bottom',anchor='w',padx=15,pady=(0,8))
        inner=tk.Frame(self.footer,bg=C['card']);inner.pack(fill='x',padx=15,pady=12)
        self.footer_job=label(inner,'No job currently running',10,True)
        self.footer_job.pack(side='left',padx=(0,14))
        self.footer_progress=ttk.Progressbar(inner,length=px(170),mode='determinate',maximum=100)
        self.footer_progress.pack(side='left',padx=(0,10))
        self.footer_phase=label(inner,'Idle',9,muted=True)
        self.footer_phase.pack(side='left',padx=(0,14))
        self.footer_stop=button(inner,'Stop',lambda:self.job_action('stop'))
        self.footer_stop.pack(side='right',padx=(7,0))
        self.footer_pause=button(inner,'Pause',lambda:self.job_action('pause'))
        self.footer_pause.pack(side='right')
        # All pages now use a real scrollable viewport. The status strip remains fixed.
        self.page_wrap=tk.Frame(self.right,bg=C['bg'])
        self.page_wrap.pack(fill='both',expand=True,padx=(px(30),px(18)),pady=(px(8),px(12)))
        self.page_canvas=tk.Canvas(self.page_wrap,bg=C['bg'],bd=0,highlightthickness=0)
        self.page_scroll=ttk.Scrollbar(self.page_wrap,orient='vertical',command=self.page_canvas.yview)
        self.page_canvas.configure(yscrollcommand=self.page_scroll.set)
        self.page_scroll.pack(side='right',fill='y')
        self.page_canvas.pack(side='left',fill='both',expand=True)
        self.body=tk.Frame(self.page_canvas,bg=C['bg'])
        self._body_window=self.page_canvas.create_window((0,0),window=self.body,anchor='nw')
        self.body.bind('<Configure>',self._resize_page)
        self.page_canvas.bind('<Configure>',self._resize_page)

    def navigate(self,page):
        self.page=page
        self.selected_analysis=None
        self.current_result_id=None
        for x,b in self.nav.items():
            b.configure(bg=C['nav2'] if x==page else C['nav'],
                        fg='white' if x==page else C['nav_text'])
        for child in self.body.winfo_children():child.destroy()
        self.page_canvas.yview_moveto(0)
        self.head_title.configure(text=page)
        if page=='Overview':self.show_overview()
        elif page=='Create job':self.show_analyses()
        elif page=='Job queue':self.show_queue()
        elif page=='Seed library':self.show_seeds()
        elif page=='Parameter inspection':
            from app.parameter_inspection import ParameterInspection
            if not hasattr(self, 'inspection_values'): self.inspection_values = {}
            ParameterInspection(self.body, self.inspection_values).pack(fill='both', expand=True)
        elif page=='Results':self.show_results()
        elif page=='Settings':self.show_settings()

    def refresh_page(self):
        self.analyses=discover()
        self.navigate(self.page)

    def heading(self,parent,title,subtitle=None):
        label(parent,title,15,True,bg=C['bg']).pack(anchor='w',pady=(4,2))
        if subtitle:label(parent,subtitle,10,muted=True,bg=C['bg'],wraplength=950,justify='left').pack(anchor='w',pady=(0,17))

    def show_overview(self):
        jobs=self.repo.list_jobs()
        self.heading(self.body,'Research workspace',
                     'Create experiments, run one computation at a time, and review figures without leaving the app.')
        counters=tk.Frame(self.body,bg=C['bg']); counters.pack(fill='x',pady=(4,22))
        n={s:sum(j['status']==s for j in jobs) for s in ('draft','queued','running','paused','completed','failed')}
        for text,value,desc in [('Created',n['draft'],'Ready to schedule'),
                                ('Waiting',n['queued'],'Queued for execution'),
                                ('Running',n['running']+n['paused'],'Maximum one'),
                                ('Completed',n['completed'],'Available in results')]:
            panel=card(counters);panel.pack(side='left',expand=True,fill='both',padx=(0,12))
            label(panel,text.upper(),9,True,muted=True).pack(anchor='w',padx=18,pady=(16,5))
            label(panel,str(value),25,True).pack(anchor='w',padx=18)
            label(panel,desc,9,muted=True).pack(anchor='w',padx=18,pady=(1,17))
        quick=tk.Frame(self.body,bg=C['bg']);quick.pack(fill='x',pady=(0,15))
        for i,(title,description,page) in enumerate([
           ('＋  Create a new job','Choose analysis and configure parameters','Create job'),
           ('◉  Browse saved seeds','Reuse parameters from any completed analysis','Seed library'),
           ('▤  Explore results','Open plots and browse previous run folders','Results')]):
            pane=card(quick,cursor='hand2',height=86)
            pane.grid(row=i//2,column=i%2,sticky='nsew',padx=(0,12),pady=(0,9))
            pane.grid_propagate(False)
            t=label(pane,title,11,True);t.pack(anchor='w',padx=17,pady=(9,5))
            u=label(pane,description,9,muted=True,wraplength=350,justify='left');u.pack(anchor='w',padx=17)
            for widget in (pane,t,u):widget.bind('<Button-1>',lambda e,p=page:self.navigate(p))
        quick.columnconfigure(0,weight=1)
        quick.columnconfigure(1,weight=1)
        section=card(self.body);section.pack(fill='both',expand=True)
        top=tk.Frame(section,bg=C['card']);top.pack(fill='x',padx=16,pady=(14,8))
        label(top,'Recent jobs',13,True).pack(side='left')
        button(top,'View all results',lambda:self.navigate('Results')).pack(side='right')
        self.add_tree(section,('name','type','status','created'),
                      ('Job','Analysis','Status','Created'),(240,210,135,140),height=5,
                      data=[(j['id'],(j['name'],self.analysis_name(j['analysis_id']),
                                           j['status'].upper(),human_date(j['created_at']))) for j in jobs[:13]],
                      on_open=lambda ident:self.open_result(ident))

    def add_tree(self,parent,cols,titles,widths,height,data=None,on_open=None):
        holder=tk.Frame(parent,bg=C['card']);holder.pack(fill='both',expand=True,padx=15,pady=(0,13))
        tree=ttk.Treeview(holder,columns=cols,show='headings',selectmode='extended',height=height)
        for c,t,w in zip(cols,titles,widths):
            tree.heading(c,text=t);tree.column(c,width=w,stretch=True,minwidth=65)
        scrollbar=ttk.Scrollbar(holder,orient='vertical',command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side='left',fill='both',expand=True);scrollbar.pack(side='right',fill='y')
        for ident,values in data or []:tree.insert('', 'end',iid=ident,values=values)
        if on_open:tree.bind('<Double-1>',lambda e:on_open(tree.focus()))
        return tree

    def analysis_name(self,ident):return self.analyses.get(ident).name if ident in self.analyses else ident

    def show_analyses(self):
        self.selected_analysis=None
        self.heading(self.body,'Choose an analysis',
             'Choose an analysis to explore the model and configure a run.')
        if not self.analyses:
            label(self.body,'No valid analysis manifests found.',12,bg=C['bg']).pack(anchor='w');return
        groups={}
        for spec in self.analyses.values():groups.setdefault(spec.group,[]).append(spec)
        preferred=('Continuation','Time series Analysis','Sensitivity')
        for group in sorted(groups,key=lambda name:(preferred.index(name) if name in preferred else len(preferred),name)):
            specs=groups[group]
            label(self.body,group.upper(),10,True,muted=True,bg=C['bg']).pack(anchor='w',pady=(10,8))
            strip=tk.Frame(self.body,bg=C['bg']);strip.pack(fill='x')
            strip.columnconfigure(0,weight=1,uniform='analysis_cards');strip.columnconfigure(1,weight=1,uniform='analysis_cards')
            for i,spec in enumerate(specs):
                p=card(strip,cursor='hand2',height=px(132))
                p.grid(row=i//2,column=i%2,sticky='nsew',padx=(0,13),pady=(0,10))
                p.pack_propagate(False)
                title=label(p,spec.name+'  ›',12,True);title.pack(anchor='w',padx=17,pady=(15,8))
                desc=label(p,spec.description,9,muted=True,wraplength=320,justify='left',anchor='nw')
                desc.pack(fill='both',expand=True,padx=17,pady=(0,12))
                p.bind('<Configure>',lambda e,text=desc:text.configure(wraplength=max(80,e.width-34)))
                for widget in (p,title,desc):widget.bind('<Button-1>',lambda e,s=spec:self.show_job_form(s))

    def show_job_form(self,spec,seed_id=None):
        self.selected_analysis=spec
        self.page='Create job'
        for item,control in self.nav.items():
            control.configure(bg=C['nav2'] if item=='Create job' else C['nav'],
                              fg='white' if item=='Create job' else C['nav_text'])
        for child in self.body.winfo_children():child.destroy()
        self.page_canvas.yview_moveto(0)
        self.head_title.configure(text='Create job  /  '+spec.name)
        if spec.form == 'sensitivity':
            from app.sensitivity_form import SensitivityForm
            SensitivityForm(self.body,self,spec).pack(fill='both',expand=True)
            return
        if spec.form in ('continuation_1d', 'continuation_2d', 'timeseries'):
            from app.continuation_form import ContinuationForm
            ContinuationForm(self.body, self, spec, seed_id).pack(fill='both', expand=True)
            return
        if spec.id=='sensitivity_demo':
            banner=tk.Frame(self.body,bg=C['blue2']);banner.pack(fill='x',pady=(0,10))
            label(banner,'ONE SENSITIVITY RUN → MANY PARAMETER SAMPLES',11,True,
                  bg=C['blue2']).pack(anchor='w',padx=14,pady=(9,3))
            label(banner,'Sample count specifies the number of saved parameter sets; Random seed reproduces the sampling.',
                  9,muted=True,bg=C['blue2']).pack(anchor='w',padx=14,pady=(0,9))
        guide=tk.Frame(self.body,bg=C['bg']);guide.pack(fill='x',pady=(0,9))
        button(guide,'← Back to analyses',lambda:self.navigate('Create job')).pack(side='left')
        label(guide,'Every declared input can be edited. Selecting a seed only prefills values.',
              9,muted=True,bg=C['bg']).pack(side='left',padx=15)

        # Universal reusable seed selector: works even if the manifest lacks seed_id.
        seedpanel=card(self.body);seedpanel.pack(fill='x',pady=(0,10))
        sr=tk.Frame(seedpanel,bg=C['card']);sr.pack(fill='x',padx=15,pady=(11,7))
        label(sr,'USE SAVED SEED (OPTIONAL)',10,True).pack(side='left')
        button(sr,'Browse seed library',self.open_seed_picker,primary=False).pack(side='right',padx=(8,0))
        button(sr,'Clear seed',self.clear_seed).pack(side='right')
        self.job_seed_var=tk.StringVar(value='')
        self.seed_note=label(seedpanel,'No seed selected. All analysis defaults are editable.',9,muted=True,
                             justify='left',wraplength=1000)
        self.seed_note.pack(anchor='w',padx=15,pady=(0,11))
        self.controls={}
        self.extra_seed_controls={}
        self.seed_original_values={}
        for key,p in spec.parameters.items():
            if p.get('type')=='seed':self.controls[key]=self.job_seed_var

        outer=card(self.body);outer.pack(fill='both',expand=True)
        canvas=tk.Canvas(outer,bg=C['card'],bd=0,highlightthickness=0,height=400)
        vscroll=ttk.Scrollbar(outer,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        form=tk.Frame(canvas,bg=C['card'])
        window=canvas.create_window((0,0),window=form,anchor='nw')
        form.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width))
        groups={}
        for key,p in spec.parameters.items():
            if p.get('type')!='seed':
                groups.setdefault(p.get('category','Parameters'),[]).append((key,p))
        for group,fields in groups.items():
            block=tk.Frame(form,bg=C['card']);block.pack(fill='x',padx=22,pady=(13,4))
            label(block,group.upper(),9,True,muted=True).pack(anchor='w',pady=(5,7))
            for key,p in fields:
                line=tk.Frame(block,bg=C['card']);line.pack(fill='x',pady=4)
                desc=key.replace('_',' ')
                if p.get('unit'):desc+=f"  [{p['unit']}]"
                label(line,desc,10,bold=True,width=31,anchor='w').pack(side='left')
                default=p.get('default','')
                var=tk.StringVar(value=str(default).lower() if isinstance(default,bool) else str(default))
                typ=p.get('type','float')
                if typ=='choice':
                    w=ttk.Combobox(line,textvariable=var,values=[str(x) for x in p.get('options',[])],state='readonly',width=34)
                elif typ=='boolean':
                    w=ttk.Combobox(line,textvariable=var,values=['true','false'],state='readonly',width=34)
                else:w=ttk.Entry(line,textvariable=var,width=37)
                w.pack(side='left',padx=8)
                if p.get('description'):
                    label(line,p['description'],9,muted=True,wraplength=390,justify='left').pack(side='left',padx=8)
                self.controls[key]=var
        # Seed fields absent from this analysis manifest are still visible and editable.
        self.seed_extra_section=tk.Frame(form,bg=C['card'])
        self.seed_extra_section.pack(fill='x',padx=22,pady=(9,12))
        self.extra_seed_caption=label(self.seed_extra_section,
            'ADDITIONAL SEED PARAMETERS · NONE SELECTED',9,True,muted=True)
        self.extra_seed_caption.pack(anchor='w',pady=(5,5))
        self.extra_seed_fields=tk.Frame(self.seed_extra_section,bg=C['card'])
        self.extra_seed_fields.pack(fill='x')

        bottom=card(self.body);bottom.pack(fill='x',pady=(12,0))
        label(bottom,'Run name',11,True).pack(side='left',padx=(17,8),pady=16)
        self.job_name_var=tk.StringVar(value=spec.name+' test')
        ttk.Entry(bottom,textvariable=self.job_name_var,width=28).pack(side='left',padx=(0,14))
        button(bottom,'Create (do not run)',lambda:self.save_job(False)).pack(side='right',padx=(7,15))
        button(bottom,'Create & queue',lambda:self.save_job(True),primary=True).pack(side='right')
        if seed_id:self.apply_seed_id(seed_id)

    def _clear_extra_fields(self):
        if hasattr(self,'extra_seed_fields'):
            for w in self.extra_seed_fields.winfo_children(): w.destroy()
        self.extra_seed_controls={}
        self.seed_original_values={}

    def clear_seed(self):
        self.job_seed_var.set('')
        self._clear_extra_fields()
        self.extra_seed_caption.configure(text='ADDITIONAL SEED PARAMETERS · NONE SELECTED')
        self.seed_note.configure(text='Seed cleared. Previously filled declared fields retain their current editable values.')

    @staticmethod
    def _string_value(value):
        if isinstance(value,bool):return str(value).lower()
        if isinstance(value,(list,dict)):return json.dumps(value,ensure_ascii=False)
        return str(value)

    @staticmethod
    def _parse_seed_override(raw,original):
        if isinstance(original,bool):
            if raw.lower() in ('true','yes','1'):return True
            if raw.lower() in ('false','no','0'):return False
            raise ValueError('Expected true or false')
        if isinstance(original,int):
            v=int(raw)
            if str(v)!=raw.strip() and '+'+str(v)!=raw.strip():
                raise ValueError('Expected integer')
            return v
        if isinstance(original,float):
            v=float(raw)
            if not math.isfinite(v):raise ValueError('Value must be finite')
            return v
        if isinstance(original,(list,dict)):
            decoded=json.loads(raw)
            if type(decoded) is not type(original):raise ValueError('Expected JSON of the original type')
            return decoded
        return raw

    def apply_seed_id(self,seed_id):
        seed=self.repo.get_seed(seed_id)
        if not seed:
            messagebox.showerror('Seed not found',f'{seed_id} is unavailable.',parent=self)
            return
        self.job_seed_var.set(seed_id)
        values=json.loads(seed['values_json'])
        matching=0
        for key,value in values.items():
            if key in self.controls and key!='seed_id':
                self.controls[key].set(self._string_value(value))
                matching+=1
        self._clear_extra_fields()
        extras={key:value for key,value in values.items() if key not in self.selected_analysis.parameters and key!='seed_id'}
        self.seed_original_values=extras
        for key,value in sorted(extras.items()):
            row=tk.Frame(self.extra_seed_fields,bg=C['card']);row.pack(fill='x',pady=3)
            label(row,key.replace('_',' '),10,bold=True,width=31,anchor='w').pack(side='left')
            var=tk.StringVar(value=self._string_value(value))
            if isinstance(value,bool):
                entry=ttk.Combobox(row,textvariable=var,values=['true','false'],state='readonly',width=34)
            else: entry=ttk.Entry(row,textvariable=var,width=37)
            entry.pack(side='left',padx=8)
            label(row,'Inherited from selected seed · editable',9,muted=True).pack(side='left',padx=8)
            self.extra_seed_controls[key]=var
        self.extra_seed_caption.configure(text=f'ADDITIONAL SEED PARAMETERS · {len(extras)} EDITABLE')
        self.seed_note.configure(text=f'Selected {seed_id} from “{seed["source_name"]}”. '
            f'{matching} declared parameter(s) prefilled; {len(extras)} additional fields editable below. '
            'Changes override the selected seed; the original remains saved in the run record.')

    def open_seed_picker(self):
        picker=tk.Toplevel(self)
        picker.title('Select a parameter seed')
        picker.geometry(f'{min(px(1020),self.winfo_screenwidth()-px(50))}x{min(px(710),self.winfo_screenheight()-px(90))}')
        picker.minsize(850,560)
        picker.configure(bg=C['bg'])
        self._set_child_icon(picker)
        picker.transient(self)
        def picked(record):
            self.apply_seed_id(record['seed_id'])
            picker.destroy()
        SeedBrowser(picker,self.repo,on_use=picked).pack(fill='both',expand=True,padx=15,pady=15)
        picker.grab_set()

    def _set_child_icon(self,window):
        try:
            window.iconphoto(True,self._icon_photo)
            if sys.platform=='win32':window.iconbitmap(str(ASSETS/'simp_icon.ico'))
        except (AttributeError,OSError,tk.TclError):pass

    def values_from_controls(self):
        return {k:v.get() for k,v in self.controls.items()}

    def save_job(self,queue=False):
        try:
            extras={}
            for key,var in self.extra_seed_controls.items():
                try: extras[key]=self._parse_seed_override(var.get(),self.seed_original_values[key])
                except (ValueError,TypeError,json.JSONDecodeError) as exc:
                    raise ValueError(f'Inherited parameter {key}: {exc}') from exc
            job=self.repo.create_job(self.job_name_var.get(),self.selected_analysis,
               self.values_from_controls(),seed_id=self.job_seed_var.get(),extra_parameters=extras)
            if queue:self.repo.enqueue([job['id']]);start_worker()
            self.navigate('Job queue')
        except (ValueError,OSError) as exc:
            messagebox.showerror('Cannot create job',str(exc),parent=self)

    def show_queue(self):
        self.heading(self.body,'Sequential job queue',
             'Create first, schedule when ready. Only one computation can run at a time; closing this window does not stop the worker.')
        jobs=self.repo.list_jobs()
        self.queue_trees={}
        groups=[('RUNNING / PAUSED',[j for j in jobs if j['status'] in ('running','paused')],1),
                ('WAITING TO RUN',[j for j in jobs if j['status']=='queued'],2),
                ('CREATED · NOT YET RUN',[j for j in jobs if j['status']=='draft'],2)]
        for title,items,height in groups:
            block=card(self.body);block.pack(fill='both',expand=True,pady=(0,10))
            top=tk.Frame(block,bg=C['card']);top.pack(fill='x',padx=14,pady=(10,5))
            label(top,title,10,True).pack(side='left')
            label(top,f'{len(items)} job(s)',9,muted=True).pack(side='left',padx=10)
            if title.startswith('CREATED'):
                button(top,'Queue selected',self.queue_selected).pack(side='right',padx=(6,0))
                button(top,'Queue all',self.queue_all).pack(side='right')
            if title.startswith('WAITING'):
                button(top,'Remove selected from queue',self.dequeue_selected).pack(side='right')
            tree=self.add_tree(block,('name','type','created','status'),
                               ('Name','Analysis','Created','Status'),(220,210,130,95),height,
                               [(j['id'],(j['name'],self.analysis_name(j['analysis_id']),human_date(j['created_at']),j['status'].upper())) for j in items])
            self.queue_trees[title.split(' · ')[0]]=tree
        row=tk.Frame(self.body,bg=C['bg']);row.pack(fill='x',pady=(3,0))
        button(row,'Delete selected drafts',self.delete_drafts).pack(side='left')
        button(row,'Pause / resume current',self.toggle_pause).pack(side='left',padx=9)
        button(row,'Stop current',lambda:self.job_action('stop')).pack(side='left')
        button(row,'Force terminate',lambda:self.job_action('force')).pack(side='left',padx=8)

    def sync_queue_view(self):
        """Update rows live without destroying the user's current selection."""
        if not hasattr(self,'queue_trees'):return
        jobs=self.repo.list_jobs()
        mapping={
            'RUNNING / PAUSED':('running','paused'),
            'WAITING TO RUN':('queued',),
            'CREATED':('draft',),
        }
        for key,statuses in mapping.items():
            tree=self.queue_trees.get(key)
            if tree is None or not tree.winfo_exists():continue
            items=[j for j in jobs if j['status'] in statuses]
            target={j['id']:j for j in items}
            for ident in tree.get_children():
                if ident not in target:tree.delete(ident)
            for j in items:
                values=(j['name'],self.analysis_name(j['analysis_id']),
                        human_date(j['created_at']),j['status'].upper())
                if tree.exists(j['id']):
                    if tuple(tree.item(j['id'],'values'))!=values:
                        tree.item(j['id'],values=values)
                else:tree.insert('','end',iid=j['id'],values=values)

    def selected_ids(self,prefix):
        for title,tree in self.queue_trees.items():
            if title.startswith(prefix): return list(tree.selection())
        return []

    def queue_selected(self):
        ids=self.selected_ids('CREATED')
        if ids:self.repo.enqueue(ids);start_worker();self.navigate('Job queue')

    def queue_all(self):
        ids=[j['id'] for j in self.repo.list_jobs() if j['status']=='draft']
        if ids:self.repo.enqueue(ids);start_worker();self.navigate('Job queue')

    def dequeue_selected(self):
        ids=self.selected_ids('WAITING')
        for ident in ids:self.repo.update_job(ident,status='draft',phase='')
        if ids:self.navigate('Job queue')

    def delete_drafts(self):
        ids=self.selected_ids('CREATED')
        if ids and messagebox.askyesno('Remove drafts','Remove selected draft entries from the queue? Files remain on disk.',parent=self):
            self.repo.delete_draft(ids);self.navigate('Job queue')

    def running(self):
        return next((j for j in self.repo.list_jobs() if j['status'] in ('running','paused')),None)

    def job_action(self,action):
        j=self.running()
        if not j:return
        if action=='force' and not messagebox.askyesno('Force termination',
            'Immediately terminate the running analysis process and any children? Unsaved progress may be lost.',parent=self):return
        if action=='stop' and not messagebox.askyesno('Stop calculation',
            'Request a cooperative stop? The script stops at its next checkpoint.',parent=self):return
        request(j['id'],action)
        if action=='stop':self.footer_phase.configure(text='Stop requested')

    def toggle_pause(self):
        j=self.running()
        if not j:return
        flag=Path(j['output_dir'])/'pause.request'
        request(j['id'],'resume' if flag.exists() else 'pause')

    def show_seeds(self,source_id=None):
        # The main window already displays 'Seed library' in the header.
        self.seed_browser=SeedBrowser(self.body,self.repo,on_use=self.use_seed_from_library,show_continuation_options=True)
        self.seed_browser.pack(fill='both',expand=True)
        if source_id:self.seed_browser.set_source(source_id)

    def use_seed_from_library(self,record,which='continuation_1d'):
        spec=self.analyses.get(which)
        if spec is None:
            messagebox.showinfo('Analysis unavailable','Analysis is unavailable.',parent=self)
            return
        self.show_job_form(spec,seed_id=record['seed_id'])

    def open_result(self,job_id):
        self.navigate('Results')
        if hasattr(self,'results_tree') and self.results_tree.exists(job_id):
            self.results_tree.selection_set(job_id)
            self.results_tree.focus(job_id)
            self.show_result_detail(job_id)

    def show_results(self):
        self.heading(self.body,'Results browser',
             'Filter previous jobs, inspect saved configurations, and open generated plots directly in the application.')
        filters=tk.Frame(self.body,bg=C['bg']);filters.pack(fill='x',pady=(0,13))
        self.result_search=tk.StringVar();self.result_type=tk.StringVar(value='All types')
        self.result_status=tk.StringVar(value='All statuses')
        ttk.Entry(filters,textvariable=self.result_search,width=24).pack(side='left',padx=(0,8))
        ttk.Combobox(filters,textvariable=self.result_type,state='readonly',width=24,
                     values=['All types']+[x.name for x in self.analyses.values()]).pack(side='left',padx=8)
        ttk.Combobox(filters,textvariable=self.result_status,state='readonly',width=17,
                     values=['All statuses','draft','queued','running','paused','completed','failed','stopped','interrupted']).pack(side='left',padx=8)
        button(filters,'Apply filters',self.load_results).pack(side='left',padx=8)
        main=tk.PanedWindow(self.body,orient='horizontal',bg=C['bg'],sashwidth=7,bd=0)
        main.pack(fill='both',expand=True)
        listpanel=card(main);detail=card(main)
        main.add(listpanel,minsize=390,width=480);main.add(detail,minsize=370)
        self.results_tree=self.add_tree(listpanel,('name','type','status','date'),
                ('Run','Analysis','Status','Created'),(180,130,90,110),14)
        self.results_tree.bind('<<TreeviewSelect>>',lambda e:self.show_result_detail(self.results_tree.focus()))
        self.results_detail=detail
        self.load_results()
        self.clear_detail()

    def load_results(self):
        if not hasattr(self,'results_tree') or not self.results_tree.winfo_exists():return
        search=self.result_search.get().strip().lower()
        typ=self.result_type.get();status=self.result_status.get()
        previously=self.results_tree.focus()
        for row in self.results_tree.get_children():self.results_tree.delete(row)
        for j in self.repo.list_jobs():
            name=self.analysis_name(j['analysis_id'])
            if search and search not in (j['name']+' '+j['id']).lower():continue
            if typ!='All types' and typ!=name:continue
            if status!='All statuses' and status!=j['status']:continue
            self.results_tree.insert('','end',iid=j['id'],values=(j['name'],name,j['status'],human_date(j['created_at'])))
        if previously and self.results_tree.exists(previously):
            self.results_tree.selection_set(previously)
            self.results_tree.focus(previously)

    def clear_detail(self):
        for w in self.results_detail.winfo_children():w.destroy()
        label(self.results_detail,'Select a run to inspect its files',12,True).pack(anchor='center',pady=35)

    def show_result_detail(self,ident):
        if not ident:return
        job=self.repo.get_job(ident)
        if not job:return
        self.current_result_id=ident
        for w in self.results_detail.winfo_children():w.destroy()
        outer=tk.Frame(self.results_detail,bg=C['card']);outer.pack(fill='both',expand=True,padx=16,pady=14)
        label(outer,job['name'],14,True,wraplength=450,justify='left').pack(anchor='w',pady=(0,5))
        label(outer,f"{self.analysis_name(job['analysis_id'])}   •   {job['status'].upper()}   •   {human_date(job['created_at'])}",
              9,muted=True).pack(anchor='w',pady=(0,12))
        tools=tk.Frame(outer,bg=C['card']);tools.pack(fill='x',pady=(0,10))
        button(tools,'Open folder',lambda:self.open_folder(Path(job['output_dir']))).pack(side='left')
        button(tools,'View config',lambda:self.open_config(job)).pack(side='left',padx=8)
        delete = button(tools,'Delete result',lambda:self.delete_result(ident))
        delete.pack(side='left')
        if job['status'] not in ('completed','failed','stopped','interrupted'):
            delete.configure(state='disabled')
        count=self.repo.seed_count(ident)
        if count:
            seedrow=tk.Frame(outer,bg=C['blue2']);seedrow.pack(fill='x',pady=(1,8))
            label(seedrow,f'{count:,} saved parameter seeds',10,True,bg=C['blue2']).pack(side='left',padx=10,pady=9)
            button(seedrow,'Browse seeds',lambda:self.open_seeds_from_result(ident)).pack(side='right',padx=9,pady=4)
        folder=Path(job['output_dir'])
        figs=sorted((folder/'figures').glob('*.png')) if (folder/'figures').is_dir() else []
        if figs:
            label(outer,'FIGURES',9,True,muted=True).pack(anchor='w',pady=(4,5))
            names=[p.name for p in figs]
            self.plot_select=tk.StringVar(value=names[0])
            c=ttk.Combobox(outer,values=names,textvariable=self.plot_select,state='readonly',width=40)
            c.pack(anchor='w',fill='x',pady=(0,8))
            c.bind('<<ComboboxSelected>>',lambda e:self.render_plot(folder/'figures'/self.plot_select.get()))
            self.plot_box=tk.Label(outer,bg=C['card'],text='Loading image...',fg=C['muted'])
            self.plot_box.pack(fill='both',expand=True)
            self.render_plot(figs[0])
        else:
            label(outer,'No figures yet. Completed jobs save PNG plots here.',
                  10,muted=True,wraplength=410,justify='left').pack(anchor='w',pady=(20,10))
        files=sorted(p.name for p in folder.glob('*') if p.is_file() and p.name.endswith(('.csv','.log','.json')))
        label(outer,'OUTPUT FILES',9,True,muted=True).pack(anchor='w',pady=(13,5))
        label(outer,'    ·    '.join(files)[:270] or 'No files',9,muted=True,wraplength=475,justify='left').pack(anchor='w')

    def delete_result(self, ident):
        job = self.repo.get_job(ident)
        if not job:return
        count = self.repo.seed_count(ident)
        if not messagebox.askyesno('Delete result permanently?',
                f'Delete “{job["name"]}”?\n\nThis removes its output folder, plots, logs, '
                f'job record and {count} saved seeds. Other jobs and their copied seed values remain.\n\n'
                'This cannot be undone.',parent=self):return
        try:self.repo.delete_result(ident)
        except (ValueError,OSError) as exc:
            messagebox.showerror('Cannot delete result',str(exc),parent=self);return
        self.current_result_id=None
        self.img=None
        self.load_results()
        self.clear_detail()

    def render_plot(self,path):
        """Downsample full-resolution scientific figures with anti-aliasing.

        Previous versions used PhotoImage.subsample(), which discards rows and
        columns and makes diagonals/text visibly pixelated at high DPI.
        """
        try:
            max_width = px(515)
            max_height = px(330)
            if hasattr(self, 'results_detail') and self.results_detail.winfo_width()>80:
                max_width = max(px(250), min(max_width, self.results_detail.winfo_width()-px(68)))
            with Image.open(path) as source:
                img = source.convert('RGBA')
                img.thumbnail((max_width,max_height),Image.Resampling.LANCZOS)
                self.img = ImageTk.PhotoImage(img,master=self)
            self.plot_box.configure(image=self.img,text='')
        except (OSError,ValueError,tk.TclError):
            self.plot_box.configure(image='',text='Cannot read image')

    def open_seeds_from_result(self,job_id):
        self.navigate('Seed library')
        self.seed_browser.set_source(job_id)

    def open_config(self,job):
        win=tk.Toplevel(self);win.title('Run configuration — '+job['name']);win.geometry(f'{min(px(650),self.winfo_screenwidth()-px(50))}x{min(px(520),self.winfo_screenheight()-px(90))}');self._set_child_icon(win)
        raw=Path(job['output_dir'])/'config.json'
        area=tk.Text(win,font=('Consolas',10),wrap='none',padx=12,pady=12)
        area.pack(fill='both',expand=True)
        try:area.insert('1.0',raw.read_text(encoding='utf-8'))
        except OSError as exc:area.insert('1.0',str(exc))
        area.configure(state='disabled')

    def open_folder(self,path):
        try:
            if sys.platform=='win32':os.startfile(str(path))
            elif sys.platform=='darwin':subprocess.Popen(['open',str(path)])
            else:subprocess.Popen(['xdg-open',str(path)])
        except (OSError,AttributeError) as exc:messagebox.showerror('Cannot open folder',str(exc),parent=self)

    def show_settings(self):
        self.heading(self.body,'Environment & project settings',
                     'The GUI is read-only with respect to scripts. Change manifests or analysis code using your regular editor.')
        p=card(self.body);p.pack(fill='x',pady=(2,15))
        for title,value in [('Project source folder',str(ROOT)),('Results folder',str(RESULTS)),
                            ('Job database',str(STATE/'jobs.sqlite')),('Python executable',sys.executable),
                            ('Parallel computation','Disabled: maximum one running analysis'),
                            ('Display scale',f'{self.dpi_scale*100:.0f}% (DPI-aware GUI; text is rendered natively)')]:
            block=tk.Frame(p,bg=C['card']);block.pack(fill='x',padx=20,pady=11)
            label(block,title,10,True,width=24,anchor='w').pack(side='left')
            label(block,value,10,muted=True,wraplength=630,justify='left').pack(side='left')
        button(self.body,'Open results directory',lambda:self.open_folder(RESULTS)).pack(anchor='w')
        label(self.body,'Scripts are discovered again whenever you refresh or open Create job.\n'
                        'The queue uses a separate process, with cooperative pause/stop checkpoints.\n'
                        'A non-cooperative scientific script cannot be paused safely without adapting it.',
              10,muted=True,bg=C['bg'],justify='left').pack(anchor='w',pady=16)

    def heartbeat(self):
        try:
            j=self.running()
            if j:
                self.footer_job.configure(text=j['name'][:32]);self.footer_progress['value']=j.get('progress') or 0
                paused=(Path(j['output_dir'])/'pause.request').exists()
                self.footer_pause.configure(text='Resume' if paused else 'Pause',state='normal')
                self.footer_stop.configure(state='normal')
                self.footer_phase.configure(text=f"{j['progress']:.0f}% · {j['phase'][:30] or 'Running'}")
                self.footer_eta.configure(text=self.repo.runtime_label(j,paused))
            else:
                self.footer_job.configure(text='No job currently running')
                self.footer_phase.configure(text='Worker online' if worker_is_alive() else 'Worker offline')
                self.footer_progress['value']=0
                waiting=next((job for job in self.repo.list_jobs() if job['status']=='queued'),None)
                if waiting:
                    self.footer_eta.configure(text='Next run: '+self.repo.runtime_label(waiting))
                else:self.footer_eta.configure(text='')
                self.footer_pause.configure(state='disabled');self.footer_stop.configure(state='disabled')
        except (OSError,ValueError):pass
        if self.page=='Job queue':
            try:self.sync_queue_view()
            except (OSError,ValueError):pass
        self.after(900,self.heartbeat)


class SeedBrowser(tk.Frame):
    """Reusable, SQL-paginated seed picker; never puts every seed in a Tk widget."""
    PAGE_SIZE=80

    def __init__(self,parent,repo,on_use,show_continuation_options=False):
        super().__init__(parent,bg=C['bg'])
        self.repo=repo
        self.on_use=on_use
        self.show_continuation_options=show_continuation_options
        self.offset=0
        self.rows={}
        self.source_labels={'All source analyses':None}
        self.source_var=tk.StringVar(value='All source analyses')
        self.query=tk.StringVar()
        self.bistable=tk.StringVar(value='Any outcome')
        self.seed_kind=tk.StringVar(value='All seeds')
        self.counter=tk.StringVar(value='No seeds found')
        self.build()
        self.reload()

    def build(self):
        panel=card(self);panel.pack(fill='x',pady=(0,9))
        filters=tk.Frame(panel,bg=C['card']);filters.pack(fill='x',padx=12,pady=12)
        label(filters,'Source analysis / job',9,True).grid(row=0,column=0,sticky='w',padx=(0,8))
        self.sources=self.repo.list_seed_sources()
        for x in self.sources:
            name=f"{x['source_analysis_id'] or 'Archived'} · {x['source_name']}  [{x['source_job_id'][:8]}]  ·  {x['seed_count']:,}"
            self.source_labels[name]=x['source_job_id']
        self.source_combo=ttk.Combobox(filters,textvariable=self.source_var,state='readonly',
                           values=list(self.source_labels),width=53)
        self.source_combo.grid(row=1,column=0,sticky='ew',padx=(0,12))
        self.source_combo.bind('<<ComboboxSelected>>',lambda e:self.search())
        label(filters,'Find Seed-ID',9,True).grid(row=0,column=1,sticky='w',padx=(0,8))
        entry=ttk.Entry(filters,textvariable=self.query,width=25)
        entry.grid(row=1,column=1,sticky='ew',padx=(0,12))
        entry.bind('<Return>',lambda e:self.search())
        label(filters,'Seed origin',9,True).grid(row=0,column=2,sticky='w')
        origin=ttk.Combobox(filters,textvariable=self.seed_kind,state='readonly',
                           values=['All seeds','Generated samples','Completed runs'],width=19)
        origin.grid(row=1,column=2,sticky='w',padx=(0,10))
        origin.bind('<<ComboboxSelected>>',lambda e:self.search())
        label(filters,'Bistability',9,True).grid(row=0,column=3,sticky='w')
        cb=ttk.Combobox(filters,textvariable=self.bistable,state='readonly',
                       values=['Any outcome','Bistable','Not bistable'],width=17)
        cb.grid(row=1,column=3,sticky='w',padx=(0,10))
        cb.bind('<<ComboboxSelected>>',lambda e:self.search())
        button(filters,'Search',self.search,primary=True).grid(row=1,column=4,sticky='w')
        filters.columnconfigure(0,weight=2)
        filters.columnconfigure(1,weight=1)
        actions=tk.Frame(self,bg=C['bg']);actions.pack(side='bottom',fill='x',pady=(2,0))
        if self.show_continuation_options:
            button(actions,'Use in 2D continuation',
                   lambda:self.select('continuation_2d')).pack(side='right',padx=(7,0))
            button(actions,'Use in 1D continuation',
                   lambda:self.select('continuation_1d'),primary=True).pack(side='right')
            button(actions,'Use in time series',lambda:self.select('timeseries')).pack(side='right',padx=(7,0))
        else:
            button(actions,'Use selected Seed-ID',self.select,primary=True).pack(side='right')
        label(actions,'Samples and snapshots from all analyses are reusable. Select then edit parameters.',
              9,muted=True,bg=C['bg']).pack(side='left')

        detail=card(self);detail.pack(side='bottom',fill='x',pady=(0,6))
        label(detail,'SELECTED SEED · PARAMETERS & RESULTS',9,True,muted=True).pack(anchor='w',padx=14,pady=(12,5))
        self.preview_text=tk.Text(detail,height=3,wrap='word',font=('Consolas',9),
                                 bg=C['card'],fg=C['text'],bd=0,highlightthickness=0)
        self.preview_text.pack(fill='x',padx=14,pady=(0,10))
        self._set_preview('Select a seed to inspect the saved parameters.')
        navigation=tk.Frame(self,bg=C['bg']);navigation.pack(side='bottom',fill='x',pady=6)
        label(navigation,'',9,muted=True,bg=C['bg'],textvariable=self.counter).pack(side='left')
        self.next_btn=button(navigation,'Next page  ›',self.next_page)
        self.next_btn.pack(side='right',padx=(7,0))
        self.prev_btn=button(navigation,'‹  Previous',self.prev_page)
        self.prev_btn.pack(side='right')
        tableframe=card(self);tableframe.pack(fill='both',expand=True)
        self.table=ttk.Treeview(tableframe,columns=('id','source','bistability','delta'),
                               show='headings',height=9,selectmode='browse')
        for col,title,width in [('id','Seed-ID',205),('source','Source analysis / run',250),
                                ('bistability','Bistable?',95),('delta','ΔS (demo)',115)]:
            self.table.heading(col,text=title)
            self.table.column(col,width=width,minwidth=80,stretch=True)
        scroll=ttk.Scrollbar(tableframe,orient='vertical',command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side='left',fill='both',expand=True,padx=(12,0),pady=12)
        scroll.pack(side='right',fill='y',padx=(0,9),pady=12)
        self.table.bind('<<TreeviewSelect>>',lambda e:self.preview())
        self.table.bind('<Double-1>',lambda e:self.select())
    def set_source(self,source_job_id):
        for key,value in self.source_labels.items():
            if value==source_job_id:
                self.source_var.set(key)
                break
        self.search()

    def search(self):
        self.offset=0
        self.reload()

    def _outcome(self):
        return {'Any outcome':'all','Bistable':'yes','Not bistable':'no'}.get(self.bistable.get(),'all')

    def reload(self):
        self.total,items=self.repo.find_seeds(
            source_job_id=self.source_labels.get(self.source_var.get()),
            query=self.query.get(),bistability=self._outcome(),
            seed_kind={'Generated samples':'generated','Completed runs':'run'}.get(self.seed_kind.get(),'all'),
            limit=self.PAGE_SIZE,offset=self.offset)
        self.rows={row['seed_id']:row for row in items}
        for key in self.table.get_children():self.table.delete(key)
        for row in items:
            b={1:'Yes',0:'No',None:'—'}.get(row.get('bistable'),'—')
            delta='—' if row.get('delta_s') is None else f"{row['delta_s']:.3g}"
            self.table.insert('','end',iid=row['seed_id'],values=(
                row['seed_id'],row['source_name'],b,delta))
        start=self.offset+1 if items else 0
        self.counter.set(f'{self.total:,} matching seeds · Showing {start:,}–{self.offset+len(items):,} '
                         f'(page size {self.PAGE_SIZE})')
        self.prev_btn.configure(state='normal' if self.offset>0 else 'disabled')
        self.next_btn.configure(state='normal' if self.offset+len(items)<self.total else 'disabled')
        self._set_preview('Select a seed to inspect the saved parameters.')

    def _set_preview(self,text):
        self.preview_text.configure(state='normal')
        self.preview_text.delete('1.0','end')
        self.preview_text.insert('1.0',text)
        self.preview_text.configure(state='disabled')

    def preview(self):
        selected=self.table.selection()
        if not selected:return
        row=self.rows.get(selected[0])
        if not row:return
        pars=json.loads(row['values_json'])
        metrics=json.loads(row.get('metrics_json') or '{}')
        params=', '.join(f'{k}={v:g}' if isinstance(v,(int,float)) else f'{k}={v}'
                         for k,v in pars.items())
        outputs=', '.join(f'{k}={v}' for k,v in metrics.items())
        self._set_preview(f"Seed: {row['seed_id']}  ·  Source run: {row['source_name']}\n"
                          f"Parameters: {params or '(none)'}\nResults: {outputs or 'Not recorded'}")

    def select(self,which=None):
        picked=self.table.selection()
        if not picked:
            messagebox.showinfo('Select a seed','Select one Seed-ID in the list first.',parent=self)
            return
        if self.show_continuation_options and which is not None:
            self.on_use(self.rows[picked[0]],which)
        else:self.on_use(self.rows[picked[0]])

    def prev_page(self):
        self.offset=max(0,self.offset-self.PAGE_SIZE)
        self.reload()

    def next_page(self):
        if self.offset+self.PAGE_SIZE<self.total:
            self.offset+=self.PAGE_SIZE
            self.reload()
