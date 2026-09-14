
"""Small, non-invasive v60m UI policy patch."""
from __future__ import annotations
import functools
import inspect


def _children(w):
    out=[]
    try: todo=list(w.winfo_children())
    except Exception: return out
    while todo:
        x=todo.pop(0); out.append(x)
        try: todo.extend(x.winfo_children())
        except Exception: pass
    return out


def _text(w):
    try: return str(w.cget('text') or '')
    except Exception: return ''


def bind_energy_master(editor):
    widgets=_children(editor)
    master=next((w for w in widgets if 'u.recs energy' in _text(w).lower() and 'eval-run' in _text(w).lower()),None)
    native=next((w for w in widgets if _text(w).strip().lower()=='native energy'),None)
    if master is None or native is None or getattr(master,'_v60m_bound',False): return
    try:
        import tkinter as tk
        mv=str(master.cget('variable')); nv=str(native.cget('variable'))
        mvar=tk.BooleanVar(master=editor,name=mv); nvar=tk.BooleanVar(master=editor,name=nv)
        old=str(master.cget('command') or '')
        def apply(call_old=True):
            if call_old and old:
                try: editor.tk.call(old)
                except Exception: pass
            enabled=bool(mvar.get())
            if not enabled: nvar.set(False)
            try: native.configure(state='normal' if enabled else 'disabled')
            except Exception: pass
        master.configure(command=apply); master._v60m_bound=True; apply(False)
        parent=native.master
        try:
            from tkinter import ttk
            label=ttk.Label(parent,text='Master switch: when u.RECS Energy is off, Native Energy is forced off.')
            cols=max(1,parent.grid_size()[0]); row=parent.grid_size()[1]
            label.grid(row=row,column=0,columnspan=cols,sticky='w',padx=4,pady=(4,0))
        except Exception: pass
    except Exception: return


def add_integrity_hint(editor):
    if getattr(editor,'_v60m_integrity_hint',False): return
    frames=[w for w in _children(editor) if 'workflow execution' in _text(w).lower()]
    if not frames: return
    try:
        from tkinter import ttk
        f=frames[0]; row=f.grid_size()[1]; cols=max(1,f.grid_size()[0])
        ttk.Label(f,text='Integrity: cached SHA-256 in development; strict SHA-256 is enforced only for final campaigns.').grid(row=row,column=0,columnspan=cols,sticky='w',padx=4,pady=(3,0))
        editor._v60m_integrity_hint=True
    except Exception: pass


def install_profile_editor_class_patches(globs):
    mod=globs.get('__name__')
    for name,cls in list(globs.items()):
        if not inspect.isclass(cls) or cls.__module__!=mod or getattr(cls,'_v60m_ui_class',False): continue
        orig=getattr(cls,'__init__',None)
        if not callable(orig): continue
        @functools.wraps(orig)
        def init(self,*a,__orig=orig,**kw):
            __orig(self,*a,**kw)
            try: bind_energy_master(self); add_integrity_hint(self)
            except Exception: pass
        cls.__init__=init; cls._v60m_ui_class=True
