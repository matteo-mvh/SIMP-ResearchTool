"""Utilities exclusively for fake example analyses. Replace with SIMP later."""
from __future__ import annotations
import csv
import math
from pathlib import Path

def csv_write(path, fields, data):
    with Path(path).open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

def linspace(a,b,n):
    return [a+(b-a)*i/(n-1) for i in range(n)]

def plt_module():
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    return plt

def finish_plot(fig,ctx,name):
    from pathlib import Path
    folder=ctx.folder/'figures'
    folder.mkdir(exist_ok=True)
    # Render sharp scientific text/lines, even on high-DPI monitors.
    fig.savefig(folder/name,dpi=200,bbox_inches='tight')
    plt=plt_module()
    plt.close(fig)
