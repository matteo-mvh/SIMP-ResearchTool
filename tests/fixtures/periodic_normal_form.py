"""Isolated native regression: unstable cycles through a fold to stable cycles."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from model.shared_eqn import Parameters
from model.continuation_config import Axis
from model.periodic_config import periodic_options,node_count
from model.periodic_native import periodic_problem,install_guess,inspect_orbit,solve_options


def growth(M,S,P,p,exp):
    x,y=M-3,S-3
    r2=x*x+y*y
    a=p.N_wc-100.5+r2-r2*r2
    return ((a*x-y)/M,(x+a*y)/S,(1-P)/P)


def main():
    options=periodic_options(dict(periodic_mode='collocation',periodic_nodes=60))
    axis=Axis('N_wc',100,101,100);base=Parameters(H=1,N_wc=100.4)
    radius=np.sqrt((1-np.sqrt(.6))/2)
    theta=2*np.pi*np.arange(node_count(options))/node_count(options)
    logs=np.log(np.column_stack((3+radius*np.cos(theta),3+radius*np.sin(theta),np.ones(len(theta)))))
    records=[]
    with patch('model.pyoomph_continuation.growth_expressions',growth),\
         patch('model.periodic_math.growth_expressions',growth),\
         periodic_problem(base,[axis],[.4],np.exp(logs[0])) as problem:
        orbit=install_guess(problem,logs,2*np.pi,options)
        records.append(inspect_orbit(problem,orbit,base,[axis],options))
        ds=-.01
        for _ in range(100):
            ds=problem.arclength_continuation(problem.controls[0],ds,max_ds=.03,min_ds=1e-7,**solve_options(options))
            row=inspect_orbit(problem,orbit,base,[axis],options)
            records.append(row)
            if row['stability']=='stable' and row['u']>.4:break
    Path(sys.argv[1]).write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
