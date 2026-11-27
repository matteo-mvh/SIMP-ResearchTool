"""Native pyoomph continuation driven by the shared SIMP equations.

The scalar solver discovers starts only. Branches and codimension-one event
curves are corrected and traced by pyoomph; no home-grown arclength solver.
Resident log-biomasses retain invariant faces and positivity. Full stability
always includes transverse invasion modes from shared_eqn.jacobian.
"""
from contextlib import contextmanager, ExitStack
from collections import OrderedDict
from time import perf_counter
from dataclasses import asdict, replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linear_sum_assignment, least_squares, brentq
from pyoomph import Problem, ODEEquations, InitialCondition
from pyoomph.expressions import exp, partial_t, var_and_test
from model.shared_eqn import (Parameters, growth_expressions, per_capita, rhs,
                              resources, jacobian, classify_stability, solve_equilibria)
from model.native_compiler import QuietTCCCompiler


class ResidentEquations(ODEEquations):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def define_fields(self):
        names = list(self.owner.active)
        if self.owner.kind == 'invasion': names.append('event_control')
        self.define_ode_variable(*names)

    def define_residuals(self):
        p = self.owner
        fields = {c: var_and_test(c) for c in p.active}
        control, test = var_and_test('event_control') if p.kind == 'invasion' else (None, None)
        parameters = p.symbolic_parameters(control)
        state = [exp(fields[c][0]) if c in fields else 0 for c in 'MSP']
        growth = growth_expressions(*state, parameters, exp)
        for c in p.active:
            u, v = fields[c]
            residual = growth['MSP'.index(c)]
            self.add_residual((residual if p.kind == 'invasion' else partial_t(u)-residual)*v)
        if p.kind == 'invasion':
            self.add_residual(growth['MSP'.index(p.invader)]*test)


class NativeProblem(Problem):
    def __init__(self, base, axes, coordinates, state, face, kind='', invader='', solved_axis=0):
        super().__init__()
        self.base, self.axes = base, axes
        self.ignore_command_line = True
        self.active = tuple(c for c in 'MSP' if c in face)
        self.face, self.kind, self.invader, self.solved_axis = face, kind, invader, solved_axis
        self.controls = [self.define_global_parameter(**{f'control_{i}': float(t)})
                         for i, t in enumerate(coordinates)]
        self.initial_state = tuple(state)
        self.initial_coordinates = tuple(coordinates)
        # Controls are normalized to [0,1]; keep log biomass and the
        # algebraic invasion control on comparable scales.
        self.set_scaling(**{c: 5. for c in self.active},
                         **({'event_control': 1.} if kind=='invasion' else {}))
        self.set_eigensolver('scipy')
        self.set_c_compiler(QuietTCCCompiler())
        self.set_arclength_inner_product('ndof')
        self.warn_about_unused_global_parameters = False
        self.write_states = False
        self.logfile_name = None
        self.quiet()

    def symbolic_parameters(self, event_control=None):
        values = asdict(self.base)
        for i, axis in enumerate(self.axes):
            t = event_control if i == self.solved_axis and event_control is not None else self.controls[i]+0
            values[axis.name] = axis.physical(t)
        return SimpleNamespace(**values)

    def define_problem(self):
        initial = {c: float(np.log(max(self.initial_state['MSP'.index(c)], 1e-12))) for c in self.active}
        if self.kind == 'invasion': initial['event_control'] = self.initial_coordinates[self.solved_axis]
        self.add_equations((ResidentEquations(self)+InitialCondition(**initial)) @ 'ecosystem')

    def vector(self):
        ode = self.get_ode('ecosystem')
        state = np.array([np.exp(float(ode.get_value(c, as_float=True))) if c in self.active else 0.
                          for c in 'MSP'])
        coordinates = np.array([float(c.value) for c in self.controls])
        if self.kind == 'invasion':
            coordinates[self.solved_axis] = float(ode.get_value('event_control', as_float=True))
        return state, coordinates

    def reset_seed(self, coordinates, state):
        """Reset a compiled problem before a completely independent solve."""
        if self.kind in ('fold', 'hopf'):
            self.deactivate_bifurcation_tracking()
        self.reset_arc_length_parameters()
        for control, value in zip(self.controls, coordinates):
            control.value = float(value)
        values = {c: float(np.log(max(state['MSP'.index(c)], 1e-12))) for c in self.active}
        if self.kind == 'invasion': values['event_control'] = float(coordinates[self.solved_axis])
        self.get_ode('ecosystem').set_value(**values)
        self.invalidate_eigendata()
        self.correct(initialise=False)

    def correct(self, initialise=True):
        if initialise and self.kind in ('fold', 'hopf'):
            self.setup_for_stability_analysis(analytic_hessian=True)
        if initialise: self.initialise()
        if self.kind in ('fold', 'hopf'):
            eig, vectors = self.solve_eigenproblem(len(self.active))
            if self.kind == 'hopf':
                options = [i for i, v in enumerate(eig) if v.imag > 1e-6]
                if not options: raise ValueError('No resident complex pair at Hopf seed')
                index = min(options, key=lambda i: abs(eig[i].real))
                self.activate_bifurcation_tracking(self.controls[self.solved_axis], 'hopf',
                    eigenvector=vectors[index], omega=float(eig[index].imag))
            else:
                index = int(np.argmin(abs(eig)))
                self.activate_bifurcation_tracking(self.controls[self.solved_axis], 'fold',
                                                   eigenvector=vectors[index].real)
        self.solve(newton_solver_tolerance=1e-11)

# A single-entry cache per depth/process: never retain multiple native Problems.
# Reuse is opt-in until tested with the user's Windows pyoomph build.
_session = None
_geometry = OrderedDict()

class NativeSession:
    def __init__(self, reuse=False, step_factor=1.):
        self.reuse, self.step_factor = reuse, step_factor
        self.stack = ExitStack(); self.key = self.problem = None
        self.stats = dict(native_initialisations=0, native_reuses=0, reuse_failures=0,
                          native_setup_seconds=0., native_total_seconds=0.,
                          prepared_seeds=0, covered_seeds=0, native_corrections=0,
                          adaptive_steps=0, adaptive_fallbacks=0)
    def clear(self):
        self.key = self.problem = None
        self.stack.close(); self.stack = ExitStack()

@contextmanager
def native_session(reuse=False, step_factor=1.):
    global _session
    if _session is not None: raise RuntimeError('Nested native sessions are not supported')
    _session = NativeSession(reuse,step_factor)
    try: yield _session.stats
    finally:
        try: _session.clear()
        finally: _session=None; _geometry.clear()

@contextmanager
def native_problem(base, axes, coordinates, state, face, kind='', invader='', solved_axis=0):
    session=_session
    # base and axes are immutable; all constants baked into generated C are
    # included in the key, including H and the selected control chart.
    key=(base,tuple(axes),face,kind,invader,solved_axis)
    started=perf_counter()
    if session is None or not session.reuse:
        with TemporaryDirectory(prefix='simp_native_') as folder:
            with NativeProblem(base,axes,coordinates,state,face,kind,invader,solved_axis) as problem:
                problem.set_output_directory(folder)
                if session: session.stats['native_initialisations']+=1
                problem.correct()
                if session: session.stats['native_setup_seconds']+=perf_counter()-started
                try: yield problem
                finally:
                    if session: session.stats['native_total_seconds']+=perf_counter()-started
        return
    try:
        if session.key==key:
            try:
                session.problem.reset_seed(coordinates,state)
                session.stats['native_reuses']+=1
            except Exception:
                # A failed reset must never contaminate the next solve.
                session.stats['reuse_failures']+=1
                session.clear()
                session.reuse=False
        if session.problem is None or session.key!=key:
            session.clear()
            folder=session.stack.enter_context(TemporaryDirectory(prefix='simp_native_'))
            problem=session.stack.enter_context(NativeProblem(base,axes,coordinates,state,face,kind,invader,solved_axis))
            problem.set_output_directory(folder)
            session.stats['native_initialisations']+=1
            problem.correct()
            session.problem,session.key=problem,key
        session.stats['native_setup_seconds']+=perf_counter()-started
        yield session.problem
    except BaseException:
        session.clear()
        raise
    finally:
        session.stats['native_total_seconds']+=perf_counter()-started
        if not session.reuse:session.clear()


def parameters_at(base, axes, coordinates):
    return replace(base, **{axis.name: float(axis.physical(t)) for axis, t in zip(axes, coordinates)})


def point_record(base, axes, coordinates, state, face, *, equilibrium=None, event=None):
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(coordinates)):
        raise ValueError('Nonfinite continuation point')
    p = parameters_at(base, axes, coordinates)
    present = [i for i, c in enumerate('MSP') if c in face]
    if present and min(state[present]) < 1e-8:
        raise ValueError('Resident biomass reached a presence boundary')
    _, N = resources(state, p)
    growth = per_capita(state, p)
    residual = float(np.max(np.abs(state*growth)))
    if N < -1e-8 or residual > 1e-7 or (present and max(abs(growth[present])) > 1e-8):
        raise ValueError('Point failed physical-domain/stationarity checks')
    matrix=jacobian(state,p) if event is not None and event[0]!='invasion' else None
    stability = equilibrium.stability if equilibrium is not None else classify_stability(state,p,matrix=matrix)
    record = dict(community=face, H=p.H, M=float(state[0]), S=float(state[1]),
                  P=float(state[2]), N=float(N), P_area=float(state[2]*p.H), N_area=float(N*p.H),
                  stability=stability.classification, max_real=stability.spectral_abscissa,
                  residual=residual)
    for i, (axis, t) in enumerate(zip(axes, coordinates)):
        record['xy'[i]] = float(axis.physical(t)); record['uv'[i]] = float(t)
    for i, eigenvalue in enumerate(stability.eigenvalues, 1):
        record[f'eig{i}_real'], record[f'eig{i}_imag'] = eigenvalue.real, eigenvalue.imag
    for i,c in enumerate('MSP'):
        record['invasion_'+c] = float(growth[i]) if c not in face else None
    if event is not None and not validate_event(record,base,axes,*event,matrix=matrix):
        raise ValueError('Unverified event residual')
    return record


def discover(base, axes, coordinates, *, grid_points=257):
    p = parameters_at(base, axes, coordinates)
    return [point_record(base, axes, coordinates, np.array(e.biomass), e.community, equilibrium=e)
            for e in solve_equilibria(p,grid_points=grid_points)]


def discovery_coordinates(count):
    """Uniform coverage plus edge-clustered samples for thin boundary branches."""
    t=np.linspace(0,1,count)
    return np.unique(np.round(np.r_[t,(1-np.cos(np.pi*t))/2],14))


def state_of(row):
    return np.array([row[c] for c in 'MSP'])


def match_roots(before, after):
    pairs = []
    for face in sorted({r['community'] for r in before+after}):
        aa = [r for r in before if r['community'] == face]
        bb = [r for r in after if r['community'] == face]
        if not aa or not bb: continue
        distance = np.max(abs(np.log1p([state_of(a) for a in aa])[:,None,:]
                             -np.log1p([state_of(b) for b in bb])[None,:,:]), axis=2)
        ii, jj = linear_sum_assignment(distance)
        pairs.extend((aa[i], bb[j]) for i,j in zip(ii,jj)
                     if len(aa)==len(bb)==1 or distance[i,j] < 1.)
    return pairs


def _coverage_vector(row, dimensions):
    return np.array([*(row['uv'[i]] for i in range(dimensions)),
                     *(np.log1p(state_of(row))/5)],float)


def _segments(path, dimensions):
    key=(id(path),dimensions,len(path))
    item=_geometry.get(key)
    if item is not None and item[0] is path:
        _geometry.move_to_end(key);return item[1:]
    points=np.array([_coverage_vector(r,dimensions) for r in path])
    start,delta=points[:-1],np.diff(points,axis=0)
    denom=np.maximum(np.einsum('ij,ij->i',delta,delta),1e-30)
    _geometry[key]=(path,points,start,delta,denom)
    if len(_geometry)>128:_geometry.popitem(last=False)
    return points,start,delta,denom


def covered(row, paths, axes, tolerance=.008, *, interior=False):
    """Parameter/log-biomass distance; do not suppress retries near open ends."""
    target=_coverage_vector(row,len(axes))
    for path in paths:
        if len(path)<2 or path[0]['community']!=row['community']:continue
        points,start,delta,denom=_segments(path,len(axes))
        if interior and any(np.linalg.norm(target-end)<4*tolerance for end in (points[0],points[-1])):
            continue
        offset=target-start
        fraction=np.clip(np.einsum('ij,ij->i',offset,delta)/denom,0,1)
        error=offset-fraction[:,None]*delta
        if np.min(np.einsum('ij,ij->i',error,error))<tolerance*tolerance:return True
    return False


def event_covered(guess, curves, axes):
    same=[c for c in curves if c and all(c[0][k]==guess[k] for k in ('kind','invader','community'))]
    return covered(guess,same,axes,tolerance=.0002,interior=True)


def adaptive_cap(arm, nominal, factor):
    """Conservative curvature predictor in normalized parameter/state space.

    No increase before two nearly aligned secants. Near boundaries, sharp
    turns, small resident biomass or uncertain curvature use the original cap.
    This is a predictor, not a certified interpolation-error bound.
    """
    if factor<=1 or len(arm)<4:return nominal
    pts=np.array([_coverage_vector(r,2) for r in arm[-4:]])
    d=np.diff(pts,axis=0);length=np.linalg.norm(d,axis=1)
    if min(length)<1e-10:return nominal
    uv=pts[-1,:2]
    if min(*uv,*(1-uv))<factor*nominal:return nominal
    if any(arm[-1][c]<1e-4 for c in arm[-1]['community'] if c in 'MSP'):return nominal
    tang=d/length[:,None];turn=np.linalg.norm(np.diff(tang,axis=0),axis=1)
    curvature=max(turn/(.5*(length[:-1]+length[1:])))
    # Estimated sagitta of a chord, with a conservative target in normalized space.
    cap=np.sqrt(8*2e-5/max(curvature,1e-15))
    return float(np.clip(cap,nominal,nominal*factor))


def trace_branch(base, axis, seed, max_steps, checkpoint):
    arms, diagnostics = [], []
    for sign in (-1, 1):
        arm = []
        reason = 'step_limit'
        boundary_guess=None
        try:
            with native_problem(base, [axis], [seed['u']], state_of(seed), seed['community']) as p:
                state, uv = p.vector()
                arm.append(point_record(base, [axis], uv, state, seed['community']))
                ds = sign/min(axis.resolution*4, 1000)
                for step in range(max_steps):
                    checkpoint(f'Native {seed["community"]} branch · step {step+1}')
                    ds = p.arclength_continuation(p.controls[0], ds, max_ds=1/axis.resolution,
                                                 min_ds=1e-8, newton_solver_tolerance=1e-11)
                    state, uv = p.vector()
                    if not 0 <= uv[0] <= 1:
                        edge = float(np.clip(uv[0],0,1))
                        boundary_guess=(edge,state)
                        break
                    row = point_record(base,[axis],uv,state,seed['community'])
                    arm.append(row)
                    if len(arm)>30 and np.linalg.norm(np.r_[uv[0]-seed['u'],
                        (np.log1p(state)-np.log1p(state_of(seed)))/5]) < 1e-4:
                        reason='closed_loop'; break
        except (RuntimeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            reason = str(exc).splitlines()[0][:240]
        if boundary_guess is not None:
            try:
                edge,state=boundary_guess
                with native_problem(base,[axis],[edge],state,seed['community']) as boundary:
                    y,t=boundary.vector()
                    arm.append(point_record(base,[axis],t,y,seed['community']))
                reason='range_boundary'
            except (RuntimeError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                reason='Boundary correction: '+str(exc).splitlines()[0][:200]
        arms.append(arm)
        diagnostics.append(dict(community=seed['community'], direction=sign, reason=reason, points=len(arm)))
    return join_arms(arms, 1), diagnostics


def continue_1d(base, axis, discovery_points=17, max_steps=1200, checkpoint=lambda _: None):
    scan = []
    coordinates=discovery_coordinates(discovery_points)
    for i, u in enumerate(coordinates):
        checkpoint(f'Equilibrium overview {i+1}/{len(coordinates)}')
        scan.extend(discover(base,[axis],[u],grid_points=513))
    paths, diagnostics = [], []
    # Empty equilibrium is an exact constant branch, with no native unknowns.
    paths.append([r for r in scan if r['community']=='0'])
    seeds = sorted((r for r in scan if r['community']!='0'),
                   key=lambda r:(abs(r['u']-.5), r['community']))
    for i, seed in enumerate(seeds):
        checkpoint(f'Checking start {i+1}/{len(seeds)}')
        if covered(seed, paths, [axis]):continue
        path, notes = trace_branch(base,axis,seed,max_steps,checkpoint)
        diagnostics.extend(notes)
        if path:paths.append(path)
    # Mark starts with no native representation instead of silently treating scan points as continuation.
    for row in scan:
        if row['community']!='0' and not covered(row,paths,[axis]):
            diagnostics.append(dict(community=row['community'], reason='uncovered_discovery_root', x=row['x']))
    return dict(H=base.H, paths=paths, scan=scan, diagnostics=diagnostics)


def resident_matrix(y,p,face):
    ids=[i for i,c in enumerate('MSP') if c in face]
    return ids,jacobian(y,p)[np.ix_(ids,ids)]


def event_test(y,p,face,kind,invader,*,matrix=None):
    F=per_capita(y,p)
    if kind=='invasion':return float(F['MSP'.index(invader)])
    ids=[i for i,c in enumerate('MSP') if c in face]
    A=jacobian(y,p)[np.ix_(ids,ids)] if matrix is None else matrix[np.ix_(ids,ids)]
    if kind=='fold':
        # Remove the biomass factor: a species approaching zero is not a
        # resident saddle-node merely because det(J) approaches zero.
        D=(A-np.diag(F[ids]))/y[ids,None]
        return float(np.linalg.det(D/np.maximum(np.linalg.norm(D,axis=1),1e-30)[:,None]))
    # Continuous Hopf test, including intervals whose endpoint eigenvalues
    # are real. A 3D pair is imaginary when a1*a2=a3 and a2>0.
    scale=max(.01,float(np.max(np.abs(A))))
    coeff=np.poly(A/scale)
    return float(coeff[1] if len(ids)==2 else coeff[1]*coeff[2]-coeff[3])


def event_value(row,base,axes,kind,invader=''):
    if kind=='invasion' and row.get('invasion_'+invader) is not None:return row['invasion_'+invader]
    p=parameters_at(base,axes,[row['uv'[i]] for i in range(len(axes))])
    return event_test(state_of(row),p,row['community'],kind,invader)


def validate_event(row,base,axes,kind,invader,*,matrix=None):
    p=parameters_at(base,axes,[row['uv'[i]] for i in range(len(axes))])
    y=state_of(row)
    if kind=='invasion':return abs(row['invasion_'+invader])<1e-7 if row.get('invasion_'+invader) is not None else abs(event_test(y,p,row['community'],kind,invader))<1e-7
    if matrix is None:matrix=jacobian(y,p)
    ids=[i for i,c in enumerate('MSP') if c in row['community']]
    eig=np.linalg.eigvals(matrix[np.ix_(ids,ids)])
    if kind=='hopf':
        return (sum(abs(v.real)<1e-7 and abs(v.imag)>1e-6 for v in eig)==2
                and sum(abs(v)<1e-7 for v in eig)==0)
    return (sum(abs(v)<1e-7 for v in eig)==1 and
            abs(event_test(y,p,row['community'],kind,invader,matrix=matrix))<1e-5)


def precondition_event(base,axes,seed,kind,invader,solved_axis,bounds=None):
    """Find a nearby augmented-system guess; no SciPy curve is exported.

    The resident equations and event equation are solved together at fixed
    transverse control. Bounds prevent a remote Newton solution being used
    as the event inside a discovery bracket.
    """
    coords=np.array([seed['uv'[i]] for i in range(len(axes))],float)
    ids=[i for i,c in enumerate('MSP') if c in seed['community']]
    lo,hi=(0.,1.) if bounds is None else bounds
    lo=max(0.,lo-1e-9);hi=min(1.,hi+1e-9)
    if hi<=lo:raise ValueError('Empty event bracket')
    def unpack(z):
        y=np.zeros(3);y[ids]=np.exp(z[:-1]);t=coords.copy();t[solved_axis]=z[-1]
        return y,t,parameters_at(base,axes,t)
    def fun(z):
        y,t,p=unpack(z)
        _,N=resources(y,p)
        # Negative dissolved N is not physical. Keep trial residuals finite
        # near the Monod poles instead of permitting false singular roots.
        if N<0:
            # Evaluate at a feasible trial with the same relative M/P share.
            yy=y.copy();inventory=y[0]+p.H*y[2]
            if inventory>0:yy[[0,2]]*=p.N_wc/inventory
            F=per_capita(yy,p)[ids]
            return np.r_[F/.1+min(100.,-N),1.+min(100.,-N)]
        try:
            val=event_test(y,p,seed['community'],kind,invader)
            return np.r_[per_capita(y,p)[ids]/.1,val*(10. if kind=='invasion' else 1.)]
        except (ValueError,ArithmeticError,np.linalg.LinAlgError):return np.full(len(ids)+1,1e3)
    z=np.r_[np.maximum(np.log(np.maximum(state_of(seed)[ids],1e-8)),-18.42),np.clip(coords[solved_axis],lo,hi)]
    if not ids and kind=='invasion':
        fa,fb=fun([lo])[-1],fun([hi])[-1]
        if fa*fb>0:raise ValueError('No empty-state invasion crossing in bracket')
        z=np.array([brentq(lambda t:fun([t])[-1],lo,hi,xtol=1e-13)])
    else:
        result=least_squares(fun,z,bounds=(np.r_[np.full(len(ids),-18.42),lo],
                             np.r_[np.full(len(ids),max(30.,float(np.max(z[:-1]))+5.)),hi]),
                             x_scale='jac',max_nfev=160,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        z=result.x
    y,t,p=unpack(z)
    if np.max(np.abs(fun(z)))>2e-7:raise ValueError('Numerical event preconditioner did not converge')
    row=point_record(base,axes,t,y,seed['community'],event=(kind,invader))
    return dict(row,kind=kind,invader=invader,solved_axis=solved_axis)


def correct_event(base,axes,seed,kind,invader,solved_axis,bounds=None,*,prepared=False):
    # First prepare a well-conditioned seed; pyoomph remains the final
    # corrector and the sole producer of accepted continuation curves.
    guess=seed if prepared else precondition_event(base,axes,seed,kind,invader,solved_axis,bounds)
    coords=[guess['uv'[i]] for i in range(len(axes))]
    if _session:_session.stats['native_corrections']+=1
    with native_problem(base,axes,coords,state_of(guess),guess['community'],kind,invader,solved_axis) as p:
        y,t=p.vector()
        if np.any(t < -1e-7) or np.any(t > 1+1e-7):raise ValueError('Corrector left parameter domain')
        t=np.clip(t,0,1)
        row=point_record(base,axes,t,y,guess['community'],event=(kind,invader))
        if bounds is not None and not bounds[0]-1e-7<=t[solved_axis]<=bounds[1]+1e-7:
            raise ValueError('Native corrector left discovery bracket')
        return dict(row,kind=kind,invader=invader,solved_axis=solved_axis)


def event_candidates(before,after,base,axes,solved_axis):
    candidates=[];values={}
    def value(row,kind,invader=''):
        key=(id(row),kind,invader)
        if key not in values:values[key]=event_value(row,base,axes,kind,invader)
        return values[key]
    matched=match_roots(before,after)
    for a,b in matched:
        face=a['community']
        kinds=[('invasion',c) for c in 'MSP' if c not in face]
        if len(face)>=2:kinds.extend([('hopf',''),('fold','')])
        for kind,invader in kinds:
            va,vb=value(a,kind,invader),value(b,kind,invader)
            if not np.isfinite(va+vb) or va*vb>0 or abs(va)+abs(vb)<1e-12:continue
            for seed in sorted((a,b),key=lambda r:abs(value(r,kind,invader))):
                candidates.append((seed,kind,invader,solved_axis))
    # Saddle-node candidates require a pair of unmatched resident equilibria
    # on the same face, not just one species entering/leaving at zero biomass.
    for face in {r['community'] for r in before+after if len(r['community'])>=2}:
        aa=[r for r in before if r['community']==face]
        bb=[r for r in after if r['community']==face]
        if abs(len(aa)-len(bb))<2:continue
        side=0 if len(aa)>len(bb) else 1
        excess=[r for r in (aa if side==0 else bb) if not any(r is pair[side] for pair in matched)]
        for seed in sorted(excess,key=lambda r:abs(value(r,'fold'))):
            candidates.append((seed,'fold','',solved_axis))
    # Prioritize stability-relevant events; inexpensive preconditioning will
    # reject false fold/Hopf candidates before native compilation.
    return sorted(candidates,key=lambda c:(c[1]!='invasion',c[1]=='fold',
                        abs(value(c[0],c[1],c[2]))))


def same_event(a,b,axes,tolerance=1e-5):
    return (all(a[k]==b[k] for k in ('kind','community','invader')) and
            np.linalg.norm([a['uv'[i]]-b['uv'[i]] for i in range(len(axes))])<tolerance and
            np.max(np.abs(np.log1p(state_of(a))-np.log1p(state_of(b))))<1e-4)


def locate_events(base,axes,samples,solved_axis,checkpoint,max_attempts=None,*,branch_trace=False,on_prepared=None):
    events,notes=[],[];attempts=0
    for before,after in zip(samples[:-1],samples[1:]):
        if not before or not after:continue
        bounds=sorted((before[0]['uv'[solved_axis]],after[0]['uv'[solved_axis]]))
        for seed,kind,invader,axis in event_candidates(before,after,base,axes,solved_axis):
            if max_attempts is not None and attempts>=max_attempts:
                notes.append(dict(reason='Local event correction budget exhausted',attempts=attempts));return events,notes
            attempts+=1
            event_bounds=bounds
            # On a traced 1D branch the fold can lie just outside the
            # parameter interval between points on either side of a turn.
            if branch_trace and kind=='fold':
                step=1/axes[axis].resolution
                event_bounds=(max(0.,bounds[0]-step),min(1.,bounds[1]+step))
            try:
                # Precondition first; only one native correction per distinct
                # seed, even when neighboring brackets share the same event.
                guess=precondition_event(base,axes,seed,kind,invader,axis,event_bounds)
                if any(same_event(guess,e,axes) for e in events):continue
                if _session:_session.stats['prepared_seeds']+=1
                if on_prepared is not None:
                    event=on_prepared(guess,event_bounds)
                    if event is None:continue
                else:
                    checkpoint(f'Native {kind} correction · {seed["community"]}')
                    event=correct_event(base,axes,guess,kind,invader,axis,event_bounds,prepared=True)
                if not any(same_event(event,e,axes) for e in events):events.append(event)
            except (RuntimeError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                notes.append(dict(kind=kind,community=seed['community'],invader=invader,solved_axis=axis,
                    seed_coordinates=[seed['uv'[i]] for i in range(len(axes))],
                    seed_biomass=list(map(float,state_of(seed))),reason=str(exc).splitlines()[0][:240]))
    return events,notes


def join_arms(arms,dimensions):
    """Do not drop the successful arm's seed if the other arm failed."""
    a,b=arms
    if not a:return b
    if not b:return list(reversed(a))
    return list(reversed(a))+b[1:]


def adaptive_samples(lookup,coordinates,axis,held=None,rounds=5,budget=48):
    """Bisect changes of community multiplicity or stability on a transect."""
    def uv(t):return (t,) if held is None else (t,held) if axis==0 else (held,t)
    points={float(t):lookup(uv(float(t))) for t in coordinates}
    def fingerprint(rows):return sorted((r['community'],r['stability']) for r in rows)
    remaining=budget
    for _ in range(rounds):
        ts=sorted(points)
        intervals=[(a,b) for a,b in zip(ts[:-1],ts[1:]) if b-a>2e-6 and
                   fingerprint(points[a])!=fingerprint(points[b])]
        if not intervals or remaining<=0:break
        for a,b in sorted(intervals,key=lambda ab:ab[1]-ab[0],reverse=True)[:remaining]:
            t=(a+b)/2;points[t]=lookup(uv(t));remaining-=1
    return [points[t] for t in sorted(points)]


def _trace_event(base, axes, seed, max_steps, checkpoint, step_factor):
    arms, notes = [], []
    solved_axis=seed['solved_axis'];moving=1-solved_axis
    kind,invader=seed['kind'],seed['invader']
    for sign in (-1,1):
        arm=[];reason='step_limit';boundary_guess=None
        try:
            with native_problem(base,axes,[seed['u'],seed['v']],state_of(seed),seed['community'],
                                kind,invader,solved_axis) as p:
                y,t=p.vector();start=point_record(base,axes,t,y,seed['community'],event=(kind,invader))
                arm.append(dict(start,kind=kind,invader=invader))
                ds=sign/(4*max(a.resolution for a in axes))
                for step in range(max_steps):
                    checkpoint(f'Native 2D {kind} · {seed["community"]} · step {step+1}')
                    nominal=1/max(a.resolution for a in axes)
                    cap=adaptive_cap(arm,nominal,step_factor)
                    enlarged=abs(ds)>nominal and cap>nominal
                    if _session and enlarged:_session.stats['adaptive_steps']+=1
                    ds=p.arclength_continuation(p.controls[moving],ds,
                        max_ds=cap,min_ds=1e-8,newton_solver_tolerance=1e-11)
                    y,t=p.vector()
                    if np.any(t<0) or np.any(t>1):
                        previous=arm[-1];uv0=np.array([previous['u'],previous['v']])
                        crossings=[]
                        for i in (0,1):
                            if t[i]<0 or t[i]>1:
                                edge=float(np.clip(t[i],0,1));f=(edge-uv0[i])/(t[i]-uv0[i])
                                crossings.append((f,i,edge))
                        f,i,edge=min(crossings)
                        # Keep the solved control paired with its known biomass;
                        # changing both coordinates while retaining the old state
                        # can destabilize the augmented Newton initial condition.
                        guess=dict(previous)
                        guess['uv'[i]]=edge
                        boundary_guess=(guess,1-i)
                        break
                    row=point_record(base,axes,t,y,seed['community'],event=(kind,invader))
                    arm.append(dict(row,kind=kind,invader=invader,_adaptive_trial=enlarged))
                    if len(arm)>30 and np.linalg.norm(np.r_[t-[seed['u'],seed['v']],
                            (np.log1p(y)-np.log1p(state_of(seed)))/5])<1e-4:
                        reason='closed_loop';break
        except (RuntimeError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
            reason=str(exc).splitlines()[0][:240]
        # Correct only after the tracing Problem has closed. Nested pyoomph
        # Problems can interfere with native assembly state at boundary solves.
        if boundary_guess is not None:
            try:
                arm.append(correct_event(base,axes,boundary_guess[0],kind,invader,boundary_guess[1]))
                reason='range_boundary'
            except (RuntimeError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                reason='Boundary correction: '+str(exc).splitlines()[0][:200]
        arms.append(arm);notes.append(dict(kind=kind,community=seed['community'],invader=invader,direction=sign,reason=reason,points=len(arm),
            endpoint=[arm[-1]['u'],arm[-1]['v']] if arm else None))
        if reason=='closed_loop':return arm,notes
    return join_arms(arms,2),notes


def midpoint_ok(a,b,base,axes,tol=2e-5):
    """Independent numerical midpoint probe of an enlarged native segment.

    A failed probe requests a full native retrace at the original step cap.
    The probe is never exported as a native point or used to colour a region.
    """
    uva=np.array([a['u'],a['v']]);uvb=np.array([b['u'],b['v']]);delta=uvb-uva
    held=int(np.argmax(np.abs(delta)))
    if abs(delta[held])<1e-10:return False
    mid=(uva+uvb)/2
    y=np.expm1((np.log1p(state_of(a))+np.log1p(state_of(b)))/2)
    guess=dict(a,u=mid[0],v=mid[1],M=y[0],S=y[1],P=y[2])
    try:
        test=precondition_event(base,axes,guess,a['kind'],a['invader'],1-held)
    except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError):return False
    target=(_coverage_vector(a,2)+_coverage_vector(b,2))/2
    return np.linalg.norm(_coverage_vector(test,2)-target)<=tol


def trace_event(base,axes,seed,max_steps,checkpoint):
    factor=_session.step_factor if _session else 1.
    curve,notes=_trace_event(base,axes,seed,max_steps,checkpoint,factor)
    nominal=1/max(a.resolution for a in axes)
    if factor>1:
        for a,b in zip(curve[:-1],curve[1:]):
            if not (a.get('_adaptive_trial') or b.get('_adaptive_trial')):continue
            if np.linalg.norm(_coverage_vector(a,2)-_coverage_vector(b,2))<=nominal*1.05:continue
            if not midpoint_ok(a,b,base,axes):
                if _session:_session.stats['adaptive_fallbacks']+=1
                checkpoint('Adaptive chord check failed · retracing at original step cap')
                fixed,extra=_trace_event(base,axes,seed,max_steps,checkpoint,1.)
                notes.append(dict(kind=seed['kind'],community=seed['community'],reason='adaptive_midpoint_fallback'))
                notes.extend(extra);curve=fixed;break
    for row in curve:row.pop('_adaptive_trial',None)
    return curve,notes


def continue_2d(base,axes,discovery_points=17,transects=5,max_steps=1200,
                checkpoint=lambda _:None,discovery_refinement=5):
    events,notes,cache,curves,prepared=[],[],{},[],[]
    def process(guess,bounds):
        if event_covered(guess,curves,axes):
            if _session:_session.stats['covered_seeds']+=1
            return None
        checkpoint(f'Native {guess["kind"]} correction · {guess["community"]}')
        event=correct_event(base,axes,guess,guess['kind'],guess['invader'],guess['solved_axis'],bounds,prepared=True)
        curve,diagnostics=trace_event(base,axes,event,max_steps,checkpoint)
        notes.extend(diagnostics)
        if len(curve)>=2:curves.append(curve);return event
        # Leave the seed eligible for retry from a different transect.
        return None
    def lookup(coords):
        coords=tuple(round(float(t),14) for t in coords)
        if coords not in cache:cache[coords]=discover(base,axes,coords,grid_points=257)
        return cache[coords]
    for axis in (0,1):
        # Include edge-clustered held values as well as uniform transects.
        # Nutrient transitions often occupy only the low-inventory tail.
        held_values=discovery_coordinates(transects)
        # Wide nitrogen inventories concentrate important transitions at
        # the low-inventory end (observed in the supplied SIMP run).
        if axes[1-axis].name=='N_wc' and axes[1-axis].high-axes[1-axis].low>500:
            held_values=np.unique(np.r_[held_values,.025,.05,.075])
        for held in held_values:
            checkpoint(f'2D equilibrium overview · axis {axis+1} · slice {held:.3f}')
            samples=adaptive_samples(lookup,discovery_coordinates(discovery_points),axis,held,
                                     rounds=discovery_refinement,budget=48)
            def collect(guess,bounds):
                prepared.append((guess,bounds));return None
            found,failed=locate_events(base,axes,samples,axis,checkpoint,on_prepared=collect)
            notes.extend(failed)
    # Prefer central starts; keep alternate charts and retries when a trace fails.
    prepared.sort(key=lambda item:(item[0]['kind']=='fold',item[0]['kind']=='hopf',
                                   abs(item[0]['u']-.5)+abs(item[0]['v']-.5)))
    for guess,bounds in prepared:
        try:
            event=process(guess,bounds)
            if event is not None:events.append(event)
        except (RuntimeError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
            notes.append(dict(kind=guess['kind'],community=guess['community'],invader=guess['invader'],
                              solved_axis=guess['solved_axis'],seed_coordinates=[guess['u'],guess['v']],
                              seed_biomass=list(map(float,state_of(guess))),reason=str(exc).splitlines()[0][:240]))
    return dict(H=base.H,curves=curves,events=events,diagnostics=notes,
                scan=[r for rows in cache.values() for r in rows])
