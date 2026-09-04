"""
Benchmark of QP solvers on the MPC QP benchmark set
(LIPMWALK / WHLIPBAL / QUADCMPC, converted by generate_MPC_data.py).

Loads instances from MPC_data/ through data_loader. They are stored in the
two-sided form of the original repository

    minimize    0.5 x'Qx + q'x + c   s.t.   l_h <= Gx <= u_h,  l_x <= x <= u_x

and are converted here with data_loader.to_split to the split form

    minimize    0.5 x'Qx + q'x + c   s.t.   Gx <= h,   Ax = b

with the genuine inequalities in G and the equality rows kept as Ax = b, so
every solver receives equality constraints natively (no ± doubling). Each
instance is solved with Warm-IP (path-following ADMM warm start + interior
point), the same interior-point method cold-started (IP), PIQP, MOSEK,
Clarabel, OSQP, and Gurobi. For every solver the runtime, objective value,
and normalized KKT quality (primal_res, dual_res, dual_gap and their
maximum) are reported, all computed with the same metric
(data_loader.kkt_metrics, which includes the equality residual and A'v
term on mixed problems). Results go to results/MPC_solver_benchmark.csv;
figures are regenerated from the CSV by generate_paper_figures.py.

How Warm-IP/IP treat the equality block is set by EQ_MODE below ('direct':
native, augmented KKT; 'absorb': rewritten as ± inequality pairs inside the
solver). Equality-only instances (all constraints are equalities and no
variable is bounded) have no strictly feasible interior, so an
interior-point iteration does not apply: Warm-IP and IP both solve the
saddle-point system in closed form
(warm_ip_solver.solve_equality_constrained_qp) and therefore report the
same time and solution for those problems.

Warm-IP and IP optionally run Ruiz equilibration first ('ruiz' in their
configs; OSQP's modified Ruiz algorithm). Termination then uses
unscaled residuals, as OSQP does by default, so eps_overall keeps its
meaning, and the returned solution is mapped back to the original scale.

All solvers are configured with the same termination tolerance eps_overall
to the extent their parameter sets allow. Note: Gurobi and MOSEK require
licenses; PIQP, OSQP, and Clarabel are free. Each solver runs in its own
try/except, so a missing solver or license only blanks that solver's columns.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))


import os
import time

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from scipy import sparse

import sparse_dot_mkl as smkl
from sparse_dot_mkl import dot_product_mkl as dot_mkl

from data_loader import (kkt_metrics, select_instances, load_data,
                         to_split, ensure_local)
from warm_ip_solver import solve_qp
from generate_paper_figures import print_runtime_summary, print_kkt_summary

# ---------------------------------------------------------------------------
# Which problems to run: "all", or a list of entries. An entry may be an
# instance number (1, 31, 35 -> MPC_001..., MPC_031..., MPC_035...), a full name
# ("MPC_001_LIPMWALK0"), or the original short name ("LIPMWALK0", "QUADCMPC1").
# ---------------------------------------------------------------------------
PROBLEMS = "all"




# ---------------------------------------------------------------------------
# Which solvers to run: remove/comment names to skip them (for debugging).
# ---------------------------------------------------------------------------
SOLVERS = [
    "Warm-IP",     # path-following ADMM warm start + interior point (paper)
    "IP",          # the same interior-point method, cold-started
    "PIQP",
    "MOSEK",
    "Clarabel",
    "OSQP",
    "Gurobi",
]

# Per-instance convergence and time-comparison plots. With all 64 problems
# this writes over a hundred PNGs, so it is off by default.
MAKE_PLOTS = False

# Save generated figures (.png / .pdf) to results/. Set False to only show
# them interactively (with WARMIP_NOSHOW=1 nothing is displayed either).
SAVE_FIGS = False

# A solver counts as FAILED on a problem when it returns no result, exposes
# no multipliers, or max(primal_res, dual_res, dual_gap) exceeds this.
FAIL_KKT_TOL = 1e-3

# ---------------------------------------------------------------------------
# Equality handling in the Warm-IP / IP runs on problems with BOTH
# inequalities and equalities:
#   'direct'  ADMM and IP handle Ax = b natively (augmented KKT; PARDISO
#             backend only)
#   'absorb'  Ax = b is stacked into G as the pair Ax <= b, -Ax <= -b
# Pure-equality problems always take the closed-form path. The external
# solvers always receive equalities natively.
# ---------------------------------------------------------------------------
EQ_MODE = 'absorb'

# ---------------------------------------------------------------------------
# Fault isolation. Native errors inside a solver library (PARDISO/MKL for
# Warm-IP/IP, but equally the C/C++/Rust cores of PIQP, MOSEK, Clarabel,
# OSQP and Gurobi) can kill the Python process outright (segfault/abort),
# which no try/except can catch. With isolation on, each solve runs in its
# own spawned child process, so a native crash -- or exceeding
# SOLVER_TIMEOUT (seconds, None = no limit) -- only blanks that solver's
# columns and the benchmark continues with the next solver/problem.
# Timings are measured inside the child (process startup and data loading
# are excluded). Set False to run in-process (easier debugging).
# ---------------------------------------------------------------------------
ISOLATE_WARMIP = True     # Warm-IP / IP solves
ISOLATE_EXTERNAL = True   # PIQP / MOSEK / Clarabel / OSQP / Gurobi solves
# Solvers EXCLUDED from external isolation (they run in-process, the
# pre-isolation behavior): on some Windows/conda setups cvxpy's
# compiled core fails to import inside a spawned child (numpy DLL
# resolution) although it imports fine in the parent, which breaks
# MOSEK-via-CVXPY. MOSEK is crash-rare, so in-process is acceptable;
# note SOLVER_TIMEOUT cannot interrupt a non-isolated solve.
#ISOLATE_SKIP = ("MOSEK",)
ISOLATE_SKIP = ()
SOLVER_TIMEOUT = 360

# All program outputs (plots, CSV) are written here
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, 'MPC_solver_benchmark.csv')

# Set MKL threads
num_threads_to_use = int(psutil.cpu_count(logical=False) * 0.45)
smkl.mkl_set_num_threads(num_threads_to_use)
smkl.mkl_set_num_threads_local(num_threads_to_use)

eps_overall = 1e-6   # single accuracy knob shared by all solvers

# Configs
ADMM_config = {
    'rho': 0.5,
    'mu': 'auto',          # ADMM barrier weight; None -> variant default
    'eps_res': 5e-1,
    'num_itr_check_convergence': 50,
    'rho_update_threshold': 2,
    'max_iter': 200,
    'verbose': 1
}

WARM_IP_config = {
    'max_iter': 100,
    'tol': eps_overall,
    'warm_start': True,
    'admm_config': ADMM_config,
    'ruiz': False,        # Ruiz equilibration preprocessing (OSQP algorithm)
}

IP_config = {
    'max_iter': 100,
    'tol': eps_overall,
    'warm_start': False,
    'admm_config': ADMM_config,
    'ruiz': False,        # Ruiz equilibration preprocessing (OSQP algorithm)
}


def objective_value(x, Q, q, c):
    return 0.5 * dot_mkl(x.T, dot_mkl(Q, x)) + dot_mkl(q.T, x) + c


def _warmip_worker(out_queue, name, data_dir, config, eq_mode_):
    """Child-process entry point for one isolated Warm-IP / IP solve: load
    the instance, solve, ship the results back through the queue. Runs in a
    fresh interpreter (spawn), so a native PARDISO/MKL crash kills only
    this process, not the benchmark."""
    try:
        prob = load_data(name, data_dir)
        Q, q, c, G, h, A_eq, b_eq = to_split(prob)
        m_ineq = G.shape[0]
        t0 = time.time()
        x, info, admm_time, info_admm = solve_qp(
            Q, q, G if m_ineq else None, h if m_ineq else None, A_eq, b_eq,
            **config,
            eq_mode=eq_mode_)
        elapsed = time.time() - t0
        out_queue.put(('ok', (x, info, admm_time, info_admm, elapsed)))
    except Exception as e:                       # Python-level failure
        out_queue.put(('error', f"{type(e).__name__}: {e}"))


def _run_isolated(worker, worker_args):
    """Run ``worker`` in a spawned child process and return its payload.

    The worker's first argument is the result queue; it must put
    ('ok', payload) or ('error', message). Raises RuntimeError on a child
    exception, a native crash (child dies without reporting), or when
    SOLVER_TIMEOUT elapses (child is terminated). Shared by the Warm-IP/IP
    and the external-solver isolation paths."""
    import multiprocessing as mp
    import queue as _queue_mod
    ctx = mp.get_context('spawn')
    out_queue = ctx.Queue()
    proc = ctx.Process(target=worker, args=(out_queue,) + tuple(worker_args))
    t_start = time.time()
    proc.start()
    result = None
    while result is None:
        try:
            # fetch BEFORE join: large payloads deadlock a joined queue
            result = out_queue.get(timeout=1.0)
        except _queue_mod.Empty:
            if not proc.is_alive():
                break        # died without reporting: native crash
            if SOLVER_TIMEOUT is not None and \
                    time.time() - t_start > SOLVER_TIMEOUT:
                proc.terminate()
                proc.join()
                raise RuntimeError(
                    f"timed out after {SOLVER_TIMEOUT:g} s (terminated)")
    proc.join()
    if result is None:
        raise RuntimeError(
            f"solver process died (exit code {proc.exitcode}), likely a "
            f"native crash inside the solver library")
    status, payload = result
    if status == 'error':
        raise RuntimeError(payload)
    return payload


def run_warmip(name, data_dir, config, solver_args):
    """One Warm-IP / IP solve; isolated in a child process when
    ISOLATE_WARMIP is set. Returns (x, info, admm_time, info_admm, elapsed)
    or raises RuntimeError describing the failure (exception, native crash,
    or timeout) so the caller's try/except records it and moves on."""
    if not ISOLATE_WARMIP:
        t0 = time.time()
        x, info, admm_time, info_admm = solve_qp(
            *solver_args, **config, eq_mode=EQ_MODE)
        return x, info, admm_time, info_admm, time.time() - t0
    return _run_isolated(_warmip_worker,
                         (name, data_dir, config, EQ_MODE))


def _solve_external(solver_name, Q, q, c, G, h, A_eq, b_eq):
    """One external-solver run (PIQP / MOSEK / Clarabel / OSQP / Gurobi) in
    whatever process this executes in. Returns {'time', 'x', 'dual', 'v'}
    with dual the multipliers of Gx <= h and v those of Ax = b (either may
    be None when the solver exposes none). Raises on solver errors."""
    n = Q.shape[0]
    m_ineq = G.shape[0] if G is not None else 0
    has_eq = A_eq is not None and A_eq.shape[0] > 0
    p_eq = A_eq.shape[0] if has_eq else 0

    if solver_name == "PIQP":
        import piqp
        solver = piqp.SparseSolver()
        solver.settings.verbose = True
        solver.settings.eps_abs = eps_overall
        solver.settings.eps_rel = eps_overall
        for _gap_attr in ("eps_duality_gap_abs", "eps_duality_gap_rel"):
            if hasattr(solver.settings, _gap_attr):
                setattr(solver.settings, _gap_attr, eps_overall)
        solver.settings.max_iter = 250
        solver.setup(P=sp.csc_matrix(Q), c=q,
                     A=sp.csc_matrix(A_eq) if has_eq else None,
                     b=b_eq if has_eq else None,
                     G=sp.csc_matrix(G) if m_ineq else None,
                     h_l=np.full(m_ineq, -np.inf) if m_ineq else None,
                     h_u=h if m_ineq else None)
        t0 = time.time()
        solver.solve()
        elapsed = time.time() - t0
        res = solver.result
        if hasattr(res, 'z_u') and hasattr(res, 'z_l'):
            z_piqp = np.asarray(res.z_u) - np.asarray(res.z_l)
        elif hasattr(res, 'z'):
            z_piqp = np.asarray(res.z)
        else:
            z_piqp = None
        # PIQP Lagrangian: L = ... + y'(Ax - b), so v = y directly
        v_piqp = np.asarray(res.y) if has_eq and hasattr(res, 'y') else None
        return {'time': elapsed, 'x': np.asarray(res.x),
                'dual': z_piqp, 'v': v_piqp}

    if solver_name == "MOSEK":
        import cvxpy as cp
        x = cp.Variable(n)
        constraints = []
        ineq_con = eq_con = None
        if m_ineq:
            ineq_con = G @ x <= h
            constraints.append(ineq_con)
        if has_eq:
            eq_con = A_eq @ x == b_eq
            constraints.append(eq_con)
        # convex QPs: Q is PSD; assume_PSD skips CVXPY's expensive check
        prob_cp = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(x, Q, assume_PSD=True) + q @ x),
            constraints)
        t0 = time.time()
        prob_cp.solve(solver='MOSEK', mosek_params={
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": eps_overall,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": eps_overall,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": eps_overall,
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 400,
        }, verbose=True)
        elapsed = time.time() - t0
        # CVXPY dual conventions match ours: Qx + q + G'z + A'v = 0
        z_mosek = ineq_con.dual_value if ineq_con is not None else np.zeros(0)
        v_mosek = eq_con.dual_value if eq_con is not None else None
        return {'time': elapsed, 'x': np.asarray(x.value),
                'dual': np.asarray(z_mosek) if z_mosek is not None else None,
                'v': np.asarray(v_mosek) if v_mosek is not None else None}

    if solver_name == "Clarabel":
        import clarabel
        # Clarabel reads a triangle and treats it as symmetric. Equality
        # rows go first as a ZeroCone block, inequalities after as a
        # NonnegativeCone block.
        Q_bar = sparse.triu(Q, format="csc")
        _blocks = ([sp.csc_matrix(A_eq)] if has_eq else []) + \
                  ([sp.csc_matrix(G)] if m_ineq else [])
        A_full = sp.csc_matrix(sp.vstack(_blocks, format="csc"))
        b_full = np.concatenate(
            ([b_eq] if has_eq else []) + ([h] if m_ineq else []))
        cones = ([clarabel.ZeroConeT(p_eq)] if has_eq else []) + \
                ([clarabel.NonnegativeConeT(m_ineq)] if m_ineq else [])
        settings = clarabel.DefaultSettings()
        settings.verbose = True
        settings.tol_feas = eps_overall
        settings.tol_gap_abs = eps_overall
        settings.tol_gap_rel = eps_overall
        settings.max_iter = 200
        solver = clarabel.DefaultSolver(Q_bar, q, A_full, b_full, cones,
                                        settings)
        t0 = time.time()
        solution = solver.solve()
        elapsed = time.time() - t0
        z_full = np.asarray(solution.z)
        v_cl = z_full[:p_eq] if has_eq else None
        z_cl = z_full[p_eq:]
        return {'time': elapsed, 'x': np.asarray(solution.x),
                'dual': z_cl, 'v': v_cl}

    if solver_name == "OSQP":
        import osqp
        prob_osqp = osqp.OSQP()
        # inequality rows first (l = -inf), equality rows as l = u
        _blocks = ([sp.csc_matrix(G)] if m_ineq else []) + \
                  ([sp.csc_matrix(A_eq)] if has_eq else [])
        A_osqp = sp.csc_matrix(sp.vstack(_blocks, format="csc"))
        l_osqp = np.concatenate(
            ([np.full(m_ineq, -np.inf)] if m_ineq else []) +
            ([b_eq] if has_eq else []))
        u_osqp = np.concatenate(
            ([h] if m_ineq else []) + ([b_eq] if has_eq else []))
        prob_osqp.setup(sp.csc_matrix(Q), q, A_osqp, l_osqp, u_osqp,
                        eps_abs=eps_overall, eps_rel=eps_overall,
                        max_iter=4000, verbose=True)
        t0 = time.time()
        res = prob_osqp.solve()
        elapsed = time.time() - t0
        y_osqp = np.asarray(res.y) if res.y is not None else None
        z_osqp = y_osqp[:m_ineq] if y_osqp is not None else None
        v_osqp = y_osqp[m_ineq:] if (y_osqp is not None and has_eq) else None
        return {'time': elapsed, 'x': res.x, 'dual': z_osqp, 'v': v_osqp}

    if solver_name == "Gurobi":
        import gurobipy as gp
        from gurobipy import GRB
        model = gp.Model("QP_Gx_leq_h")
        model.setParam("OutputFlag", 0)
        model.setParam("BarConvTol", eps_overall)
        model.setParam("FeasibilityTol", eps_overall)
        model.setParam("OptimalityTol", eps_overall)
        model.setParam("BarIterLimit", 1000)
        x = model.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        model.setObjective(0.5 * x @ Q @ x + q @ x, GRB.MINIMIZE)
        ineq_c = model.addConstr(G @ x <= h, name="ineq") if m_ineq else None
        eq_c = model.addConstr(A_eq @ x == b_eq, name="eq") if has_eq else None
        t0 = time.perf_counter()
        model.optimize()
        elapsed = time.perf_counter() - t0
        # Gurobi's Pi satisfies Qx + q - G'Pi_ineq - A'Pi_eq = 0; our
        # convention (Qx + q + G'z + A'v = 0) needs z = -Pi, v = -Pi.
        try:
            z_grb = -np.asarray(ineq_c.Pi) if ineq_c is not None \
                else np.zeros(0)
        except Exception:
            z_grb = None
        try:
            v_grb = -np.asarray(eq_c.Pi) if eq_c is not None else None
        except Exception:
            v_grb = None
        return {'time': elapsed, 'x': np.asarray(x.X),
                'dual': z_grb, 'v': v_grb}

    raise ValueError(f"unknown external solver {solver_name!r}")


def _external_worker(out_queue, solver_name, name, data_dir):
    """Child-process entry point for one isolated external solve: load the
    instance, solve, ship {'time','x','dual','v'} back through the queue.
    A native crash inside the solver library kills only this process."""
    try:
        prob = load_data(name, data_dir)
        Q, q, c, G, h, A_eq, b_eq = to_split(prob)
        out_queue.put(('ok', _solve_external(
            solver_name, Q, q, c, G, h, A_eq, b_eq)))
    except Exception as e:
        out_queue.put(('error', f"{type(e).__name__}: {e}"))


def run_external(solver_name, name, data_dir, canon):
    """One external solve; isolated in a child process when
    ISOLATE_EXTERNAL is set (native crashes and SOLVER_TIMEOUT then only
    blank this solver's columns). Returns {'time','x','dual','v'}."""
    if not ISOLATE_EXTERNAL or solver_name in ISOLATE_SKIP:
        return _solve_external(solver_name, canon['Q'], canon['q'],
                               canon['c'], canon['G'], canon['h'],
                               canon.get('A'), canon.get('b'))
    return _run_isolated(_external_worker, (solver_name, name, data_dir))


def record_solution(results, canon, solver_name, x, dual, v=None):
    """Store objective and KKT quality columns for one solver's solution.

    canon carries the split problem ({'Q','q','c','G','h','A','b'}; A/b may
    be None). dual are the multipliers of Gx <= h, v those of Ax = b. When
    the problem has equalities but the solver exposed no v (or no dual at
    all), the KKT columns stay blank except primal_res, which only needs x.
    """
    Q, q, c = canon['Q'], canon['q'], canon['c']
    A, b = canon.get('A'), canon.get('b')
    has_eq = A is not None and A.shape[0] > 0
    results[f'{solver_name}_obj'] = objective_value(x, Q, q, c)
    if dual is not None and (not has_eq or v is not None):
        primal_res, dual_res, dual_gap = kkt_metrics(
            canon, x, dual, A=A, b=b, v=(v if has_eq else None))
        results[f'{solver_name}_primal_res'] = primal_res
        results[f'{solver_name}_dual_res'] = dual_res
        results[f'{solver_name}_dual_gap'] = dual_gap
        results[f'{solver_name}_kkt_max'] = max(primal_res, dual_res, dual_gap)
        print(f">>> {solver_name} KKT: primal_res={primal_res:.2e} "
              f"dual_res={dual_res:.2e} dual_gap={dual_gap:.2e} "
              f"(max {max(primal_res, dual_res, dual_gap):.2e})")
    else:
        G, h = canon['G'], canon['h']
        parts = []
        if G is not None and G.shape[0] > 0:
            Gx = G @ x
            parts.append(np.linalg.norm(np.maximum(Gx - h, 0.0), np.inf) / (
                1.0 + max(np.linalg.norm(Gx, np.inf),
                          np.linalg.norm(h, np.inf))))
        if has_eq:
            Ax = A @ x
            parts.append(np.linalg.norm(Ax - b, np.inf) / (
                1.0 + max(np.linalg.norm(Ax, np.inf),
                          np.linalg.norm(b, np.inf))))
        primal_res = max(parts) if parts else 0.0
        results[f'{solver_name}_primal_res'] = primal_res
        print(f">>> {solver_name} KKT: primal_res={primal_res:.2e} "
              f"(no multipliers exposed)")


def plot_warmip_convergence(info, info_admm, instance, solver_name):
    """One log-scale figure per run: ADMM primal/dual residuals (recorded
    every num_itr_check_convergence iterations) followed by the IP stage's
    primal/dual/complementarity residuals (every iteration). Saved to
    results/warmip_convergence_<solver>_<instance>.png; set WARMIP_NOSHOW=1
    to skip the interactive window."""
    import matplotlib
    if os.environ.get("WARMIP_NOSHOW") == "1":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_admm = 0
    if isinstance(info_admm, dict) and info_admm.get('primal_residual'):
        ap = np.asarray(info_admm['primal_residual'], dtype=float)
        ad = np.asarray(info_admm['dual_residual'], dtype=float)
        xs = np.arange(ap.size)          # each reported ADMM point = 1 tick
        n_admm = int(ap.size)
        _mp, _md = ap > 0, ad > 0   # log scale cannot display zeros
        ax.semilogy(xs[_mp], ap[_mp], 'o-', color='tab:blue', ms=4,
                    label='ADMM primal res')
        ax.semilogy(xs[_md], ad[_md], 'o-', color='tab:orange', ms=4,
                    label='ADMM dual res')
    ip_x0 = n_admm
    if isinstance(info, dict):
        for key, color, label in [('primal_residual', 'tab:blue', 'IP primal res'),
                                  ('dual_residual', 'tab:orange', 'IP dual res'),
                                  ('compl_res', 'tab:green', 'IP complementarity')]:
            vals = info.get(key)
            if vals is not None and len(vals):
                v = np.asarray(vals, dtype=float)
                _mv = v > 0
                ax.semilogy((ip_x0 + np.arange(1, v.size + 1))[_mv], v[_mv],
                            's--', color=color, ms=4, label=label)
    if n_admm:
        ax.axvline(ip_x0, color='gray', lw=1, ls=':')
        ax.annotate('ADMM -> IP', xy=(ip_x0, 1),
                    xycoords=('data', 'axes fraction'),
                    xytext=(4, -12), textcoords='offset points', color='gray')
    ax.set_xlabel('reported point (ADMM checks, then IP iterations)')
    ax.set_ylabel('residual')
    ax.set_title(f'{solver_name} convergence on {instance} '
                 f'(circles: ADMM, squares: IP; log scale)')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.legend(loc='best')
    fig.tight_layout()
    fname = f'warmip_convergence_{solver_name}_{instance}.png'
    if SAVE_FIGS:
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
        print(f'convergence plot saved to {fname}', flush=True)
    if os.environ.get("WARMIP_NOSHOW") != "1":
        plt.show()
    plt.close(fig)


def plot_time_comparison(warm_run, ip_run, instance, origin=None):
    """When both Warm-IP and IP were run on an instance: max residual vs
    wall-clock time on a log scale, one curve per method. For Warm-IP the
    star marker shows where the algorithm switches from ADMM to IP."""
    import matplotlib
    if os.environ.get("WARMIP_NOSHOW") == "1":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def ip_curve(info, t0):
        t = t0 + np.cumsum(np.asarray(info.get('time', []), dtype=float))
        parts = [np.asarray(info.get(k, []), dtype=float)
                 for k in ('primal_residual', 'dual_residual', 'compl_res')]
        nmin = min([t.size] + [p.size for p in parts])
        if nmin == 0:
            return np.array([]), np.array([])
        return t[:nmin], np.maximum.reduce([p[:nmin] for p in parts])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    info, info_admm, admm_time = warm_run
    if isinstance(info_admm, dict) and info_admm.get('time'):
        ta = np.asarray(info_admm['time'], dtype=float)      # cumulative
        ra = np.maximum(np.asarray(info_admm['primal_residual'], dtype=float),
                        np.asarray(info_admm['dual_residual'], dtype=float))
        n = min(ta.size, ra.size)
        ti, ri = ip_curve(info, admm_time)
        ax.semilogy(np.concatenate([ta[:n], ti]),
                    np.concatenate([ra[:n], ri]),
                    '-', color='tab:blue', lw=1.8, label='Warm-IP')
        y_switch = ri[0] if ri.size else ra[n - 1]
        ax.semilogy([admm_time], [y_switch], marker='*', color='tab:blue',
                    ms=15, ls='none', label='ADMM -> IP switch')
    else:
        ti, ri = ip_curve(info, 0.0)
        ax.semilogy(ti, ri, '-', color='tab:blue', lw=1.8, label='Warm-IP')

    info_ip = ip_run[0]
    t_init = ip_run[2] if len(ip_run) > 2 else 0.0  # cold-start initializer time
    tc, rc = ip_curve(info_ip, t_init)
    if origin is not None and tc.size:
        tc = np.concatenate([[0.0], tc])
        rc = np.concatenate([[origin], rc])
    if tc.size:
        ax.semilogy(tc, rc, '-', color='tab:red', lw=1.8,
                    label='IP (cold start)')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('max residual')
    ax.set_title(f'Warm-IP vs IP on {instance}: max residual vs time '
                 f'(log scale)')
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.legend(loc='best')
    fig.tight_layout()
    fname = f'residual_vs_time_{instance}.png'
    if SAVE_FIGS:
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
        print(f'time-comparison plot saved to {fname}', flush=True)
    if os.environ.get("WARMIP_NOSHOW") != "1":
        plt.show()
    plt.close(fig)


def benchmark_qp_solvers(data_dir=None):
    all_results = []
    names = select_instances(PROBLEMS, data_dir, prefix="MPC_")
    ensure_local(names, data_dir)   # prefetch any missing instances
    print(f"{len(names)} problem(s) selected")

    for name in names:
        prob = load_data(name, data_dir)
        Q, q, c, G, h, A_eq, b_eq = to_split(prob)
        canon = {'Q': Q, 'q': q, 'c': c, 'G': G, 'h': h,
                 'A': A_eq, 'b': b_eq}
        n = Q.shape[0]
        m_ineq = G.shape[0]
        p_eq = A_eq.shape[0] if A_eq is not None else 0
        has_eq = p_eq > 0
        pure_eq = has_eq and m_ineq == 0
        results = {"Problem": name, "n": n, "m": m_ineq, "p": p_eq,
                   "pure_equality": int(pure_eq)}
        _warm_run = None
        _ip_run = None

        print("=" * 79)
        print("Solving", name, f"(n={n}, m={m_ineq}, p={p_eq}"
              f"{', equality-only' if pure_eq else ''})")

        # Pure-equality problems are passed with no inequality block, which
        # triggers the closed-form path in the IP solvers; mixed problems
        # carry both blocks and EQ_MODE decides how Warm-IP/IP treat A.
        solver_args = (Q, q, G if m_ineq else None, h if m_ineq else None,
                       A_eq, b_eq)

        def _ip_duals(info):
            """(z, v) of the split rows, from any warm_ip path."""
            if not isinstance(info, dict):
                return None, None
            return info.get('z'), info.get('v')

        if "Warm-IP" in SOLVERS:
            print("Solving with Warm-IP")
            try:
                x_final, info, admm_time, info_admm, _elapsed = run_warmip(
                    name, data_dir, WARM_IP_config, solver_args)
                results['Warm-IP_time'] = _elapsed
                # IP iterations (one history entry per iteration; an
                # equality-only problem solved in closed form reports 1)
                results['Warm-IP_iters'] = len(info.get(
                    'objective_original', [])) if isinstance(info, dict) else None
                # how the run ended: converged | stalled | diverged |
                # max_iter | closed_form (see IP_SAFEGUARDS in warm_ip_solver)
                results['Warm-IP_stop'] = info.get(
                    'stop_reason') if isinstance(info, dict) else None
                _z, _v = _ip_duals(info)
                record_solution(results, canon, 'Warm-IP', x_final, _z, _v)
                print(f">>> Warm-IP time: {results['Warm-IP_time']:.3f} s")
                if MAKE_PLOTS and not pure_eq:
                    try:
                        plot_warmip_convergence(info, info_admm, name, 'Warm-IP')
                    except Exception as _pe:
                        print(f"(convergence plot failed: {_pe})")
                _warm_run = (info, info_admm, admm_time)
            except Exception as e:
                print(f"Warm-IP error in {name}: {e}")
                results['Warm-IP_error'] = str(e)

        if "IP" in SOLVERS:
            print("Solving with IP (cold start)")
            try:
                x_final_ip, info_ip, admm_time, info_admm_ip, _elapsed = \
                    run_warmip(name, data_dir, IP_config, solver_args)
                results['IP_time'] = _elapsed
                results['IP_iters'] = len(info_ip.get(
                    'objective_original', [])) if isinstance(info_ip, dict) else None
                results['IP_stop'] = info_ip.get(
                    'stop_reason') if isinstance(info_ip, dict) else None
                _z, _v = _ip_duals(info_ip)
                record_solution(results, canon, 'IP', x_final_ip, _z, _v)
                print(f">>> IP (cold start) time: {results['IP_time']:.3f} s")
                if MAKE_PLOTS and not pure_eq:
                    try:
                        plot_warmip_convergence(info_ip, info_admm_ip, name, 'IP')
                    except Exception as _pe:
                        print(f"(convergence plot failed: {_pe})")
                _ip_run = (info_ip, info_admm_ip, admm_time)
            except Exception as e:
                print(f"IP error in {name}: {e}")
                results['IP_error'] = str(e)

        if MAKE_PLOTS and not pure_eq and _warm_run is not None and _ip_run is not None:
            try:
                _rhs = np.concatenate([h, b_eq]) if has_eq else h
                _ph = float(np.linalg.norm(_rhs, np.inf))
                _dq = float(np.linalg.norm(q, np.inf))
                _origin = max(_ph / (1.0 + _ph), _dq / (1.0 + _dq))
                plot_time_comparison(_warm_run, _ip_run, name, origin=_origin)
            except Exception as _pe:
                print(f"(time-comparison plot failed: {_pe})")

        for _sname in SOLVERS:
            if _sname in ("Warm-IP", "IP"):
                continue                     # handled above
            print(f"Solving with {_sname}")
            try:
                out = run_external(_sname, name, data_dir, canon)
                results[f'{_sname}_time'] = out['time']
                record_solution(results, canon, _sname, out['x'],
                                out.get('dual'), out.get('v'))
                print(f">>> {_sname} time: {out['time']:.3f} s")
            except Exception as e:
                print(f"{_sname} error in {name}: {e}")
                results[f'{_sname}_error'] = str(e)

        all_results.append(results)
        # Incremental checkpoint: rewrite the CSV after EVERY instance, so
        # a crash or interrupt costs at most the in-flight solve.
        pd.DataFrame(all_results).to_csv(CSV_PATH, index=False)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    df = benchmark_qp_solvers()
    df.to_csv(CSV_PATH, index=False)
    print("Benchmark complete. Results saved to results/MPC_solver_benchmark.csv.")
    print_runtime_summary(df)
    print_kkt_summary(df)
