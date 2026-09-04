"""
Benchmark of QP solvers on the public IMRT radiotherapy instances.

Loads every canonical instance (min 0.5 x'Qx + q'x + c s.t. Gx <= h) of
the IMRT_lung family (data/IMRT_lung_data/) through data_loader, solves
it with Warm-IP (path-following ADMM warm start + interior-point method), the same interior-point method
cold-started (IP), PIQP, MOSEK, Clarabel, OSQP, and Gurobi, and reports for
each solver: runtime, objective value, and the normalized KKT quality of the
returned solution (primal_res, dual_res, dual_gap, and their maximum), all
computed with the same metric (data_loader.kkt_metrics). Results go
to results/IMRT_solver_benchmark.csv; figures are regenerated from the
CSV by generate_paper_figures.py.

All solvers are configured with the same termination tolerance eps_overall
to the extent their parameter sets allow, so the accuracy of the final
solutions is comparable. Note: Gurobi and MOSEK require licenses; PIQP,
OSQP, and Clarabel are free. Each solver runs in its own try/except, so a
missing solver or license only blanks that solver's columns.
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

from data_loader import (kkt_metrics, load_data, select_instances,
                         ensure_local)
from warm_ip_solver import solve_qp
from generate_paper_figures import print_runtime_summary, print_kkt_summary

# ---------------------------------------------------------------------------
# Which problems to run: "all", or a list of entries. An entry may be an
# instance number (1, 16 -> IMRT_lung_001, IMRT_lung_016) or a full name
# ("IMRT_lung_001").
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

# Per-instance convergence and time-comparison plots. With all instances
# selected this writes many PNGs per run, so it is off by default.
MAKE_PLOTS = False

# ---------------------------------------------------------------------------
# Linear-solver backend for the Warm-IP / IP runs: MKL PARDISO.
# ---------------------------------------------------------------------------
# All program outputs (plots, CSV) are written here
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

# Save generated figures (.pdf) from the shared plotting module
SAVE_FIGS = False
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, 'IMRT_solver_benchmark.csv')


# ---------------------------------------------------------------------------
# Fault isolation for the Warm-IP / IP runs. PARDISO/MKL errors can kill the
# Python process outright (segfault / abort), which no try/except can catch;
# with ISOLATE_WARMIP = True each Warm-IP / IP solve runs in its own child
# process, so a native crash -- or exceeding SOLVER_TIMEOUT (seconds,
# None = no limit) -- only blanks that solver's columns and the benchmark
# continues with the next solver/problem. Timings are measured inside the
# child (process startup is excluded). Set False to run in-process
# (easier debugging).
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
SOLVER_TIMEOUT = 3600     # per-solver limit (s), applied to EVERY isolated solve

# A solver counts as FAILED on a problem when it returns no result, exposes
# no multipliers, or max(primal_res, dual_res, dual_gap) exceeds this.
FAIL_KKT_TOL = 1e-3

# Set MKL threads
num_threads_to_use = int(psutil.cpu_count(logical=False)*0.45)
smkl.mkl_set_num_threads(num_threads_to_use)
smkl.mkl_set_num_threads_local(num_threads_to_use)

eps_overall = 1e-6   # single accuracy knob shared by all solvers

# Configs
ADMM_config = {
    'rho': 0.5,
    'mu': 'auto',   # ADMM barrier weight; None -> variant default (0.01*eps_res)
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
    'ruiz': False,  # Ruiz equilibration preprocessing (OSQP algorithm)
}

IP_config = {
    'max_iter': 100,
    'tol': eps_overall,
    'warm_start': False,
    'admm_config': ADMM_config,
    'ruiz': False,  # Ruiz equilibration preprocessing (OSQP algorithm)
}


def objective_value(x, Q, q, c):
    return 0.5 * dot_mkl(x.T, dot_mkl(Q, x)) + dot_mkl(q.T, x) + c


def _warmip_worker(out_queue, name, data_dir, config):
    """Child-process entry point for one isolated Warm-IP / IP solve: load
    the instance, solve, ship the results back through the queue. Runs in a
    fresh interpreter (spawn), so a native PARDISO/MKL crash kills only
    this process, not the benchmark."""
    try:
        prob = load_data(name, data_dir)
        Q, q, G, h = prob['Q'], prob['q'], prob['G'], prob['h']
        t0 = time.time()
        x, info, admm_time, info_admm = solve_qp(
            Q, q, G, h, None, None, **config)
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


def run_warmip(name, data_dir, config, prob):
    """One Warm-IP / IP solve; isolated in a child process when
    ISOLATE_WARMIP is set. Returns (x, info, admm_time, info_admm, elapsed)
    or raises RuntimeError describing the failure (exception, native crash,
    or timeout) so the caller's try/except records it and moves on."""
    if not ISOLATE_WARMIP:
        t0 = time.time()
        x, info, admm_time, info_admm = solve_qp(
            prob['Q'], prob['q'], prob['G'], prob['h'], None, None,
            **config)
        return x, info, admm_time, info_admm, time.time() - t0
    return _run_isolated(_warmip_worker, (name, data_dir, config))


def _solve_external(solver_name, Q, q, G, h):
    """One external-solver run (PIQP / MOSEK / Clarabel / OSQP / Gurobi) in
    whatever process this executes in; IMRT instances are inequality-only.
    Returns {'time', 'x', 'dual'} (dual may be None). Raises on errors."""
    n = Q.shape[0]

    if solver_name == "PIQP":
        import piqp
        solver = piqp.SparseSolver()
        solver.settings.verbose = True
        solver.settings.eps_abs = eps_overall
        solver.settings.eps_rel = eps_overall
        # PIQP also terminates on the duality gap, with much tighter
        # defaults than eps_abs/eps_rel; align them with eps_overall.
        for _gap_attr in ("eps_duality_gap_abs", "eps_duality_gap_rel"):
            if hasattr(solver.settings, _gap_attr):
                setattr(solver.settings, _gap_attr, eps_overall)
        solver.settings.max_iter = 250
        solver.setup(P=sp.csc_matrix(Q), c=q, A=None, b=None,
                     G=sp.csc_matrix(G),
                     h_l=np.full(G.shape[0], -np.inf), h_u=h)
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
        return {'time': elapsed, 'x': np.asarray(res.x), 'dual': z_piqp}

    if solver_name == "MOSEK":
        import cvxpy as cp
        x = cp.Variable(n)
        constraints = [G @ x <= h]
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
        z_mosek = constraints[0].dual_value
        return {'time': elapsed, 'x': np.asarray(x.value),
                'dual': np.asarray(z_mosek) if z_mosek is not None else None}

    if solver_name == "Clarabel":
        import clarabel
        Q_bar = sparse.triu(Q, format="csc")
        A_full = sp.csc_matrix(G)
        b_full = h
        cones = [clarabel.NonnegativeConeT(len(b_full))]
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
        return {'time': elapsed, 'x': np.asarray(solution.x),
                'dual': np.asarray(solution.z)}

    if solver_name == "OSQP":
        import osqp
        prob_osqp = osqp.OSQP()
        prob_osqp.setup(sp.csc_matrix(Q), q, sp.csc_matrix(G),
                        -np.inf * np.ones(h.shape), h,
                        eps_abs=eps_overall, eps_rel=eps_overall,
                        max_iter=4000, verbose=True)
        t0 = time.time()
        res = prob_osqp.solve()
        elapsed = time.time() - t0
        return {'time': elapsed, 'x': res.x, 'dual': res.y}

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
        model.addConstr(G @ x <= h, name="ineq")
        t0 = time.perf_counter()
        model.optimize()
        elapsed = time.perf_counter() - t0
        try:
            z_grb = -np.asarray(model.getAttr("Pi", model.getConstrs()))
        except Exception:
            z_grb = None
        return {'time': elapsed, 'x': np.asarray(x.X), 'dual': z_grb}

    raise ValueError(f"unknown external solver {solver_name!r}")


def _external_worker(out_queue, solver_name, name, data_dir):
    """Child-process entry point for one isolated external solve: load the
    instance, solve, ship {'time','x','dual'} back through the queue. A
    native crash inside the solver library kills only this process."""
    try:
        prob = load_data(name, data_dir)
        out_queue.put(('ok', _solve_external(
            solver_name, prob['Q'], prob['q'], prob['G'], prob['h'])))
    except Exception as e:
        out_queue.put(('error', f"{type(e).__name__}: {e}"))


def run_external(solver_name, name, data_dir, prob):
    """One external solve; isolated in a child process when
    ISOLATE_EXTERNAL is set (native crashes and SOLVER_TIMEOUT then only
    blank this solver's columns). Returns {'time','x','dual'}."""
    if not ISOLATE_EXTERNAL or solver_name in ISOLATE_SKIP:
        return _solve_external(solver_name, prob['Q'], prob['q'],
                               prob['G'], prob['h'])
    return _run_isolated(_external_worker, (solver_name, name, data_dir))


def record_solution(results, prob, solver_name, x, dual):
    """Store objective and KKT quality columns for one solver's solution.
    dual may be None (solver exposed no multipliers): KKT columns stay blank
    except primal_res, which only needs x."""
    Q, q, c = prob['Q'], prob['q'], prob['c']
    results[f'{solver_name}_obj'] = objective_value(x, Q, q, c)
    if dual is not None:
        primal_res, dual_res, dual_gap = kkt_metrics(prob, x, dual)
        results[f'{solver_name}_primal_res'] = primal_res
        results[f'{solver_name}_dual_res'] = dual_res
        results[f'{solver_name}_dual_gap'] = dual_gap
        results[f'{solver_name}_kkt_max'] = max(primal_res, dual_res, dual_gap)
        print(f">>> {solver_name} KKT: primal_res={primal_res:.2e} "
              f"dual_res={dual_res:.2e} dual_gap={dual_gap:.2e} "
              f"(max {max(primal_res, dual_res, dual_gap):.2e})")
    else:
        G, h = prob['G'], prob['h']
        Gx = G @ x
        primal_res = np.linalg.norm(np.maximum(Gx - h, 0.0), np.inf) / (
            1.0 + max(np.linalg.norm(Gx, np.inf), np.linalg.norm(h, np.inf)))
        results[f'{solver_name}_primal_res'] = primal_res
        print(f">>> {solver_name} KKT: primal_res={primal_res:.2e} "
              f"(no dual solution exposed; dual_res/dual_gap not available)")



def plot_warmip_convergence(info, info_admm, instance, solver_name):
    """One log-scale figure per run: ADMM primal/dual residuals (recorded
    every num_itr_check_convergence iterations) followed by the IP stage's
    primal/dual/complementarity residuals (every iteration). Saved as
    warmip_convergence_<solver>_<instance>.png next to this script; set
    WARMIP_NOSHOW=1 to skip the interactive window."""
    import matplotlib
    if os.environ.get("WARMIP_NOSHOW") == "1":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    check = ADMM_config['num_itr_check_convergence']
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
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    print(f'convergence plot saved to {fname}', flush=True)
    if os.environ.get("WARMIP_NOSHOW") != "1":
        plt.show()
    plt.close(fig)



def plot_time_comparison(warm_run, ip_run, instance, origin=None):
    """When both Warm-IP and IP were run on an instance: max residual vs
    wall-clock time on a log scale, one curve per method. For Warm-IP the
    star marker shows where the algorithm switches from ADMM to IP.
    ADMM logs primal/dual residuals only, so its part of the curve is
    max(primal, dual); the IP part is max(primal, dual, complementarity).
    Saved as residual_vs_time_<instance>.png; WARMIP_NOSHOW=1 skips the
    interactive window."""
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
        # Both methods start from the same state (the origin, x = s = z = 0),
        # so both curves start at the same value at t = 0. The rise of the
        # cold-IP curve to its first iterate shows the complementarity the
        # cold initializer manufactures to obtain an interior point.
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
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    print(f'time-comparison plot saved to {fname}', flush=True)
    if os.environ.get("WARMIP_NOSHOW") != "1":
        plt.show()
    plt.close(fig)


def benchmark_qp_solvers(data_dir=None):
    all_results = []

    names = select_instances(PROBLEMS, data_dir, prefix="IMRT_lung_")
    ensure_local(names, data_dir)   # prefetch any missing instances
    print(f"{len(names)} problem(s) selected")

    for name in names:
        prob = load_data(name, data_dir)
        Q, q, c, G, h = prob['Q'], prob['q'], prob['c'], prob['G'], prob['h']
        n = Q.shape[0]
        results = {"Problem": name}
        _warm_run = None
        _ip_run = None

        print("=" * 79)
        print("Solving", name, f"(n={n}, m={G.shape[0]})")

        if "Warm-IP" in SOLVERS:
            print("Solving with Warm-IP")
            try:
                x_final, info, admm_time, info_admm, _elapsed = run_warmip(
                    name, data_dir, WARM_IP_config, prob)
                results['Warm-IP_time'] = _elapsed
                # IP iterations (one history entry per iteration)
                results['Warm-IP_iters'] = len(info.get(
                    'objective_original', [])) if isinstance(info, dict) else None
                # how the run ended: converged | stalled | diverged |
                # max_iter (see IP_SAFEGUARDS in warm_ip_solver)
                results['Warm-IP_stop'] = info.get(
                    'stop_reason') if isinstance(info, dict) else None
                record_solution(results, prob, 'Warm-IP', x_final,
                                info.get('z') if isinstance(info, dict) else None)
                print(f">>> Warm-IP time: {results['Warm-IP_time']:.3f} s")
                if MAKE_PLOTS:
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
                    run_warmip(name, data_dir, IP_config, prob)
                results['IP_time'] = _elapsed
                results['IP_iters'] = len(info_ip.get(
                    'objective_original', [])) if isinstance(info_ip, dict) else None
                results['IP_stop'] = info_ip.get(
                    'stop_reason') if isinstance(info_ip, dict) else None
                record_solution(results, prob, 'IP', x_final_ip,
                                info_ip.get('z') if isinstance(info_ip, dict) else None)
                print(f">>> IP (cold start) time: {results['IP_time']:.3f} s")
                if MAKE_PLOTS:
                    try:
                        plot_warmip_convergence(info_ip, info_admm_ip, name, 'IP')
                    except Exception as _pe:
                        print(f"(convergence plot failed: {_pe})")
                _ip_run = (info_ip, info_admm_ip, admm_time)
            except Exception as e:
                print(f"IP error in {name}: {e}")
                results['IP_error'] = str(e)

        if MAKE_PLOTS and _warm_run is not None and _ip_run is not None:
            try:
                _ph = float(np.linalg.norm(prob['h'], np.inf))
                _dq = float(np.linalg.norm(prob['q'], np.inf))
                # normalized residuals at the origin (x=s=z=0), matching the
                # scalar normalizations used inside the solvers
                _origin = max(_ph / (1.0 + _ph), _dq / (1.0 + _dq))
                plot_time_comparison(_warm_run, _ip_run, name, origin=_origin)
            except Exception as _pe:
                print(f"(time-comparison plot failed: {_pe})")

        for _sname in SOLVERS:
            if _sname in ("Warm-IP", "IP"):
                continue                     # handled above
            print(f"Solving with {_sname}")
            try:
                out = run_external(_sname, name, data_dir, prob)
                results[f'{_sname}_time'] = out['time']
                record_solution(results, prob, _sname, out['x'],
                                out.get('dual'))
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
    print("Benchmark complete. Results saved to results/IMRT_solver_benchmark.csv.")
    print_runtime_summary(df)
    print_kkt_summary(df)
