"""
Benchmark of QP solvers on synthetic convex QP instances.

Instances are generated IN MEMORY (nothing is written to disk) exactly as
described in the paper's synthetic-benchmark section:

    Q = M'M + delta*I  (M random dense, delta = 1; positive definite)
    q = -Q x0          (x0 random: the unconstrained minimizer)
    A = sparse random (density 0.2),  b = A x0          (x0 feasible)
    G = sparse random (density 0.2),  h = G x0 + 0.01*|xi|  (small slack)

so every instance has both equality and inequality constraints and a known
strictly feasible point. The fixed SEED makes every instance bit-identical
to the ones behind the paper's synthetic experiments.

Each instance is solved with Warm-IP, IP, PIQP, MOSEK, Clarabel, OSQP and
Gurobi (equalities passed natively to the external solvers; EQ_MODE
selects the Warm-IP/IP treatment). Runtime, objective, iteration counts,
stop reasons and normalized KKT quality (data_loader.kkt_metrics) go to
results/Synthetic_solver_benchmark.csv; all figures are regenerated
from the CSV alone by generate_paper_figures.py.
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

from data_loader import kkt_metrics
from warm_ip_solver import solve_qp
from generate_paper_figures import print_runtime_summary, print_kkt_summary

# ---------------------------------------------------------------------------
# Problem sizes (n, m_eq, m_ineq): the paper's 21 synthetic
# configurations.
# ---------------------------------------------------------------------------
PROBLEM_CONFIGS = [
    # Small
    (10, 10, 1000),
    (100, 10, 1000),
    (1000, 10, 1000),
    (100, 10, 10000),
    (1000, 10, 10000),
    (5000, 10, 10000),

    # Medium
    (100, 10, 50000),
    (1000, 10, 50000),
    (10000, 10, 50000),
    (100, 10, 500000),
    (1000, 10, 500000),
    (2000, 10, 500000),
    # Large
    (5000, 10, 500000),
    (10000, 10, 500000),
    (20000, 10, 500000),
    (100, 10, 1000000),
    (1000, 10, 1000000),
    (2000, 10, 1000000),
    (3000, 10, 1000000),
    (5000, 10, 1000000),
    (10000, 10, 1000000),
]

# Which configs to run: "all", or a list of 1-based indices into
# PROBLEM_CONFIGS (e.g. [1, 7, 18]).
PROBLEMS = "all"




# RNG seed used for EVERY instance; with this value the generated
# problems are bit-identical to the paper's.
SEED = 100

# ---------------------------------------------------------------------------
# Difficulty knobs (both default to 0.0 = the baseline generator used in
# the paper; the RNG stream is untouched at the defaults, so baseline
# instances stay bit-identical to the paper's).
#
# KAPPA -- conditioning parameter. kappa > 0 applies log-uniform scalings
#   spanning 10^kappa: the columns of M are scaled by 10^(kappa*U[0,1])
#   before forming Q = M'M + delta*I (spectrum spread ~10^(2*kappa)), and
#   the rows of A and G are scaled by 10^(kappa*U[-1,1]) (the row-scaling
#   pathology that Ruiz equilibration targets). b and h are computed AFTER
#   the scaling, so x0 stays feasible.
#
# ACTIVE_FRAC -- fraction of inequality rows made ACTIVE at the solution.
#   The baseline construction (q = -Q x0, x0 strictly feasible) makes x0
#   the unconstrained minimizer, so every instance is solved with all
#   constraints inactive and z* = 0. With active_frac > 0, a random subset
#   S of ceil(active_frac * m_ineq) rows is made tight (h_S = G_S x0),
#   multipliers z_S = 10^(kappa*U[0,1]) > 0 and v0 ~ N(0,1) are drawn, and
#   q = -(Q x0 + G_S' z_S + A' v0), so (x0, z, v0) is the EXACT KKT
#   solution with a genuinely active set (and dual spread tied to kappa).
# ---------------------------------------------------------------------------
KAPPA = 1
ACTIVE_FRAC = 0.0

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

# Per-instance convergence and time-comparison plots (off by default).
MAKE_PLOTS = False

# Save generated figures (.png / .pdf) to results/.
SAVE_FIGS = False

# A solver counts as FAILED on a problem when it returns no result, exposes
# no multipliers, or max(primal_res, dual_res, dual_gap) exceeds this.
FAIL_KKT_TOL = 1e-3

# ---------------------------------------------------------------------------
# Linear-solver backend for the Warm-IP / IP runs (case-insensitive).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Equality handling in the Warm-IP / IP runs ('direct' | 'absorb'); the
# external solvers always receive the equalities natively.
# ---------------------------------------------------------------------------
EQ_MODE = 'absorb'

# ---------------------------------------------------------------------------
# Fault isolation for the Warm-IP / IP runs (see run_benchmark_MM.py).
# The isolated child REGENERATES the instance from (n, m_eq, m_ineq, SEED)
# -- generation is deterministic and cheap, so no data ever hits the disk.
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

# All program outputs (plots, CSV) are written here
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, 'Synthetic_solver_benchmark.csv')

# Set MKL threads
num_threads_to_use = int(psutil.cpu_count(logical=False) * 0.45)
smkl.mkl_set_num_threads(num_threads_to_use)
smkl.mkl_set_num_threads_local(num_threads_to_use)

eps_overall = 1e-6   # single accuracy knob shared by all solvers

# Configs (MM template)
ADMM_config = {
    'rho': 0.5,
    'mu': 'auto',  # ADMM barrier weight; None -> variant default
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


def instance_name(idx, n, m_eq, m_ineq):
    name = f"SYN_{idx:03d}_n{n}_meq{m_eq}_mineq{m_ineq}"
    if KAPPA:
        name += f"_k{KAPPA:g}"
    if ACTIVE_FRAC:
        name += f"_a{ACTIVE_FRAC:g}"
    return name


def generate_qp(n, m_eq, m_ineq, seed, kappa=0.0, active_frac=0.0):
    """Generate one synthetic instance. With kappa = active_frac = 0 this
    reproduces the paper's instances bit-identically (fixed seed handling
    and RNG call order: M -> x0 -> A -> G/h; the knobs draw extra random
    numbers ONLY when nonzero, leaving the baseline stream untouched).

    kappa > 0: log-uniform column scaling of M (Q spectrum spread
    ~10^(2*kappa)) and row scaling of A and G over 10^(kappa*[-1,1]);
    b and h are computed after scaling, so x0 remains feasible.

    active_frac > 0: a random subset S of ceil(active_frac*m_ineq)
    inequality rows is made tight at x0 with multipliers
    z_S = 10^(kappa*U[0,1]) and equality multipliers v0 ~ N(0,1); q is set
    so that (x0, z, v0) is the EXACT KKT solution (active set S, dual
    spread tied to kappa). With active_frac = 0 the solution is x0 with
    all constraints inactive (z* = 0, v* = 0).

    Returns (Q, q, c, G, h, A, b) with Q in CSR form.
    """
    rng = np.random.default_rng(seed)  # once, at the top of generate_qp



    np.random.seed(seed)

    # positive definite Q = M'M + delta*I (paper: delta > 0)
    M = sp.random(n, n, density=0.01, format='csr', random_state=rng)
    if kappa:
        # log-uniform column scales: heterogeneous variable scaling and a
        # spectrum spanning ~10^(2*kappa)
        M = (M @ sp.diags(10.0 ** (kappa * np.random.rand(n)), format='csr')).tocsr()
    delta = 1
    # keep P SPARSE: adding a dense np.eye to a sparse product would
    # produce an np.matrix (q would come out with shape (1, n))
    P = (dot_mkl(sp.csr_matrix(M.T), M)
         + delta * sp.identity(n, format='csr')).tocsr()

    # x0: feasible reference point (and, with active_frac = 0, the
    # unconstrained minimizer)
    x0 = np.random.randn(n)
    q = -dot_mkl(P, x0)

    # equality block: A x = b holds at x0
    A = sp.random(m_eq, n, density=0.01, format='csr', random_state=rng)
    if kappa:
        r_eq = 10.0 ** (kappa * (2.0 * np.random.rand(m_eq) - 1.0))
        A = sp.diags(r_eq, format='csr') @ A
    b = dot_mkl(A, x0)

    # inequality block: G x0 <= h with a small positive slack
    G = sp.random(m_ineq, n, density=0.001, format='csr', random_state=rng)
    if kappa:
        r_in = 10.0 ** (kappa * (2.0 * np.random.rand(m_ineq) - 1.0))
        G = sp.diags(r_in, format='csr') @ G
    h = dot_mkl(G, x0) + 1 * np.abs(np.random.rand(m_ineq))

    if active_frac:
        # make a random subset ACTIVE at the solution with known duals:
        # h_S tight, z_S > 0, and stationarity built into q
        n_act = max(1, int(np.ceil(active_frac * m_ineq)))
        S = np.random.choice(m_ineq, size=n_act, replace=False)
        Gx0 = dot_mkl(G, x0)
        h[S] = Gx0[S]                        # tight rows
        z_S = 10.0 ** (kappa * np.random.rand(n_act))   # positive duals
        v0 = np.random.randn(m_eq)
        G_S = sp.csr_matrix(G)[S, :]
        q = -(dot_mkl(P, x0) + dot_mkl(G_S.T, z_S) + dot_mkl(A.T, v0))
        # exact KKT point: x* = x0, z*_S = z_S (0 elsewhere), v* = v0

    Q = sp.csr_matrix(P)
    A = sp.csr_matrix(A)
    A.sort_indices()
    G = sp.csr_matrix(G)
    G.sort_indices()
    return Q, np.asarray(q), 0.0, G, np.asarray(h), A, np.asarray(b)


def selected_configs():
    if PROBLEMS == "all":
        idxs = range(1, len(PROBLEM_CONFIGS) + 1)
    else:
        idxs = PROBLEMS
    out = []
    for i in idxs:
        if not 1 <= i <= len(PROBLEM_CONFIGS):
            raise ValueError(f"config index {i} out of range 1.."
                             f"{len(PROBLEM_CONFIGS)}")
        out.append((i,) + PROBLEM_CONFIGS[i - 1])
    return out


def objective_value(x, Q, q, c):
    return 0.5 * dot_mkl(x.T, dot_mkl(Q, x)) + dot_mkl(q.T, x) + c


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



def _warmip_worker(out_queue, cfg, config, eq_mode_):
    """Child-process entry point for one isolated Warm-IP / IP solve:
    REGENERATE the instance from its config (deterministic, nothing on
    disk), solve, ship the results back. Runs in a fresh interpreter
    (spawn), so a native PARDISO/MKL crash kills only this process."""
    try:
        n, m_eq, m_ineq = cfg
        Q, q, c, G, h, A_eq, b_eq = generate_qp(n, m_eq, m_ineq, SEED,
                                                KAPPA, ACTIVE_FRAC)
        t0 = time.time()
        x, info, admm_time, info_admm = solve_qp(
            Q, q, G, h, A_eq, b_eq, **config, eq_mode=eq_mode_)
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


def run_warmip(cfg, config, solver_args):
    """One Warm-IP / IP solve; isolated in a child process when
    ISOLATE_WARMIP is set. Returns (x, info, admm_time, info_admm, elapsed)
    or raises RuntimeError describing the failure (exception, native crash,
    or timeout) so the caller's try/except records it and moves on."""
    if not ISOLATE_WARMIP:
        t0 = time.time()
        x, info, admm_time, info_admm = solve_qp(
            *solver_args, **config, eq_mode=EQ_MODE)
        return x, info, admm_time, info_admm, time.time() - t0
    return _run_isolated(_warmip_worker, (cfg, config, EQ_MODE))


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


def _external_worker(out_queue, solver_name, cfg):
    """Child-process entry point for one isolated external solve:
    REGENERATE the instance from its config (deterministic, nothing on
    disk), solve, ship {'time','x','dual','v'} back through the queue. A
    native crash inside the solver library kills only this process."""
    try:
        n, m_eq, m_ineq = cfg
        Q, q, c, G, h, A_eq, b_eq = generate_qp(n, m_eq, m_ineq, SEED,
                                                KAPPA, ACTIVE_FRAC)
        out_queue.put(('ok', _solve_external(
            solver_name, Q, q, c, G, h, A_eq, b_eq)))
    except Exception as e:
        out_queue.put(('error', f"{type(e).__name__}: {e}"))


def run_external(solver_name, cfg, canon):
    """One external solve; isolated in a child process when
    ISOLATE_EXTERNAL is set (native crashes and SOLVER_TIMEOUT then only
    blank this solver's columns). Returns {'time','x','dual','v'}."""
    if not ISOLATE_EXTERNAL or solver_name in ISOLATE_SKIP:
        return _solve_external(solver_name, canon['Q'], canon['q'],
                               canon['c'], canon['G'], canon['h'],
                               canon.get('A'), canon.get('b'))
    return _run_isolated(_external_worker, (solver_name, cfg))


def benchmark_qp_solvers():
    all_results = []
    configs = selected_configs()
    print(f"{len(configs)} synthetic config(s) selected (seed {SEED})")

    for idx, n, m_eq, m_ineq in configs:
        name = instance_name(idx, n, m_eq, m_ineq)
        print("=" * 79)
        print(f"Generating {name} (n={n}, m_eq={m_eq}, m_ineq={m_ineq}, "
              f"seed {SEED}, kappa={KAPPA:g}, active_frac={ACTIVE_FRAC:g})")
        t_gen = time.time()
        Q, q, c, G, h, A_eq, b_eq = generate_qp(n, m_eq, m_ineq, SEED,
                                                KAPPA, ACTIVE_FRAC)
        print(f"generated in {time.time() - t_gen:.1f} s "
              f"(nnz Q {Q.nnz}, nnz G {G.nnz})")
        canon = {'Q': Q, 'q': q, 'c': c, 'G': G, 'h': h,
                 'A': A_eq, 'b': b_eq}
        m_ineq_rows = G.shape[0]
        p_eq = A_eq.shape[0]
        has_eq = True
        pure_eq = False
        results = {"Problem": name, "n": n, "m": m_ineq_rows, "p": p_eq,
                   "pure_equality": 0}
        _warm_run = None
        _ip_run = None

        print("Solving", name)

        solver_args = (Q, q, G, h, A_eq, b_eq)
        cfg = (n, m_eq, m_ineq)

        def _ip_duals(info):
            """(z, v) of the split rows, from any warm_ip path."""
            if not isinstance(info, dict):
                return None, None
            return info.get('z'), info.get('v')

        if "Warm-IP" in SOLVERS:
            print("Solving with Warm-IP")
            try:
                x_final, info, admm_time, info_admm, _elapsed = run_warmip(
                    cfg, WARM_IP_config, solver_args)
                results['Warm-IP_time'] = _elapsed
                results['Warm-IP_iters'] = len(info.get(
                    'objective_original', [])) if isinstance(info, dict) else None
                results['Warm-IP_stop'] = info.get(
                    'stop_reason') if isinstance(info, dict) else None
                _z, _v = _ip_duals(info)
                record_solution(results, canon, 'Warm-IP', x_final, _z, _v)
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
                    run_warmip(cfg, IP_config, solver_args)
                results['IP_time'] = _elapsed
                results['IP_iters'] = len(info_ip.get(
                    'objective_original', [])) if isinstance(info_ip, dict) else None
                results['IP_stop'] = info_ip.get(
                    'stop_reason') if isinstance(info_ip, dict) else None
                _z, _v = _ip_duals(info_ip)
                record_solution(results, canon, 'IP', x_final_ip, _z, _v)
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
                _rhs = np.concatenate([h, b_eq])
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
                out = run_external(_sname, cfg, canon)
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
    print("Benchmark complete. Results saved to "
          "results/Synthetic_solver_benchmark.csv.")
    print_runtime_summary(df)
    print_kkt_summary(df)
