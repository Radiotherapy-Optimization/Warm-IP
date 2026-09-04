"""
Run Warm-IP on a single benchmark instance.

This is the entry point of the repository: pick any instance from any of
the benchmark families (Maros-Meszaros "MM_*", model-predictive control
"MPC_*", synthetic "Synthetic_*", or radiotherapy "IMRT_lung_*"), set the
options in the CONFIGURATION block below, and run

    python run_warm_ip.py

MM/MPC/IMRT instances are loaded through src/data_loader.py: they are
searched for in the local data/ folders first and, when not found,
downloaded automatically from the Hugging Face dataset

    https://huggingface.co/datasets/Radiotherapy-Optimization/QP-Benchmark

(requires the huggingface_hub package). Synthetic instances ("SYN_...")
have no data files: they are regenerated in memory from the fixed seed by
scripts/run_benchmark_Synthetic.py's generator, bit-identical to the
paper's instances. The script then solves the QP with
Warm-IP -- the path-following ADMM warm start followed by the primal-dual
interior-point method described in the paper -- using the paper's default
settings, and prints the runtime, the objective value, and the normalized
KKT optimality metrics (primal residual, dual residual, duality gap).

Notes
-----
* Maros-Meszaros instances are ill-conditioned; for them Ruiz
  equilibration is switched on automatically (as in the paper). All other
  families run without it. Set RUIZ below to True/False to override.
* Two-sided instances (MM) are converted to the split form
  Gx <= h, Ax = b; equality rows are absorbed into the inequality block as
  +/- pairs inside the solver (the paper's default, EQ_MODE = 'absorb').
  Equality-only instances have no interior and are solved in closed form.
* The linear-algebra backend is Intel MKL PARDISO (via the mkl package).
"""

# ---------------------------------------------------------------------------
# CONFIGURATION -- edit these and run the script
# ---------------------------------------------------------------------------

# Instance name: the simplified family-plus-index form or the full name,
# both work (simplified names are expanded through
# data/instances_metadata.csv):
#   "MM_002"    or  "MM_002_AUG2DC"       Maros-Meszaros
#   "MPC_001"   or  "MPC_001_LIPMWALK0"   model predictive control
#   "IMRT_lung_1"  or  "IMRT_lung_001"    radiotherapy (lung IMRT)
#   "SYN_001"                             synthetic (generated in memory
#                                         from the fixed seed; 001-021)
INSTANCE = "MPC_001"

# Termination tolerance on the normalized KKT metrics (paper: 1e-6)
TOL = 1e-6

# Ruiz equilibration: None = automatic (on for MM instances, off
# otherwise, as in the paper); or force with True / False
RUIZ = None

# Directory holding the .h5 file. None = the data/<Family>_data folders of
# this repository, downloading from Hugging Face when the file is missing.
DATA_DIR = None

# ---------------------------------------------------------------------------

import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

import numpy as np

from data_loader import load_data, to_split, kkt_metrics
from warm_ip_solver import solve_qp

# Paper-default Warm-IP settings (identical to the benchmark scripts)
ADMM_CONFIG = {
    'rho': 0.5,      # ADMM penalty parameter (adapted during the run; the
                     # rho update triggers only every convergence check)
    'mu': 'auto',    # barrier weight of the log-barrier-modified QP the
                     # ADMM follows (S Z e = mu e on its central path).
                     # 'auto': re-targeted at every convergence check so the
                     #   duality gap at ADMM exit lands just below the
                     #   residual level -- the paper-default choice;
                     # None: fixed at eps_res (the paper's mu = eps_ADMM);
                     # a float: fixed at that value.
    'eps_res': 5e-1, # eps_ADMM: ADMM stops once the normalized primal and
                     # dual residuals are below this (coarse on purpose --
                     # the IP phase does the high-accuracy work)
    'num_itr_check_convergence': 50,   # residuals, convergence and the rho
                     # update are evaluated every this many iterations
    'rho_update_threshold': 2,   # tau_rho: rho is rescaled only when the
                     # proposed factor falls outside [1/tau_rho, tau_rho]
    'max_iter': 200, # ADMM iteration cap (the IP phase starts from the
                     # best iterate reached)
    'verbose': 1,    # 1: per-check progress lines; 0: quiet
}
# How equality constraints Ax = b are passed to the solver on problems that
# also have inequalities (pure-equality problems always take a closed-form
# solve):
#   'absorb'  Ax = b enters the inequality block as the pair
#             Ax <= b, -Ax <= -b  (the paper's default)
#   'direct'  kept native via an augmented KKT system
EQ_MODE = 'absorb'


def _load_synthetic(name):
    """Regenerate a synthetic instance (they have no data files): SYN_<idx>
    picks the paper configuration, generated bit-identically from the
    fixed seed by the benchmark script's generator."""
    m = re.match(r'SYN_?(\d+)', name, re.IGNORECASE)
    if not m:
        raise SystemExit(f"unrecognized synthetic instance {name!r}: use "
                         f"'SYN_<index>', e.g. 'SYN_001'")
    sys.path.insert(0, os.path.join(_HERE, 'scripts'))
    import run_benchmark_Synthetic as syn
    idx = int(m.group(1))
    if not 1 <= idx <= len(syn.PROBLEM_CONFIGS):
        raise SystemExit(f"synthetic index {idx} out of range: 1..."
                         f"{len(syn.PROBLEM_CONFIGS)}")
    n, m_eq, m_ineq = syn.PROBLEM_CONFIGS[idx - 1]
    full = syn.instance_name(idx, n, m_eq, m_ineq)
    print(f"Generating {full} (n={n}, m_eq={m_eq}, m_ineq={m_ineq}, "
          f"seed {syn.SEED}) ...")
    Q, q, c, G, h, A_eq, b_eq = syn.generate_qp(
        n, m_eq, m_ineq, syn.SEED, syn.KAPPA, syn.ACTIVE_FRAC)
    return full, (Q, q, c, G, h, A_eq, b_eq)


def main():
    ruiz = RUIZ if RUIZ is not None else INSTANCE.startswith('MM_')

    if INSTANCE.upper().startswith('SYN'):
        name, (Q, q, c, G, h, A_eq, b_eq) = _load_synthetic(INSTANCE)
    else:
        name = INSTANCE
        print(f"Loading {INSTANCE} ...")
        prob = load_data(INSTANCE, DATA_DIR)
        Q, q, c, G, h, A_eq, b_eq = to_split(prob)
    n, m_ineq = Q.shape[0], G.shape[0]
    p_eq = A_eq.shape[0] if A_eq is not None else 0
    print(f"n = {n} variables, m = {m_ineq} inequalities, "
          f"p = {p_eq} equalities; ruiz = {ruiz}")

    config = {
        'max_iter': 100,
        'tol': TOL,
        'warm_start': True,
        'admm_config': ADMM_CONFIG,
        'ruiz': ruiz,
    }

    t0 = time.time()
    x, info, admm_time, info_admm = solve_qp(
        Q, q, G if m_ineq else None, h if m_ineq else None, A_eq, b_eq,
        **config,
        eq_mode=EQ_MODE)
    elapsed = time.time() - t0

    obj = float(0.5 * x @ (Q @ x) + q @ x + c)
    z = info.get('z') if isinstance(info, dict) else None
    v = info.get('v') if isinstance(info, dict) else None
    canon = {'Q': Q, 'q': q, 'c': c, 'G': G, 'h': h}
    iters = (len(info.get('objective_original', []))
             if isinstance(info, dict) else None)
    stop = info.get('stop_reason') if isinstance(info, dict) else None

    print()
    print("=" * 60)
    print(f"instance        : {name}")
    print(f"total time      : {elapsed:.3f} s"
          + (f"  (ADMM warm start: {admm_time:.3f} s)"
             if admm_time else ""))
    print(f"IP iterations   : {iters}   (stop: {stop})")
    print(f"objective value : {obj:.10g}")
    if z is not None:
        pr, dr, gap = kkt_metrics(
            canon, x, z if m_ineq else np.zeros(0), A=A_eq, b=b_eq, v=v)
        print(f"primal residual : {pr:.3e}")
        print(f"dual residual   : {dr:.3e}")
        print(f"duality gap     : {gap:.3e}")
        print(f"KKT max         : {max(pr, dr, gap):.3e}"
              f"   (tolerance {TOL:g})")
    else:
        print("(solver returned no multipliers; KKT metrics unavailable)")
    print("=" * 60)


if __name__ == '__main__':
    main()
