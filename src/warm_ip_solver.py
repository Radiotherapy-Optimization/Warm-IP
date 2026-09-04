"""
Warm-IP: path-following ADMM warm start for a primal-dual interior-point
QP solver (reference implementation of the paper).

Solves   min 0.5 x'Qx + c'x   s.t.  Gx <= h  (and optionally Ax = b)
via solve_qp(): a path-following ADMM (path_following_admm)
follows the central path of the log-barrier-modified QP at low accuracy --
its iterates satisfy S Z e = mu e by construction -- and hands a
well-centered, strictly feasible point to a Mehrotra predictor-corrector
interior-point method (interior_point) for high-accuracy refinement. Optional
Ruiz equilibration (OSQP's algorithm) and two equality-handling modes
(EQ_MODE below). Linear algebra: Intel MKL PARDISO via sparse_dot_mkl.
"""
import numpy as np
import scipy.sparse as sp
import time
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from sparse_dot_mkl import dot_product_mkl as dot_mkl
import psutil
from sparse_dot_mkl import pardisoinit, pardiso
import sparse_dot_mkl as smkl
from scipy.sparse import csc_matrix
########################################################################################################################
# MKL thread count: about half the physical cores works best in practice
num_threads_to_use = int(psutil.cpu_count(logical=False)*0.45)
smkl.mkl_set_num_threads(num_threads_to_use)
smkl.mkl_set_num_threads_local(num_threads_to_use)

# ---------------------------------------------------------------------------
# Equality-constraint handling in solve_qp() when a problem
# has BOTH inequalities (G, h) and equalities (A, b); override per call with
# eq_mode=...
#   'direct'  ADMM and IP handle Ax = b natively: the ADMM adds an
#             augmented-Lagrangian equality block (same rho, multiplier v),
#             the IP solves the augmented indefinite KKT system
#             [[Q + G'DG, A'], [A, 0]] (PARDISO mtype=-2).
#   'absorb'  Ax = b is rewritten as the inequality pair Ax <= b, -Ax <= -b
#             and stacked into G (the paper's default).
# Either way the returned history carries history['z'] (multipliers of the
# G rows) and history['v'] (multipliers of Ax = b). Pure-equality problems
# (no G) always take the closed-form path regardless of this setting.
EQ_MODE = 'absorb'
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# IP safeguards (interior_point). The iteration tracks the best iterate seen --
# smallest kkt_max = max(violation_res, dual_res, dual_gap), where
# violation_res is the slack-free constraint violation of x
# (_primal_viol_norm), matching the external kkt_metrics measure the
# benchmarks report -- snapshots (x, s, z, v) whenever it improves, and
# returns that snapshot on EVERY exit. Two early stops protect against
# wasted iterations:
#   stall:      best kkt_max improved by less than (1 - stall_ratio) over
#               the last stall_window iterations
#   divergence: current kkt_max exceeds divergence_factor * best kkt_max,
#               or a residual became NaN/inf
# history['stop_reason'] records how the run ended ('converged' | 'stalled'
# | 'diverged' | 'max_iter'), history['best_iteration'] /
# history['best_kkt_max'] which iterate was returned.
# ---------------------------------------------------------------------------

IP_SAFEGUARDS = {
    'enabled': True,
    'stall_window': 10,       # iterations without sufficient improvement
    'stall_ratio': 0.9,       # required: best_new < 0.9 * best_{W ago}
    'divergence_factor': 100.0,
    # The IP sometimes needs 20-30 iterations before it starts making
    # progress (e.g. re-centering after a rough warm start), so the STALL
    # check is suppressed for the first min_iterations iterations; the
    # divergence and non-finite checks (unambiguous failures) stay active
    # from iteration 1, and best-iterate tracking always runs.
    'min_iterations': 30,
}
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ruiz equilibration preprocessing (optional; enable per call with
# ruiz=True): modified Ruiz equilibration exactly as in OSQP's scale_data()
# (simultaneous KKT column norms, 1/sqrt, clamping, cost scaling).

# OSQP constants (osqp/constants.h: SCALING, MIN_SCALING, MAX_SCALING)
RUIZ_OSQP_CONFIG = {
    'iters': 10,
    'min_scaling': 1e-04,
    'max_scaling': 1e+04,
}


mtype = 1  # SPD (Symmetric Positive Definite) matrix
# Initialize PARDISO solver
pt, iparm_factor = pardisoinit(mtype)
# ---------------------------
# Factorization Configuration
# ---------------------------
iparm_factor[0] = 1          # Use iparm array defined by user
iparm_factor[1] = 3          # Parallel nested dissection
iparm_factor[3] = num_threads_to_use # Parallel nested dissection
iparm_factor[4] = 2          # Return permutation vector
iparm_factor[23] = 1         # Parallel factorization

iparm_factor[18] = 1  # pardiso reads only the upper triangle part

# ------------------------
# Solve-only Configuration
# ------------------------
iparm_solve = iparm_factor.copy()
iparm_solve[4] = 1               # reuse the permutation from factorization
#######################################################################################################################
# Defining the functions and initializing the variables
########################################################################################################################
def objective_original(x, Q , c):
    return (1 / 2) * dot_mkl(x.T, dot_mkl(Q, x)) + dot_mkl(c.T, x)

def initialize_primal_dual(Q, c, A, b, G, h):
    # Compute system matrices
    # Min 0.5x.TQx + c.Tx, s.t. Ax=b, Gx<=h
    # Intial: Min 0.5x.TQx + c.Tx+0.5*||Gx-h||, s.t. Ax=b
    # closed form solution
    Q_bar = Q + dot_mkl(G.T , G)
    rhs1 = -c + dot_mkl(G.T , h)
    m,n = G.shape
    if A is not None:
        # Saddle-point system [[Q_bar, A'], [A, 0]]: symmetric indefinite
        # with a zero (2,2) block.  mtype=1 applies no numerical pivoting
        # and hits zero pivots on that block, so it is solved exactly like
        # solve_equality_constrained_qp (PARDISO mtype=-2, Bunch-Kaufman,
        # upper triangle): [[Q_bar, A'],[A, 0]][x; v] = [rhs1; b] is the
        # equality-constrained QP  min 0.5 x'Q_bar x + (c - G'h)'x
        # s.t. Ax = b.
        x, v = solve_equality_constrained_qp(Q_bar, c - dot_mkl(G.T, h),
                                             A, b)
        # The permutation of the saddle pattern is not reusable for the IP
        # loop's matrices; the caller computes its own.
        perm_user = None
        # Recover z = Gx - h
        z = dot_mkl(G , x) - h
    else:
        KKT_reduced = csr_matrix(Q_bar)
        d_pardiso = pardiso(KKT_reduced, rhs1, pt, mtype, iparm_factor, phase=13, maxfct=1, mnum=1, msglvl=0)
        sol = d_pardiso[0]
        perm_user = d_pardiso[2]
        x = sol[:n]
        # Recover z = Gx - h
        z = dot_mkl(G , x) - h
        v = None

    # Compute s and z init
    alpha_p = np.max(z) if np.any(z < 0) else 0
    s_init = -z if alpha_p < 0 else -z + (1 + alpha_p) * np.ones_like(z)

    alpha_d = np.max(-z) if np.any(z < 0) else 0
    z_init = z if alpha_d < 0 else z + (1 + alpha_d) * np.ones_like(z)

    s = s_init
    z = z_init

    return x, v, s, z, perm_user

def max_step(x, dx, tau):
    """Compute maximum step size to keep x + alpha * dx > 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_vec = np.where(dx < 0, -x / dx, np.inf)
    alpha_max = np.min(alpha_vec)
    return min(1.0, tau * alpha_max)


def _primal_res_norm(Gx, s, h, scaling=None, Ax=None, b=None):
    """Normalized primal residual (paper, stopping-criteria remark):
    ||Gx + s - h||inf / (1 + max(||Gx||inf, ||s||inf, ||h||inf)), and with
    equality constraints present the max of that and the equality residual
    ||Ax - b||inf / (1 + max(||Ax||inf, ||b||inf)).

    With Ruiz scaling active the arguments are scaled quantities; mapping
    them back with E^-1 (EA^-1 for the equality rows) makes the test
    identical to the unscaled one (OSQP terminates on unscaled residuals
    by default)."""
    if scaling is not None:
        Einv = scaling['Einv']
        Gx, s, h = Einv * Gx, Einv * s, Einv * h
    scale = 1.0 + max(np.linalg.norm(Gx, np.inf), np.linalg.norm(s, np.inf),
                      np.linalg.norm(h, np.inf))
    res = np.linalg.norm((Gx + s - h) / scale, ord=np.inf)
    if Ax is not None:
        if scaling is not None:
            EAinv = scaling['EAinv']
            Ax, b = EAinv * Ax, EAinv * b
        scale_eq = 1.0 + max(np.linalg.norm(Ax, np.inf),
                             np.linalg.norm(b, np.inf))
        res = max(res, np.linalg.norm((Ax - b) / scale_eq, ord=np.inf))
    return res


def _dual_res_norm(Qx, Gtz, c, scaling=None, Atv=None):
    """Normalized dual residual Qx + G'z (+ A'v) + c (paper, stopping-
    criteria remark): infinity norm over 1 + max of the term norms.

    With Ruiz scaling active every term maps back with cost^-1 D^-1."""
    if Atv is None:
        Atv = 0.0 * Qx
    if scaling is not None:
        f = scaling['cinv'] * scaling['Dinv']
        Qx, Gtz, c, Atv = f * Qx, f * Gtz, f * c, f * Atv
    scale = 1.0 + max(np.linalg.norm(Qx, np.inf), np.linalg.norm(Gtz, np.inf),
                      np.linalg.norm(Atv, np.inf), np.linalg.norm(c, np.inf))
    return np.linalg.norm((Qx + Gtz + Atv + c) / scale, ord=np.inf)


def _primal_viol_norm(Gx, h, scaling=None, Ax=None, b=None):
    """Violation-based primal residual, matching data_loader.kkt_metrics:
    ||max(Gx - h, 0)||inf / (1 + max(||Gx||inf, ||h||inf)), and with
    equality constraints the max of that and
    ||Ax - b||inf / (1 + max(||Ax||inf, ||b||inf)).

    Unlike the slack residual of _primal_res_norm this contains no s
    anywhere: a large slack (e.g. from a large-mu warm start) can make the
    slack residual look small -- s inflates its normalization -- while the
    true constraint violation of x is poor.  This is the measure to judge
    the KKT quality of x if it were returned right now."""
    if scaling is not None:
        Einv = scaling['Einv']
        Gx, h = Einv * Gx, Einv * h
    res = np.linalg.norm(np.maximum(Gx - h, 0.0), ord=np.inf) / (
        1.0 + max(np.linalg.norm(Gx, np.inf), np.linalg.norm(h, np.inf)))
    if Ax is not None:
        if scaling is not None:
            EAinv = scaling['EAinv']
            Ax, b = EAinv * Ax, EAinv * b
        res = max(res, np.linalg.norm(Ax - b, ord=np.inf) / (
            1.0 + max(np.linalg.norm(Ax, np.inf),
                      np.linalg.norm(b, np.inf))))
    return res


# ---------------------------------------------------------------------------
# Ruiz equilibration preprocessing
# ---------------------------------------------------------------------------
def _abs_max_rows(M):
    """Infinity norm of every row of a sparse matrix."""
    A = csr_matrix(M, copy=True)
    A.data = np.abs(A.data)
    return np.asarray(A.max(axis=1).todense()).ravel()


def _abs_max_cols(M):
    """Infinity norm of every column of a sparse matrix."""
    A = csr_matrix(M, copy=True)
    A.data = np.abs(A.data)
    return np.asarray(A.max(axis=0).todense()).ravel()


def _limit_scaling(v, lo, hi):
    """OSQP's limit_scaling: values below the minimum are reset to 1.0 (not
    clipped), values above the maximum are clipped."""
    v = np.asarray(v, dtype=np.float64).copy()
    v[v < lo] = 1.0
    v[v > hi] = hi
    return v


def _ruiz_osqp(Q, q, G, h, iters, cfg):
    """Modified Ruiz equilibration, following OSQP's scale_data()."""
    lo, hi = cfg['min_scaling'], cfg['max_scaling']
    m, n = G.shape
    D = np.ones(n)
    E = np.ones(m)
    cost = 1.0
    Q = csr_matrix(Q, copy=True)
    G = csr_matrix(G, copy=True)
    q = np.array(q, dtype=np.float64)
    h = np.array(h, dtype=np.float64)

    for _ in range(iters):
        # column norms of the KKT matrix [P; A] and row norms of A
        d = np.maximum(_abs_max_cols(Q), _abs_max_cols(G))
        e = _abs_max_rows(G)
        d = 1.0 / np.sqrt(_limit_scaling(d, lo, hi))
        e = 1.0 / np.sqrt(_limit_scaling(e, lo, hi))

        Dm = diags(d, format='csr')
        Em = diags(e, format='csr')
        Q = csr_matrix(Dm @ Q @ Dm)
        G = csr_matrix(Em @ G @ Dm)
        q = d * q
        h = e * h
        D *= d
        E *= e

        # cost scaling
        c_tmp = max(float(np.mean(_abs_max_cols(Q))),
                    float(np.linalg.norm(q, np.inf)))
        c_tmp = float(_limit_scaling(np.array([c_tmp]), lo, hi)[0])
        c_tmp = 1.0 / c_tmp
        Q = Q * c_tmp
        q = c_tmp * q
        cost *= c_tmp

    return Q, q, G, h, D, E, cost, iters


def ruiz_equilibration(Q, q, G, h, iters=None, A=None, b=None):
    """Ruiz equilibration preprocessing of  min 0.5 x'Qx + q'x  s.t. Gx <= h
    (and optionally Ax = b).

    Returns (Qs, qs, Gs, hs, scaling) -- or, when A is given,
    (Qs, qs, Gs, hs, As, bs, scaling) -- where the scaled data satisfy

        Qs = cost * D Q D,   qs = cost * D q,   Gs = E G D,   hs = E h,
        As = EA A D,         bs = EA b

    and ``scaling`` carries D, E, cost (plus EA when A is given) and their
    inverses. Equality rows are equilibrated exactly like inequality rows
    (OSQP does the same: they are just rows of its constraint matrix): the
    row block passed to the core algorithm is the stack [G; A], whose row
    scaling is split back into E and EA afterwards. A solution of the scaled
    problem maps back with

        x = D x_scaled,   z = cost^-1 E z_scaled,   v = cost^-1 EA v_scaled

    (verified: the primal residual scales by E (EA), the dual residual by
    cost*D, so both vanish exactly when the originals do).

    """
    has_eq = A is not None and A.shape[0] > 0
    m = G.shape[0]
    if has_eq:
        G_run = sp.csr_matrix(sp.vstack([csr_matrix(G), csr_matrix(A)],
                                        format='csr'))
        h_run = np.concatenate([np.asarray(h, dtype=np.float64),
                                np.asarray(b, dtype=np.float64)])
    else:
        G_run, h_run = G, h
    n_it = RUIZ_OSQP_CONFIG['iters'] if iters is None else iters
    Qs, qs, Gs, hs, D, E, cost, used = _ruiz_osqp(
        Q, q, G_run, h_run, n_it, RUIZ_OSQP_CONFIG)

    if has_eq:
        Gs = csr_matrix(Gs)
        As, bs = csr_matrix(Gs[m:, :]), hs[m:]
        Gs, hs = csr_matrix(Gs[:m, :]), hs[:m]
        E, EA = E[:m], E[m:]
    else:
        As, bs, EA = None, None, np.zeros(0)
    scaling = {'D': D, 'E': E, 'cost': cost,
               'Dinv': 1.0 / D, 'Einv': 1.0 / E, 'cinv': 1.0 / cost,
               'EA': EA, 'EAinv': 1.0 / EA if EA.size else EA}
    print(f"[ruiz] equilibration: {used} sweep(s), "
          f"D in [{D.min():.3g}, {D.max():.3g}], "
          f"E in [{E.min():.3g}, {E.max():.3g}], cost {cost:.3g}"
          + (f", EA in [{EA.min():.3g}, {EA.max():.3g}]" if EA.size else ""),
          flush=True)
    if has_eq:
        return Qs, qs, Gs, hs, As, bs, scaling
    return Qs, qs, Gs, hs, scaling


def solve_equality_constrained_qp(Q, c, A, b):
    """Closed-form solution of  min 0.5 x'Qx + c'x  s.t.  Ax = b.

    Without inequality constraints there is no interior and no central path
    to follow: the optimality conditions are the single symmetric
    saddle-point system

        [Q  A'] [x]   [-c]
        [A  0 ] [v] = [ b],

    solved here in one PARDISO call with mtype = -2 (real symmetric
    indefinite, Bunch-Kaufman pivoting; only the upper triangle is passed).
    With A empty this reduces to Qx = -c.

    Returns (x, v) with v the multipliers of Ax = b (empty if A is empty).
    """
    n = Q.shape[0]
    c = np.asarray(c, dtype=np.float64)
    if A is None or A.shape[0] == 0:
        KKT = csr_matrix(Q)
        rhs = -c
        m_eq = 0
    else:
        A = csr_matrix(A)
        m_eq = A.shape[0]
        KKT = sp.bmat([[csr_matrix(Q), A.T], [A, None]], format='csr')
        rhs = np.concatenate([-c, np.asarray(b, dtype=np.float64)])
    # mtype = -2 reads the upper triangle only
    KKT_up = sp.triu(KKT, format='csr')
    KKT_up.sort_indices()
    mtype_eq = -2
    pt_eq, iparm_eq = pardisoinit(mtype_eq)
    iparm_eq[0] = 1              # use the iparm below
    iparm_eq[1] = 3              # parallel nested dissection
    iparm_eq[23] = 1             # parallel factorization
    out = pardiso(KKT_up, rhs, pt_eq, mtype_eq, iparm_eq,
                  phase=13, maxfct=1, mnum=1, msglvl=0)
    sol = np.asarray(out[0], dtype=np.float64)
    pardiso(KKT_up, rhs, pt_eq, mtype_eq, iparm_eq, phase=-1, maxfct=1,
            mnum=1, msglvl=0)
    return sol[:n], (sol[n:] if m_eq else np.zeros(0))


def _equality_only_result(Q, c, A, b, note=None):
    """Run the closed-form equality-only solve and package it like an IP run
    (same 4-tuple), so callers need no special case."""
    if note:
        print(note, flush=True)
    t0 = time.time()
    x, v = solve_equality_constrained_qp(Q, c, A, b)
    elapsed = time.time() - t0

    Qx = dot_mkl(csr_matrix(Q), x)
    if A is not None and A.shape[0] > 0:
        A = csr_matrix(A)
        Ax = dot_mkl(A, x)
        Atv = dot_mkl(A.T.tocsr(), v)
        b = np.asarray(b, dtype=np.float64)
        primal_residual = np.linalg.norm(Ax - b, np.inf) / (
            1.0 + max(np.linalg.norm(Ax, np.inf), np.linalg.norm(b, np.inf)))
    else:
        Atv = np.zeros_like(x)
        primal_residual = 0.0
    dual_residual = np.linalg.norm(Qx + c + Atv, np.inf) / (
        1.0 + max(np.linalg.norm(Qx, np.inf), np.linalg.norm(c, np.inf),
                  np.linalg.norm(Atv, np.inf)))
    objective = float(0.5 * dot_mkl(x.T, Qx) + dot_mkl(np.asarray(c).T, x))

    print(f"Equality-only problem solved in closed form in {elapsed:.3f} s: "
          f"objective {objective:.6f}, primal_res {primal_residual:.4e}, "
          f"dual_res {dual_residual:.4e}", flush=True)

    history = {"objective_original": [objective],
               "primal_residual": [primal_residual],
               "dual_residual": [dual_residual],
               "compl_res": [0.0],
               "time": [elapsed]}
    history["stop_reason"] = 'closed_form'
    history["z"] = np.zeros(0)   # no inequality multipliers
    history["v"] = v             # equality multipliers
    return x, history, 0.0, []


def path_following_admm(Q, c, G, h , A, b, rho, eps_res, max_iter, num_itr_check_convergence, rho_update_threshold, mu=None, verbose=1,
                 scaling=None):
    """Path-following ADMM for the log-barrier-modified QP (paper, Sec. on
    path-following ADMM):

        min  0.5 x'Qx + c'x - mu * sum_i log(s_i)   s.t.  Gx + s = h,
                                                          Ax = b,

    for  min 0.5 x'Qx + c'x  s.t. Gx <= h, Ax = b (A may be None).  With the
    update order x -> s -> z the iterates satisfy the central-path
    relation S Z e = mu e at every iteration and s > 0, z > 0 throughout
    (the equality block does not enter the complementarity identity), so the
    final point is a well-centered warm start for the interior-point phase.

    Args:
        Q, c, G, h: QP data (Q n x n CSR, c (n,), G m x n CSR, h (m,)).
        A, b: optional equality constraints Ax = b (A p x n CSR, b (p,));
            enforced EXACTLY in every x-update (paper x-update KKT system):
            the x-update solves the saddle-point system
            [[Q + rho*G'G, A'], [A, 0]] [x; v] = [G'(rho*(h-s) - z) - c; b],
            which also yields the equality multiplier v at each iteration.
        rho: initial ADMM penalty parameter (adapted at runtime).
        eps_res: tolerance for the normalized primal/dual residuals.
        max_iter: maximum number of iterations.
        num_itr_check_convergence: residuals, convergence and the rho update
            are evaluated every this many iterations.
        rho_update_threshold: tau_rho; rho is rescaled only when the proposed
            scale factor falls outside [1/tau_rho, tau_rho].
        mu: barrier weight.  None -> fixed eps_res (the paper's
            mu = eps_ADMM).
            'auto' -> adapted at every convergence check so that the
            relative duality gap at ADMM exit lands AT the primal/dual
            residual level (eps_res): since the iterates satisfy
            S Z e = mu e and p - d ~ s'z = m*mu, the target is
            mu = eps_res * (1 + max(|p|, |d|)) / m.  Residuals are the
            cheap part of the IP iteration and the gap homotopy the slow
            part, so the gap should not lag; the ADMM handoff is exactly
            centered at its mu, so a low gap is safe (too-low mu costs at
            most a few re-centering steps).  The handoff point then
            sits in the balanced infeasible-central-path neighborhood the
            IP phase wants.  mu enters only the s-update, so adapting it
            needs no refactorization, and the central-path identity holds
            with the current mu after every iteration.
        verbose: 0 silent, >= 1 print the iteration table.
        scaling: Ruiz scaling dict (D, E, cost and inverses) when the data
            passed in are equilibrated; residuals and the reported objective
            are then mapped back to the original scale.

    Returns:
        x, s, z: final iterates (s > 0, z > 0, S Z e = mu e).
        history: dict with keys objective_original, primal_residual,
            dual_residual, time (one entry per report, plus iteration 0);
            history['v'] holds the final equality multipliers (empty array
            when A is None), warm-starting the IP's v.
        perm_user: PARDISO permutation, reused by the warm-started IP phase:
            without equalities the ADMM matrix Q + rho*G'G shares its
            pattern with the IP's reduced system Q + G'DG; with equalities
            the ADMM saddle matrix [[Q + rho*G'G, A'], [A, 0]] shares its
            pattern with the IP's augmented system [[Q + G'DG, A'], [A, 0]].
    """
    has_eq = A is not None and A.shape[0] > 0
    # The timeline clock starts BEFORE the x-update KKT assembly and its
    # first factorization, so history['time'] accounts for the full setup
    # cost (it is part of the ADMM phase's wall-clock).
    start_time = time.time()
    # KKT matrix of the x-update (paper x-update KKT system).  Without
    # equalities: Q + rho*G'G (mtype=1).  With equalities Ax = b is
    # enforced exactly through the symmetric indefinite saddle system
    # [[Q + rho*G'G, A'], [A, 0]] (mtype=-2, upper triangle only), whose
    # sparsity pattern is identical to the IP phase's augmented system, so
    # the permutation computed here carries over.  For fixed rho the
    # matrix is factorized once; every x-update is a cheap solve.
    GtG = dot_mkl(G.T, G)
    if has_eq:
        A = csr_matrix(A)
        At = A.T.tocsr()
        b = np.asarray(b, dtype=np.float64)
        mtype_x = -2
        pt_x, iparm_x_factor = pardisoinit(mtype_x)
        iparm_x_factor[0] = 1     # user iparm
        iparm_x_factor[1] = 3     # parallel nested dissection
        iparm_x_factor[3] = num_threads_to_use
        iparm_x_factor[4] = 2     # return permutation vector
        iparm_x_factor[23] = 1    # parallel factorization
        iparm_x_solve = iparm_x_factor.copy()
        iparm_x_solve[4] = 1      # reuse the user-supplied permutation
        KKT = sp.triu(sp.bmat([[csr_matrix(Q + rho * GtG), At],
                               [A, None]], format='csr'), format='csr')
        KKT.sort_indices()
    else:
        mtype_x, pt_x = mtype, pt
        iparm_x_factor, iparm_x_solve = iparm_factor, iparm_solve
        KKT = csr_matrix(Q + rho * GtG)

    m, n = G.shape
    p = A.shape[0] if has_eq else 0

    x = np.zeros(n)
    s = np.zeros(m)
    z = np.zeros(m)
    v = np.zeros(p)

    # PARDISO analysis + numeric factorization once (phase 12); the per-
    # iteration solves below reuse it (phase 33).
    rhs_dummy = np.zeros(KKT.shape[0], dtype=np.float64)
    factor_out = pardiso(KKT, rhs_dummy, pt_x, mtype_x, iparm_x_factor,
                         phase=12, maxfct=1, mnum=1, msglvl=0)
    perm = factor_out[2]

    adaptive_mu = isinstance(mu, str) and mu.lower() == 'auto'
    if mu is None or adaptive_mu:
        mu = (eps_res)  # default / provisional value (see docstring)
    _mu_kappa = 10.0        # damping: max mu change factor per check
    _mu_first = True        # first adaptation jumps straight to the target

    history = {"objective_original": [], "primal_residual": [],
               "dual_residual": [], "time": []}
    if verbose:
        print(f"ADMM: m={m} n={n} rho={rho:g} mu={mu:g} eps={eps_res:g}")
        print(f"{'iter':>5} | {'objective':>12} | {'primal_res':>10} | "
              f"{'dual_res':>10} | {'rho':>8} | {'time(s)':>8}")

    # Preallocated buffers for the in-place vector updates
    tmp1 = np.empty_like(h)
    tmp2 = np.empty_like(h)

    # Iteration-0 entry: residuals of the initial point x = s = z = v = 0
    # at time 0, so convergence plots start from the true initial state.
    if scaling is None:
        _p0 = float(np.linalg.norm(h, ord=np.inf))
        _d0 = float(np.linalg.norm(c, ord=np.inf))
        _b0 = float(np.linalg.norm(b, ord=np.inf)) if has_eq else 0.0
    else:   # origin residuals of the ORIGINAL problem
        _p0 = float(np.linalg.norm(scaling['Einv'] * h, ord=np.inf))
        _d0 = float(np.linalg.norm(
            scaling['cinv'] * scaling['Dinv'] * c, ord=np.inf))
        _b0 = float(np.linalg.norm(scaling['EAinv'] * b, ord=np.inf)) \
            if has_eq else 0.0
    history["time"].append(0.0)
    history["objective_original"].append(0.0)
    history["primal_residual"].append(max(_p0 / (1.0 + _p0),
                                          _b0 / (1.0 + _b0)))
    history["dual_residual"].append(_d0 / (1.0 + _d0))

    for iteration in range(max_iter):
        # --- x-update (paper eq:admm_xv_update): without equalities solve
        # (Q + rho*G'G) x = G'(rho*(h-s) - z) - c; with equalities solve
        # the saddle system for (x, v) jointly, enforcing Ax = b exactly.
        np.subtract(h, s, out=tmp1)
        np.multiply(tmp1, rho, out=tmp1)
        np.subtract(tmp1, z, out=tmp1)          # tmp1 = rho*(h - s) - z
        rhs_x = dot_mkl(G.T, tmp1)
        rhs_x -= c
        if has_eq:
            rhs_x = np.concatenate([np.asarray(rhs_x, dtype=np.float64), b])
        rhs_x = np.asarray(rhs_x, dtype=np.float64)
        solve_out = pardiso(KKT, rhs_x, pt_x, mtype_x, iparm_x_solve,
                            perm=perm, phase=33, msglvl=0)
        if has_eq:
            x = solve_out[0][:n]
            v = solve_out[0][n:]
        else:
            x = solve_out[0]

        # --- s-update, closed form: s = (r1 + sqrt(r1^2 + 2*r2)) / 2 ------
        # with r1 = h - z/rho - Gx and r2 = 2*mu/rho.  The positive root of
        # the componentwise optimality condition; s > 0 automatically
        # (slack-positivity lemma), so no clamping is needed.
        Gx = dot_mkl(G, x)
        np.divide(z, rho, out=tmp1)
        np.subtract(h, tmp1, out=tmp1)
        np.subtract(tmp1, Gx, out=tmp1)         # tmp1 = r1
        r2 = 2 * (mu / rho)
        np.square(tmp1, out=tmp2)
        np.add(tmp2, 2 * r2, out=tmp2)
        np.sqrt(tmp2, out=tmp2)                 # tmp2 = sqrt(r1^2 + 2*r2)
        np.add(tmp1, tmp2, out=tmp1)
        np.divide(tmp1, 2, out=s)

        # --- z-update: z += rho * (Gx + s - h) ----------------------------
        # The order x -> s -> z makes the new iterate satisfy S Z e = mu e
        # exactly (paper theorem on ADMM complementarity).
        np.add(Gx, s, out=tmp1)
        np.subtract(tmp1, h, out=tmp1)
        np.multiply(tmp1, rho, out=tmp1)
        np.add(z, tmp1, out=z)

        # (the equality multiplier v comes exactly from the x-update saddle
        # solve above; Ax = b holds to factorization accuracy, and Ax is
        # only needed for the residual report)
        if has_eq:
            Ax = dot_mkl(A, x)

        if iteration % num_itr_check_convergence == 0:
            # Normalized residuals (shared helpers; same convention as IP)
            Qx = dot_mkl(Q, x)
            Gtz = dot_mkl(G.T, z)
            if has_eq:
                res_prim = _primal_res_norm(Gx, s, h, scaling, Ax=Ax, b=b)
                res_dual = _dual_res_norm(Qx, Gtz, c, scaling,
                                          Atv=dot_mkl(At, v))
            else:
                res_prim = _primal_res_norm(Gx, s, h, scaling)
                res_dual = _dual_res_norm(Qx, Gtz, c, scaling)

            current_obj = objective_original(x, Q, c)
            if scaling is not None:
                current_obj *= scaling['cinv']   # report in original units
            total_time = time.time() - start_time
            history["time"].append(total_time)
            history["objective_original"].append(current_obj)
            history["primal_residual"].append(res_prim)
            history["dual_residual"].append(res_dual)
            if verbose:
                print(f"{iteration + 1:>5} | {current_obj:>12.6f} | "
                      f"{res_prim:>10.4e} | {res_dual:>10.4e} | {rho:>8.3g} "
                      f"| {total_time:>8.3f}")

            if res_prim < eps_res and res_dual < eps_res:
                if verbose:
                    print(f"ADMM converged in {iteration + 1} iterations "
                          f"({total_time:.3f}s): primal/dual residuals < "
                          f"{eps_res:g}")
                break

            # Adaptive barrier weight (mu='auto'): steer the relative
            # duality gap at ADMM exit to the same order as the residual
            # tolerance.  Since S Z e = mu e and p - d ~ s'z = m*mu, the
            # target is mu = eps_res * (1 + max(|p|,|d|)) / m in ORIGINAL
            # units; the working (scaled-space) mu carries the Ruiz cost
            # factor, exactly like the fixed-mu path.  mu enters only the
            # s-update, so no refactorization is needed; updates after the
            # first are damped so the rho balancing is not perturbed.

            if adaptive_mu:
                xQx = float(np.dot(x, Qx))
                d_obj = -0.5 * xQx - float(np.dot(h, z))
                if has_eq:
                    d_obj -= float(np.dot(b, v))
                if scaling is not None:
                    d_obj *= scaling['cinv']
                p_obj = current_obj          # already in original units
                mu_target = eps_res * (1.0 + max(
                    abs(p_obj), abs(d_obj))) / m
                if scaling is not None:
                    mu_target *= scaling['cost']
                if _mu_first:
                    new_mu = mu_target       # jump to the right scale once
                    _mu_first = False
                else:                        # damped tracking
                    new_mu = min(max(mu_target, mu / _mu_kappa),
                                 mu * _mu_kappa)
                new_mu = float(np.clip(new_mu, 1e-14, 1e10))
                if new_mu != mu:
                    if verbose:
                        print(f"      mu {mu:.3g} -> {new_mu:.3g} "
                              f"(target {mu_target:.3g})")
                    mu = new_mu

            # Adaptive rho: residual balancing, applied only when the
            # proposed scale leaves [1/tau_rho, tau_rho].  (Exponent 1/2,
            # matching the paper and OSQP.)  The [1e-3, 1e3] clamp is
            # an implementation safeguard, not part of the paper.  The
            # refactorization repeats PARDISO phase 12 (analysis + numeric):
            # with mtype=1 the analysis embeds value-dependent scaling, so a
            # numeric-only phase-22 refactorization is not safe here.
            delta = 1e-8
            rho_scale = np.sqrt(res_prim / (res_dual + delta))
            if rho_scale > rho_update_threshold or \
                    rho_scale < 1 / rho_update_threshold:
                new_rho = float(np.maximum(1e-3,
                                           np.minimum(1e3, rho * rho_scale)))
                if np.abs(rho - new_rho) / rho > 0.01 :
                    if verbose:
                        print(f"      rho {rho:.3g} -> {new_rho:.3g} "
                              f"(scale {rho_scale:.2f}); refactorizing")
                    rho = new_rho
                    if has_eq:
                        KKT = sp.triu(sp.bmat(
                            [[csr_matrix(Q + rho * GtG), At],
                             [A, None]], format='csr'), format='csr')
                        KKT.sort_indices()
                    else:
                        KKT = csr_matrix(Q + rho * GtG)
                    factor_out = pardiso(KKT, rhs_dummy, pt_x, mtype_x,
                                         iparm_x_solve, phase=12, maxfct=1,
                                         mnum=1, perm=perm, msglvl=0)
    if has_eq:
        # Release the saddle factorization (the IP phase builds its own
        # augmented factorization, reusing this permutation).
        pardiso(KKT, rhs_dummy, pt_x, mtype_x, iparm_x_solve, phase=-1,
                msglvl=0)
    history["v"] = v   # final equality multipliers (empty when A is None)
    return x, s, z, history, perm


def interior_point(Q, c, G, h , A, b, max_iter, tol, warm_start, admm_config, scaling=None):
    """Mehrotra predictor-corrector primal-dual interior-point method for

        min  0.5 x'Qx + c'x   s.t.  Gx <= h,  Ax = b

    (paper, PDIP algorithm; A may be None).  Without equalities each
    iteration forms the reduced KKT system K dx = rhs with K = Q + G'DG,
    D = diag(z/s) (paper, reduced-KKT remark, solved with PARDISO mtype=1).
    With equalities the same block elimination leaves the augmented
    symmetric indefinite system

        [K   A'] [dx]   [rhs]
        [A   0 ] [dv] = [r_eq],

    solved with PARDISO mtype=-2 (Bunch-Kaufman; upper triangle only).
    Either way ds and dz are recovered by back-substitution and the
    fraction-to-the-boundary rule uses tau = 0.995 (dv is free: the
    equality multiplier v is unrestricted in sign and does not limit the
    step length).

    Args:
        Q, c, G, h: QP data (Q n x n CSR, c (n,), G m x n CSR, h (m,)).
        A, b: optional equality constraints Ax = b (A p x n CSR, b (p,)).
        max_iter: maximum number of IP iterations.
        tol: stopping tolerance; the method stops when the normalized
            primal and dual residuals and the relative duality gap are all
            below tol.
        warm_start: True -> initialize from the path-following ADMM
            (path_following_admm, configured by admm_config); False -> heuristic
            cold-start initializer.
        admm_config: dict of path_following_admm keyword arguments (used only
            when warm_start is True).
        scaling: Ruiz scaling dict when (Q, c, G, h) are equilibrated data;
            residuals, objective and duality gap are reported in the
            original scale, and the returned x/z stay in the scaled space
            (solve_qp unscales them).

    Returns:
        x: primal solution.
        history: per-iteration objective_original, primal_residual,
            dual_residual, compl_res (RMS complementarity) and time;
            history["z"] holds the final multipliers of Gx <= h and
            history["v"] the final multipliers of Ax = b (empty when A is
            None) for external KKT checks.
        admm_time: time spent in the ADMM phase (warm start) or in the
            cold-start initializer.
        info_admm: ADMM history dict (warm start) or [] (cold start).
    """
    if G is None or G.shape[0] == 0:
        # Equality-only (or unconstrained) problem: no interior exists, so
        # the interior-point iteration does not apply -- see
        # solve_equality_constrained_qp.
        return _equality_only_result(Q, c, A, b)

    has_eq = A is not None and A.shape[0] > 0
    if has_eq:
        A = csr_matrix(A)
        At = A.T.tocsr()
        b = np.asarray(b, dtype=np.float64)

    if warm_start:
        # Path-following ADMM warm start: strictly interior, well-centered
        # (x, s, z) with S Z e = mu e (and the equality multiplier v), plus
        # the PARDISO permutation: the ADMM and IP matrices share their
        # sparsity pattern in both cases (Q + rho*G'G vs Q + G'DG without
        # equalities; the identically-bordered saddle systems with them).
        _t0 = time.time()
        x, s, z, info_admm, perm = path_following_admm(Q, c, G, h , A, b,
                                                scaling=scaling,
                                                **admm_config)
        v = info_admm["v"] if has_eq else np.zeros(0)
        admm_time = time.time() - _t0
        print("Total time of ADMM is:", admm_time, "seconds.")
    else:
        # Heuristic cold-start initialization (paper appendix)
        _t0 = time.time()
        x, v, s, z, perm = initialize_primal_dual(
            Q=Q, c=c, A=A if has_eq else None, b=b if has_eq else None,
            G=G, h=h)
        if v is None:
            v = np.zeros(0)
        admm_time = time.time() - _t0  # cold-start initializer time
        info_admm = []

    # Everything from here to the first Newton iteration (transposes,
    # PARDISO handles, bookkeeping) is charged to the first iteration's
    # duration so the reported timeline has no untimed gap after admm_time.
    _setup_t0 = time.time()
    m, n = G.shape
    p = A.shape[0] if has_eq else 0
    history = {"objective_original": [], "primal_residual": [],
               "dual_residual": [], "compl_res": [], "time": []}
    print(f"{'iter':>5} | {'objective':>13} | {'primal_res':>10} | {'dual_res':>10} | {'mu':>9} | {'dual_gap':>9} | {'sigma':>6} | {'time(s)':>8}")

    time_kkt_form = 0
    time_pardiso = 0
    time_pardiso_c = 0
    elapsed_time = 0

    Gt = G.T.tocsr()
    mtype = 1  # PARDISO: real, structurally symmetric (reduced SPD system)
    sigma = 1  # printed before it is first computed

    # Safeguard state: best-iterate snapshot + stall/divergence bookkeeping
    # (see IP_SAFEGUARDS at the top of this file)
    best_kkt = np.inf
    best_iter = 0
    best_snap = None
    best_hist = []          # best_kkt after each iteration (stall window)
    stop_reason = 'max_iter'
    if has_eq:
        # Augmented indefinite system: separate PARDISO handle, mtype=-2
        # (real symmetric indefinite), upper triangle only.  The warm-start
        # ADMM x-update solves the saddle system [[Q + rho*G'G, A'],[A, 0]],
        # whose sparsity pattern is IDENTICAL to this augmented system, so
        # its permutation is reused directly.  On a cold start the first
        # iteration lets PARDISO reorder and returns the permutation of the
        # augmented pattern for reuse (the pattern is constant across
        # iterations: only D = diag(z/s) changes values).
        mtype_aug = -2
        pt_aug, iparm_aug_factor = pardisoinit(mtype_aug)
        iparm_aug_factor[0] = 1    # user iparm
        iparm_aug_factor[1] = 3    # parallel nested dissection
        iparm_aug_factor[3] = num_threads_to_use
        iparm_aug_factor[4] = 2    # return permutation vector
        iparm_aug_factor[23] = 1   # parallel factorization
        iparm_aug_solve = iparm_aug_factor.copy()
        iparm_aug_solve[4] = 1     # use user-supplied permutation
        perm_aug = perm if (warm_start and perm is not None) else None

    for iteration in range(max_iter):

        start_time = _setup_t0 if iteration == 0 else time.time()
        Qx = dot_mkl(Q, x)
        Gx = dot_mkl(G, x)
        Gtz = dot_mkl(Gt, z)
        if has_eq:
            Ax = dot_mkl(A, x)
            Atv = dot_mkl(At, v)
        else:
            Atv = None

        # Negated perturbed KKT residuals (predictor step: mu = 0)
        r_dual = -(Qx + Gtz + c) if not has_eq else -(Qx + Gtz + Atv + c)
        r_primal = -(Gx + s - h)
        r_cs_z = -(s * z)
        if has_eq:
            r_eq = -(Ax - b)

        # Convergence measures: normalized residuals (shared helpers),
        # RMS complementarity, and the relative duality gap
        if has_eq:
            primal_residual = _primal_res_norm(Gx, s, h, scaling, Ax=Ax, b=b)
            dual_residual = _dual_res_norm(Qx, Gtz, c, scaling, Atv=Atv)
        else:
            primal_residual = _primal_res_norm(Gx, s, h, scaling)
            dual_residual = _dual_res_norm(Qx, Gtz, c, scaling)
        complementarity_gap = np.sqrt((np.linalg.norm(r_cs_z) ** 2) / m)

        xQx = dot_mkl(x.T, Qx)
        primal_objective = 0.5 * xQx + dot_mkl(c.T, x)
        dual_objective = -0.5 * xQx - dot_mkl(h.T, z)
        if has_eq:
            dual_objective -= dot_mkl(b.T, v)
        if scaling is not None:
            # s~z~ = cost*(s z) and both objectives carry a factor cost
            cinv = scaling['cinv']
            complementarity_gap *= cinv
            primal_objective *= cinv
            dual_objective *= cinv
        dual_gap = abs(primal_objective - dual_objective) / (
            1 + max(abs(primal_objective), abs(dual_objective)))

        print(f"{iteration + 1:>5} | {primal_objective:>13.6f} | "
              f"{primal_residual:>10.4e} | {dual_residual:>10.4e} | "
              f"{complementarity_gap:>9.3e} | {dual_gap:>9.3e} | "
              f"{sigma:>6.4f} | {elapsed_time:>8.3f}")

        history["objective_original"].append(primal_objective)
        history["primal_residual"].append(primal_residual)
        history["dual_residual"].append(dual_residual)
        history["compl_res"].append(complementarity_gap)

        # --- Safeguards: best-iterate tracking (before the convergence
        # check, so the converged iterate is itself eligible as best).
        # Judged by the VIOLATION-based primal residual (_primal_viol_norm,
        # no slack in the formula), so best/stall/divergence follow the
        # same measure as the external kkt_metrics check: the slack
        # residual can look small while s is huge (large-mu warm start)
        # although the true violation of x is poor. -------------------------
        if has_eq:
            viol_residual = _primal_viol_norm(Gx, h, scaling, Ax=Ax, b=b)
        else:
            viol_residual = _primal_viol_norm(Gx, h, scaling)
        kkt_max_it = max(viol_residual, dual_residual, dual_gap)
        if IP_SAFEGUARDS['enabled'] and np.isfinite(kkt_max_it) \
                and kkt_max_it < best_kkt:
            best_kkt = kkt_max_it
            best_iter = iteration + 1
            best_snap = (x.copy(), s.copy(), z.copy(),
                         v.copy() if has_eq else None)

        if primal_residual < tol and dual_residual < tol and dual_gap < tol:
            print(f"Converged at iteration {iteration + 1}")
            stop_reason = 'converged'
            # keep history["time"] aligned with the residual entries added
            # above (plots truncate to the shortest array otherwise)
            history["time"].append(time.time() - start_time)
            break

        # --- Safeguards: divergence & stall stops -------------------------
        if IP_SAFEGUARDS['enabled']:
            if not np.isfinite(kkt_max_it):
                print(f"[warm_ip] stopping at iteration {iteration + 1}: "
                      f"non-finite residuals (diverged); returning best "
                      f"iterate (iter {best_iter}, kkt_max {best_kkt:.3e})")
                stop_reason = 'diverged'
                history["time"].append(time.time() - start_time)
                break
            best_hist.append(best_kkt)
            if kkt_max_it > IP_SAFEGUARDS['divergence_factor'] * best_kkt:
                print(f"[warm_ip] stopping at iteration {iteration + 1}: "
                      f"kkt_max {kkt_max_it:.3e} > "
                      f"{IP_SAFEGUARDS['divergence_factor']:g} x best "
                      f"{best_kkt:.3e} (diverged); returning best iterate "
                      f"(iter {best_iter})")
                stop_reason = 'diverged'
                history["time"].append(time.time() - start_time)
                break
            _W = IP_SAFEGUARDS['stall_window']
            if iteration + 1 >= IP_SAFEGUARDS['min_iterations'] and \
                    len(best_hist) > _W and \
                    best_hist[-1] > IP_SAFEGUARDS['stall_ratio'] * \
                    best_hist[-1 - _W]:
                print(f"[warm_ip] stopping at iteration {iteration + 1}: "
                      f"best kkt_max improved less than "
                      f"{100 * (1 - IP_SAFEGUARDS['stall_ratio']):.0f}% over "
                      f"the last {_W} iterations (stalled); returning best "
                      f"iterate (iter {best_iter}, kkt_max {best_kkt:.3e})")
                stop_reason = 'stalled'
                history["time"].append(time.time() - start_time)
                break

        # --- Reduced KKT system: K = Q + G'DG, D = diag(z/s) -------------
        # (paper, reduced-KKT remark; rhs by block elimination).  With
        # equality constraints K is bordered by A into the augmented
        # indefinite system [[K, A'], [A, 0]].
        t0 = time.time()
        rhs_reduced = - (-r_dual + dot_mkl(G.T, (r_cs_z / s - z * r_primal / s)))
        D = diags(z / s, format='csr')
        GDG = dot_mkl(dot_mkl(Gt, D), G)
        kkt_matrix = csr_matrix(Q + GDG)
        if has_eq:
            kkt_matrix = sp.bmat([[kkt_matrix, At], [A, None]], format='csr')
            kkt_matrix = sp.triu(kkt_matrix, format='csr')  # mtype=-2
            kkt_matrix.sort_indices()
            rhs_reduced = np.concatenate([rhs_reduced, r_eq])
        time_kkt_form += time.time() - t0

        # --- Predictor (affine-scaling) direction ------------------------
        # PARDISO phase 13 = analysis + factorization + solve each
        # iteration.  With mtype=1 the analysis embeds value-dependent
        # scaling, so reusing the symbolic factorization across iterations
        # (paper remark on factorization reuse) is not safe for this matrix
        # type.
        t0 = time.time()
        if has_eq:
            if perm_aug is None:   # first iteration: let PARDISO reorder
                d_aff = pardiso(kkt_matrix, rhs_reduced, pt_aug, mtype_aug,
                                iparm_aug_factor, phase=13, maxfct=1, mnum=1,
                                msglvl=0)
                perm_aug = d_aff[2]
            else:                  # reuse the augmented-pattern permutation
                d_aff = pardiso(kkt_matrix, rhs_reduced, pt_aug, mtype_aug,
                                iparm_aug_solve, phase=13, maxfct=1, mnum=1,
                                perm=perm_aug, msglvl=0)
        else:
            d_aff = pardiso(kkt_matrix, rhs_reduced, pt, mtype, iparm_solve,
                            phase=13, maxfct=1, mnum=1, perm=perm, msglvl=0)
        d_aff = d_aff[0]
        time_pardiso += time.time() - t0

        dx_aff = d_aff[:x.size]
        ds_aff = r_primal - dot_mkl(G, dx_aff)      # back-substitution
        dz_aff = (r_cs_z - z * ds_aff) / s

        # Fraction-to-the-boundary steps, tau = 0.995
        alpha_s_aff = max_step(s, ds_aff, 0.995)
        alpha_z_aff = max_step(z, dz_aff, 0.995)

        # Mehrotra centering parameter: sigma = clip(mu_aff / mu)^3
        mu_aff = dot_mkl((s + alpha_s_aff * ds_aff).T,
                         (z + alpha_z_aff * dz_aff)) / m
        mu = dot_mkl(s.T, z) / m
        eta = mu_aff / mu
        sigma = (max(0, min(1, eta))) ** 3

        # --- Corrector step ----------------------------------------------
        # Complementarity residual with second-order and centering terms:
        # r_c = SZe + dS_aff dZ_aff e - sigma*mu*e (negated here)
        r_cs_z -= ds_aff * dz_aff - sigma * mu
        rhs_corr = - (-r_dual + dot_mkl(G.T, (r_cs_z / s - z * r_primal / s)))
        if has_eq:
            # equality residual is linear: unchanged in the corrector
            rhs_corr = np.concatenate([rhs_corr, r_eq])

        # Solve with the same matrix: phase 33 reuses the predictor's
        # factorization; phase -1 then releases PARDISO internal memory.
        t0 = time.time()
        if has_eq:
            d_c = pardiso(kkt_matrix, rhs_corr, pt_aug, mtype_aug,
                          iparm_aug_solve, perm=perm_aug, phase=33, msglvl=0)
            d_c = d_c[0]
            pardiso(kkt_matrix, rhs_corr, pt_aug, mtype_aug, iparm_aug_solve,
                    phase=-1)
        else:
            d_c = pardiso(kkt_matrix, rhs_corr, pt, mtype, iparm_solve,
                          perm=perm, phase=33, msglvl=0)
            d_c = d_c[0]
            pardiso(kkt_matrix, rhs_corr, pt, mtype, iparm_solve, phase=-1)
        time_pardiso_c += time.time() - t0

        dx_c = d_c[:x.size]
        ds_c = r_primal - dot_mkl(G, dx_c)          # back-substitution
        dz_c = (r_cs_z - z * ds_c) / s

        alpha_s = max_step(s, ds_c, 0.995)
        alpha_z = max_step(z, dz_c, 0.995)
        alpha = min(alpha_s, alpha_z)

        x += alpha * dx_c
        s += alpha * ds_c
        z += alpha * dz_c
        if has_eq:
            # v is free (no sign constraint); same step length keeps the
            # search direction consistent
            v += alpha * d_c[x.size:]

        end_time = time.time()
        elapsed_time += end_time - start_time
        history["time"].append(end_time - start_time)

    print(f"Time spent on KKT form: {time_kkt_form:.4f} seconds")
    print(f"Time spent on Pardiso: {time_pardiso:.4f} seconds")
    print(f"Time spent on Pardiso for corrector step: {time_pardiso_c:.4f} seconds")

    # Return the BEST iterate seen (smallest kkt_max), not necessarily the
    # last one; the per-iteration history keeps the true trajectory.
    if best_snap is not None:
        x, s, z = best_snap[0], best_snap[1], best_snap[2]
        if has_eq:
            v = best_snap[3]
        if best_iter != len(history["objective_original"]):
            print(f"[warm_ip] returning best iterate: iteration {best_iter} "
                  f"(kkt_max {best_kkt:.3e})")
    history["stop_reason"] = stop_reason
    history["best_iteration"] = best_iter
    history["best_kkt_max"] = best_kkt
    history["z"] = z  # dual multipliers of Gx <= h (for external KKT checks)
    history["v"] = v if has_eq else np.zeros(0)  # multipliers of Ax = b
    return x, history, admm_time, info_admm

def solve_qp(Q, c, G, h, A, b, max_iter, tol, warm_start,
                               admm_config, ruiz=False, eq_mode=None):
    """Warm-IP / IP entry point (MKL PARDISO backend).

    ruiz: run Ruiz equilibration on the problem data first (OSQP's
        modified Ruiz algorithm) and map the solution back afterwards, so
        callers always see original-scale results.
        Residual-based termination uses unscaled
        residuals (OSQP's default), so tol keeps its meaning. Equality-only
        problems take the closed-form path and are never scaled.
    eq_mode: how equality constraints Ax = b are handled when the problem
        also has inequalities: 'direct' (native, augmented KKT) or
        'absorb' (rewritten as the pair Ax <= b, -Ax <= -b and stacked into
        G). Defaults to EQ_MODE (top of this file). Either way the returned
        history carries history['z'] (multipliers of the G rows) and
        history['v'] (multipliers of Ax = b, empty when A is None).
    """
    mode = (eq_mode or EQ_MODE).lower()
    if mode not in ('absorb', 'direct'):
        raise ValueError(f"eq_mode must be 'absorb' or 'direct', got {mode!r}")
    has_ineq = G is not None and G.shape[0] > 0
    has_eq = A is not None and A.shape[0] > 0
    mixed = has_ineq and has_eq
    m_ineq = G.shape[0] if has_ineq else 0
    p_eq = A.shape[0] if has_eq else 0
    absorbed = False
    if mixed:
        if mode == 'absorb':
            # Ax = b  ->  the inequality pair  Ax <= b, -Ax <= -b, stacked
            # under the genuine inequalities; the multiplier of Ax = b is
            # recovered afterwards as v = z_plus - z_minus.
            A_ = csr_matrix(A)
            G = sp.csr_matrix(sp.vstack([csr_matrix(G), A_, -A_],
                                        format='csr'))
            b_ = np.asarray(b, dtype=np.float64)
            h = np.concatenate([np.asarray(h, dtype=np.float64), b_, -b_])
            A, b = None, None
            has_eq = False
            absorbed = True
            print(f"[warm_ip] eq_mode=absorb: {p_eq} equality rows stacked "
                  f"as ± inequality pairs", flush=True)
        else:
            print(f"[warm_ip] eq_mode=direct: {p_eq} equality rows handled "
                  f"natively", flush=True)

    scaling = None
    if ruiz and has_ineq:
        if has_eq:   # direct mode with equalities: equilibrate [G; A]
            Q, c, G, h, A, b, scaling = ruiz_equilibration(Q, c, G, h,
                                                           A=A, b=b)
        else:
            Q, c, G, h, scaling = ruiz_equilibration(Q, c, G, h)
        # The ADMM barrier weight lives in the scaled space: complementarity
        # satisfies s~z~ = cost*(s z), so mu must carry the same factor for
        # the warm start to land where the caller asked in original units.
        if scaling['cost'] != 1.0 and warm_start:
            _mu = admm_config.get('mu')
            if isinstance(_mu, str):
                # mu='auto': the adaptive rule inside ADMM applies the cost
                # factor itself when computing the working mu.
                pass
            else:
                admm_config = dict(admm_config)
                if _mu is None:
                    _mu = 0.01 * admm_config['eps_res']
                admm_config['mu'] = _mu * scaling['cost']
                print(f"[ruiz] barrier weight mu scaled by cost: {_mu:.3g} "
                      f"-> {admm_config['mu']:.3g}", flush=True)
    out = interior_point(Q, c, G, h, A, b, max_iter=max_iter, tol=tol,
                     warm_start=warm_start, admm_config=admm_config,
                     scaling=scaling)
    if scaling is None:
        x, history, admm_time, info_admm = out
        if absorbed and isinstance(history, dict) and \
                history.get('z') is not None and np.size(history['z']):
            z_ext = np.asarray(history['z'])
            history['z'] = z_ext[:m_ineq]
            history['v'] = (z_ext[m_ineq:m_ineq + p_eq]
                            - z_ext[m_ineq + p_eq:])
        if isinstance(history, dict):
            history.setdefault('v', np.zeros(0))
        return x, history, admm_time, info_admm
    # map the solution back to the original scale: x = D x~, z = cost^-1 E z~
    # (and v = cost^-1 EA v~ in direct mode)
    x, history, admm_time, info_admm = out
    x = scaling['D'] * x
    if isinstance(history, dict):
        if history.get('z') is not None and np.size(history['z']):
            history['z'] = scaling['cinv'] * scaling['E'] * history['z']
            if absorbed:
                z_ext = np.asarray(history['z'])
                history['z'] = z_ext[:m_ineq]
                history['v'] = (z_ext[m_ineq:m_ineq + p_eq]
                                - z_ext[m_ineq + p_eq:])
        if has_eq and history.get('v') is not None and np.size(history['v']):
            history['v'] = scaling['cinv'] * scaling['EA'] * history['v']
        history.setdefault('v', np.zeros(0))
    return x, history, admm_time, info_admm
