"""
Loader for the public QP benchmark instances shipped with Warm-IP.

Three families of stored instances are provided, in two storage forms
(the synthetic family is generated on the fly by its benchmark script and
has no data files):

Radiotherapy IMRT instances (IMRT_lung_data/, canonical one-sided form)

    minimize    0.5 * x' Q x + q' x + c
    subject to  G x <= h

Maros-Meszaros instances (MM_data/) and MPC instances (MPC_data/), both in
the two-sided form

    minimize    0.5 * x' Q x + q' x + c
    subject to  l_h <= G x <= u_h
                l_x <=  x  <= u_x

load_data returns whichever form the file stores, with ``form`` set to
'canonical' or 'two_sided'. to_canonical() converts either form into the
one-sided (Q, q, c, G, h) form (equality rows doubled as ± pairs);
to_split() instead keeps the equality rows separate and returns
(Q, q, c, G, h, A, b) for solvers that support Ax = b natively.

Q and G are stored as CSR (data/indices/indptr/shape); Q is the full
symmetric matrix, not a triangle.
MM and MPC files may carry a reference solution (x_opt, dual, obj_val,
solver); the IMRT files do not.

Instances live under the repository's data/ folder (data/IMRT_lung_data,
data/MM_data, data/MPC_data); radiotherapy families are named
<modality>_<site> (IMRT_lung, later e.g. IMRT_prostate, VMAT_lung) with
instance names <family>_<index> -- patient, protocol, and provenance are
recorded in data/instances_metadata.csv, not in file or folder names.
A requested instance that is not present
locally is downloaded automatically from the Hugging Face dataset
https://huggingface.co/datasets/Radiotherapy-Optimization/QP-Benchmark.

Dependencies: numpy, scipy, h5py; huggingface_hub for auto-download.

Example
-------
>>> from data_loader import load_data, to_canonical
>>> prob = load_data("MM_001_AUG2D")          # two-sided form
>>> Q, q, c, G, h = to_canonical(prob)        # one-sided form
>>> prob = load_data("IMRT_lung_001")         # already canonical
"""

import csv
import os
import re

import h5py
import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(os.path.dirname(_HERE), "data")
_DEFAULT_DATA_DIR = os.path.join(_DATA_ROOT, "IMRT_lung_data")

# Hugging Face dataset holding every benchmark family; instances that are
# not present locally are downloaded from here on first use.
HF_DATASET = "Radiotherapy-Optimization/QP-Benchmark"

# Instance-name prefix -> folder holding that family. Radiotherapy
# families are one folder per (modality, site) pair; future families
# (IMRT_prostate, VMAT_lung, ...) are added here.
_FAMILY_DIRS = {"IMRT_lung_": "IMRT_lung_data", "MM_": "MM_data",
                "MPC_": "MPC_data"}


def _family_dir(name):
    """Data folder implied by an instance name prefix (None if unknown)."""
    for prefix, folder in _FAMILY_DIRS.items():
        if name.startswith(prefix):
            return os.path.join(_DATA_ROOT, folder)
    return None


# Family-plus-index shorthand ("MM_2", "IMRT_lung_1", zero-padded or
# not, with or without .h5); the family alternatives are derived from
# _FAMILY_DIRS, longest prefix first.
_SHORT_RE = re.compile(
    r"^(%s)(\d+)(?:\.h5)?$"
    % "|".join(re.escape(p) for p in
               sorted(_FAMILY_DIRS, key=len, reverse=True)))
_MANIFEST_CACHE = None


def _manifest_names():
    """family -> sorted instance names from data/instances_metadata.csv
    (cached; empty when the manifest is missing)."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        out = {}
        try:
            with open(os.path.join(_DATA_ROOT, "instances_metadata.csv"),
                      newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    out.setdefault(row["family"], []).append(row["name"])
        except OSError:
            pass
        _MANIFEST_CACHE = {k: sorted(v) for k, v in out.items()}
    return _MANIFEST_CACHE


def _expand_short_name(name, data_dir=None):
    """Expand a family-plus-index shorthand ("MM_002", "IMRT_lung_1",
    with or without .h5) to the full instance name, using
    data/instances_metadata.csv (or, when the manifest is missing, the
    local family folder). Names not of the short form are returned
    unchanged.

    A full instance name may equal its shorthand (radiotherapy names are
    exactly <family>_<index>, e.g. IMRT_lung_001); MM/MPC names carry a
    suffix after the index (MM_001_AUG2D), so a match accepts both
    "<prefix><idx>" itself and "<prefix><idx>_...".
    """
    m = _SHORT_RE.match(name)
    if not m:
        return name
    prefix, idx = m.group(1), int(m.group(2))     # prefix ends with "_"
    fam = prefix.rstrip("_")                      # manifest family key
    base = f"{prefix}{idx:03d}"
    hits = [nm for nm in _manifest_names().get(fam, ())
            if nm == base or nm.startswith(base + "_")]
    if not hits:                       # no manifest: scan local folders
        for d in filter(None, (data_dir, _family_dir(prefix))):
            if os.path.isdir(d):
                hits += [os.path.splitext(fn)[0] for fn in os.listdir(d)
                         if fn.endswith(".h5")
                         and (os.path.splitext(fn)[0] == base
                              or fn.startswith(base + "_"))]
    if not hits:
        known = _manifest_names().get(fam)
        rng = (f"valid indices: 1..{len(known)}" if known
               else "see data/instances_metadata.csv")
        raise FileNotFoundError(
            f"no {fam} instance with index {idx} ({name!r}); {rng}")
    return sorted(set(hits))[0]


def _fetch_from_hf(name, fname):
    """Download one instance from the Hugging Face dataset into data/;
    returns the local path, or None when the family is unknown."""
    folder = None
    for prefix, d in _FAMILY_DIRS.items():
        if name.startswith(prefix):
            folder = d
            break
    if folder is None:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            f"{fname} is not available locally and the huggingface_hub "
            "package is not installed. Install it (pip install "
            "huggingface_hub) to download instances automatically from "
            f"https://huggingface.co/datasets/{HF_DATASET}, or place the "
            f"file under data/{folder}/ manually.")
    print(f"[data] {fname} not found locally -- downloading from "
          f"https://huggingface.co/datasets/{HF_DATASET}", flush=True)
    return hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                           filename=f"{folder}/{fname}",
                           local_dir=_DATA_ROOT)


def _resolve_path(name, data_dir=None):
    """Locate an instance file by name, prefix, or explicit directory;
    falls back to downloading it from the Hugging Face dataset."""
    if os.path.isfile(name):
        return name
    name = _expand_short_name(name, data_dir)
    fname = name if name.endswith(".h5") else name + ".h5"
    candidates = []
    if data_dir is not None:
        candidates.append(os.path.join(data_dir, fname))
    family = _family_dir(name)
    if family is not None:
        candidates.append(os.path.join(family, fname))
    # fall back to scanning every known family folder
    candidates += [os.path.join(_DATA_ROOT, folder, fname)
                   for folder in _FAMILY_DIRS.values()]
    candidates.append(os.path.join(_DEFAULT_DATA_DIR, fname))
    for path in candidates:
        if os.path.isfile(path):
            return path
    if family is None and data_dir is None:
        prefixes = ", ".join(p.rstrip("_") for p in _FAMILY_DIRS)
        extra = (" Synthetic instances (SYN_...) have no data files; they "
                 "are generated in memory by run_warm_ip.py and "
                 "scripts/run_benchmark_Synthetic.py."
                 if name.upper().startswith("SYN") else "")
        raise FileNotFoundError(
            f"unknown benchmark instance {name!r}: names start with one of "
            f"{prefixes} (or pass a path / data_dir).{extra}")
    fetched = _fetch_from_hf(name, fname)
    return fetched if fetched else candidates[0]


def _read_str(f, name):
    if name not in f:
        return ""
    val = f[name][()]
    return val.decode() if isinstance(val, bytes) else str(val)


def _read_csr(f, name):
    return sp.csr_matrix(
        (f[f"{name}_data"][()], f[f"{name}_indices"][()], f[f"{name}_indptr"][()]),
        shape=tuple(f[f"{name}_shape"][()]))


def load_data(name, data_dir=None):
    """Load one benchmark instance.

    Parameters
    ----------
    name : str
        Instance name (e.g. ``"IMRT_lung_001"`` or ``"MM_001_AUG2D"``, with
        or without ``.h5``), a family-plus-index shorthand (``"MM_002"``,
        ``"IMRT_lung_1"``; expanded through data/instances_metadata.csv),
        or a full path
        to an ``.h5`` file. Without ``data_dir`` the folder is chosen from
        the name prefix.
    data_dir : str, optional
        Directory containing the instance files; overrides the prefix rule.

    Returns
    -------
    dict with keys:
        ``Q``   scipy.sparse.csr_matrix, (n, n)
        ``q``   ndarray, (n,)
        ``c``   float, objective constant
        ``G``   scipy.sparse.csr_matrix, (m, n)
        ``form``         'canonical' or 'two_sided'
        ``format_note``  str, description of the format
        ``data_note``    str, provenance of the data
    canonical files add:
        ``h``   ndarray, (m,)          constraints are Gx <= h
    two-sided files add:
        ``l_h``, ``u_h``   ndarray, (m,)   l_h <= Gx <= u_h (may be +/-inf)
        ``l_x``, ``u_x``   ndarray, (n,)   l_x <= x <= u_x  (may be +/-inf)
        ``source_file``    str, original file name in the source repository
    and, when the file contains a reference solution:
        ``x_opt``    ndarray, (n,)  primal solution
        ``dual``     ndarray, (m,)  multipliers of Gx <= h (nonnegative)
        ``obj_val``  float
        ``solver``   str, e.g. "MOSEK 11.0.5"
    """
    path = _resolve_path(name, data_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"benchmark instance not found: {path}")

    out = {}
    with h5py.File(path, "r") as f:
        out["Q"] = _read_csr(f, "Q")
        out["G"] = _read_csr(f, "G")
        out["q"] = np.asarray(f["q"][()], dtype=np.float64)
        out["c"] = float(f["c"][()])
        out["format_note"] = _read_str(f, "format_note")
        out["data_note"] = _read_str(f, "data_note")
        if "h" in f:
            out["form"] = "canonical"
            out["h"] = np.asarray(f["h"][()], dtype=np.float64)
        else:
            out["form"] = "two_sided"
            for key in ("l_h", "u_h", "l_x", "u_x"):
                out[key] = np.asarray(f[key][()], dtype=np.float64)
            out["source_file"] = _read_str(f, "source_file")
        if "x_opt" in f:
            out["x_opt"] = np.asarray(f["x_opt"][()], dtype=np.float64)
            out["dual"] = np.asarray(f["dual"][()], dtype=np.float64)
            out["obj_val"] = float(f["obj_val"][()])
            out["solver"] = _read_str(f, "solver")
    return out


def to_canonical(prob):
    """Convert an instance to the canonical form  min 0.5x'Qx + q'x + c
    s.t. Gx <= h.

    Canonical instances are returned unchanged. For a two-sided instance
    every finite bound becomes one row, in this order:

        G x <= u_h      (rows with finite u_h)
       -G x <= -l_h     (rows with finite l_h)
        I x <= u_x      (variables with finite u_x)
       -I x <= -l_x     (variables with finite l_x)

    so an equality row (l_h_i == u_h_i) contributes the pair
    (G_i x <= b_i, -G_i x <= -b_i), and a pure-equality problem yields
    exactly G_c = [G; -G], h_c = [b; -b].

    Returns
    -------
    (Q, q, c, G, h)
    """
    Q, q, c = prob["Q"], prob["q"], prob["c"]
    if prob.get("form", "canonical") == "canonical":
        return Q, q, c, prob["G"], prob["h"]

    G, l_h, u_h = sp.csr_matrix(prob["G"]), prob["l_h"], prob["u_h"]
    l_x, u_x = prob["l_x"], prob["u_x"]
    n = Q.shape[0]

    blocks, rhs = [], []
    up = np.isfinite(u_h)
    if up.any():
        blocks.append(G[up, :])
        rhs.append(u_h[up])
    lo = np.isfinite(l_h)
    if lo.any():
        blocks.append(-G[lo, :])
        rhs.append(-l_h[lo])
    ux = np.isfinite(u_x)
    if ux.any():
        eye = sp.identity(n, format="csr")
        blocks.append(eye[ux, :])
        rhs.append(u_x[ux])
    lx = np.isfinite(l_x)
    if lx.any():
        eye = sp.identity(n, format="csr")
        blocks.append(-eye[lx, :])
        rhs.append(-l_x[lx])

    if blocks:
        G_c = sp.csr_matrix(sp.vstack(blocks, format="csr"))
        h_c = np.concatenate(rhs)
    else:                                   # unconstrained problem
        G_c = sp.csr_matrix((0, n))
        h_c = np.zeros(0)
    G_c.sort_indices()
    return Q, q, c, G_c, h_c


def to_split(prob):
    """Convert an instance to the split form

        min 0.5 x'Qx + q'x + c   s.t.   Gx <= h,   Ax = b

    with the genuine inequalities in G/h and the equality rows (finite
    l_h == u_h) in A/b, so solvers that support equality constraints
    natively receive them as such (no ± doubling).

    Inequality rows keep the to_canonical order, restricted to non-equality
    rows: finite u_h, then finite l_h (negated), then finite u_x, then
    finite l_x (negated).

    Returns
    -------
    (Q, q, c, G, h, A, b); A = b = None when the instance has no equality
    rows (canonical instances always return A = None).
    """
    Q, q, c = prob["Q"], prob["q"], prob["c"]
    if prob.get("form", "canonical") == "canonical":
        return Q, q, c, prob["G"], prob["h"], None, None

    G, l_h, u_h = sp.csr_matrix(prob["G"]), prob["l_h"], prob["u_h"]
    l_x, u_x = prob["l_x"], prob["u_x"]
    n = Q.shape[0]
    eq = np.isfinite(u_h) & (l_h == u_h)

    blocks, rhs = [], []
    up = np.isfinite(u_h) & ~eq
    if up.any():
        blocks.append(G[up, :])
        rhs.append(u_h[up])
    lo = np.isfinite(l_h) & ~eq
    if lo.any():
        blocks.append(-G[lo, :])
        rhs.append(-l_h[lo])
    ux = np.isfinite(u_x)
    if ux.any():
        eye = sp.identity(n, format="csr")
        blocks.append(eye[ux, :])
        rhs.append(u_x[ux])
    lx = np.isfinite(l_x)
    if lx.any():
        eye = sp.identity(n, format="csr")
        blocks.append(-eye[lx, :])
        rhs.append(-l_x[lx])

    if blocks:
        G_c = sp.csr_matrix(sp.vstack(blocks, format="csr"))
        h_c = np.concatenate(rhs)
    else:
        G_c = sp.csr_matrix((0, n))
        h_c = np.zeros(0)
    G_c.sort_indices()

    if eq.any():
        A = sp.csr_matrix(G[eq, :])
        A.sort_indices()
        b = np.asarray(u_h[eq], dtype=np.float64)
    else:
        A, b = None, None
    return Q, q, c, G_c, h_c, A, b


def is_equality_only(prob):
    """True when the instance has only equality constraints and no finite
    variable bounds. Such problems have no strictly feasible interior, so
    the interior-point solvers handle them with a single closed-form solve
    (see warm_ip_solver.solve_equality_constrained_qp)."""
    if prob.get("form", "canonical") != "two_sided":
        return False
    return bool(np.all(prob["l_h"] == prob["u_h"])
                and not np.isfinite(prob["l_x"]).any()
                and not np.isfinite(prob["u_x"]).any())


def equality_blocks(prob):
    """Return (A, b) of the equality constraints Ax = b of a two-sided
    instance (rows with l_h == u_h)."""
    if prob.get("form", "canonical") != "two_sided":
        raise ValueError("equality_blocks requires a two-sided instance")
    eq = prob["l_h"] == prob["u_h"]
    A = sp.csr_matrix(sp.csr_matrix(prob["G"])[eq, :])
    A.sort_indices()
    return A, prob["u_h"][eq]


def _inf_norm(v):
    """Infinity norm that returns 0.0 for empty vectors (np.linalg.norm
    raises on them)."""
    v = np.asarray(v)
    return float(np.linalg.norm(v, np.inf)) if v.size else 0.0


def kkt_metrics(prob, x, dual, A=None, b=None, v=None):
    """Normalized KKT quality of a solution to min 0.5 x'Qx + q'x + c
    s.t. Gx <= h (and optionally Ax = b), using the normalizations common
    in the first-order/IPM solver literature (OSQP/Clarabel style; the
    formulas of the paper's stopping-criteria remark).

    Parameters
    ----------
    prob : dict with keys Q, q, c, G, h (as returned by load_data for a
        canonical instance, or built from to_canonical / to_split output --
        pass it as {'Q': Q, 'q': q, 'c': c, 'G': G, 'h': h}; G may have
        zero rows for a pure-equality problem)
    x : ndarray (n,), primal solution
    dual : ndarray (m,), multipliers of Gx <= h (nonnegative)
    A, b : optional equality constraints Ax = b
    v : ndarray (p,), multipliers of Ax = b (sign convention:
        Qx + q + G'z + A'v = 0); required when A is given

    Returns
    -------
    (primal_res, dual_res, dual_gap) : floats; primal_res is the max of the
    inequality and equality residuals, the dual residual includes A'v, and
    the dual objective the -b'v term.
    """
    Q, q, c, G, h = prob['Q'], prob['q'], prob['c'], prob['G'], prob['h']
    has_eq = A is not None and A.shape[0] > 0
    if has_eq and v is None:
        raise ValueError("kkt_metrics: equality constraints given but no "
                         "equality multipliers v")
    Qx = Q @ x
    m = G.shape[0] if G is not None else 0
    if m > 0:
        Gx = G @ x
        Gtz = G.T @ dual
        primal_ineq = _inf_norm(np.maximum(Gx - h, 0.0)) / (
            1.0 + max(_inf_norm(Gx), _inf_norm(h)))
        hz = h @ dual
    else:
        Gtz = np.zeros_like(x)
        primal_ineq = 0.0
        hz = 0.0
    if has_eq:
        Ax = A @ x
        Atv = A.T @ v
        primal_eq = _inf_norm(Ax - b) / (
            1.0 + max(_inf_norm(Ax), _inf_norm(b)))
        bv = b @ v
    else:
        Atv = np.zeros_like(x)
        primal_eq = 0.0
        bv = 0.0
    primal_res = max(primal_ineq, primal_eq)
    dual_res = _inf_norm(Qx + q + Gtz + Atv) / (
        1.0 + max(_inf_norm(Qx), _inf_norm(q), _inf_norm(Gtz),
                  _inf_norm(Atv)))
    pobj = 0.5 * x @ Qx + q @ x + c
    dobj = -0.5 * x @ Qx - hz - bv + c
    dual_gap = abs(pobj - dobj) / (1.0 + max(abs(pobj), abs(dobj)))
    return float(primal_res), float(dual_res), float(dual_gap)


def ensure_local(names, data_dir=None):
    """Make sure every named instance is present locally, downloading the
    missing ones from the Hugging Face dataset up front (rather than one
    by one in the middle of a benchmark run). Returns the number of files
    fetched. Names may use the family-index shorthand."""
    missing = []
    for nm in names:
        full = _expand_short_name(nm, data_dir)
        fname = full if full.endswith(".h5") else full + ".h5"
        candidates = ([os.path.join(data_dir, fname)] if data_dir else [])
        fam = _family_dir(full)
        if fam is not None:
            candidates.append(os.path.join(fam, fname))
        if not any(os.path.isfile(p) for p in candidates):
            missing.append(full)
    if not missing:
        return 0
    print(f"[data] {len(missing)} of {len(names)} instances not local -- "
          f"prefetching before the run (or Ctrl-C and run "
          f"scripts/download_data.py)", flush=True)
    for k, nm in enumerate(missing, 1):
        print(f"[data] ({k}/{len(missing)}) {nm}", flush=True)
        _resolve_path(nm, data_dir)
    return len(missing)


def list_instances(data_dir=None, prefix=None):
    """Sorted instance names available in data_dir.

    Without data_dir, every known family folder (IMRT_lung_data, MM_data,
    MPC_data) is scanned. ``prefix`` keeps only names starting with it,
    e.g. "MM_" or "IMRT_lung_".
    """
    if data_dir is not None:
        dirs = [data_dir]
    elif prefix is not None and _family_dir(prefix) is not None:
        dirs = [_family_dir(prefix)]
    else:
        dirs = [os.path.join(_DATA_ROOT, folder)
                for folder in _FAMILY_DIRS.values()]
    names = []
    for base in dirs:
        if not os.path.isdir(base):
            continue
        names += [os.path.splitext(fn)[0] for fn in os.listdir(base)
                  if fn.endswith(".h5")]
    if prefix:
        names = [nm for nm in names if nm.startswith(prefix)]
    return sorted(set(names))


def _index_of(name):
    """Numeric index embedded in an instance name (MM_017_CVXQP1_L -> 17,
    IMRT_lung_001 -> 1), or None when the name does not carry one."""
    for prefix in sorted(_FAMILY_DIRS, key=len, reverse=True):
        if name.startswith(prefix):
            digits = name[len(prefix):].split("_", 1)[0]
            return int(digits) if digits.isdigit() else None
    parts = name.split("_")           # unknown family: first digit part
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def select_instances(requested, data_dir=None, prefix=None):
    """Resolve a user-supplied problem selection to instance names.

    ``requested`` is either the string "all" (every instance in scope) or a
    list of entries. An entry may be

      * a full instance name        "MM_001_AUG2D", "IMRT_lung_001"
      * a family-index shorthand     "MM_002", "IMRT_lung_1"
      * the original short name      "AUG2D", "CVXQP1_L"  (MM/MPC)
      * the instance number          1, 10, 17  (-> MM_001..., MM_010...)

    so PROBLEMS = [1, 10, 17] and PROBLEMS = list(range(1, 21)) both work.

    The scope is the UNION of the local instance files and the manifest
    (data/instances_metadata.csv), so "all" means the full benchmark family even
    when only part of the data has been downloaded yet -- missing files
    are fetched by the caller's prefetch / on first load. An explicit
    ``data_dir`` overrides this: only that folder's files are in scope.
    """
    available = list_instances(data_dir, prefix)
    if data_dir is None:               # merge in the manifest, so partial
        names = _manifest_names()      # local data never shrinks "all"
        available = sorted(set(available)
                           | {nm for fam in names for nm in names[fam]
                              if not prefix or nm.startswith(prefix)})
    if isinstance(requested, str):
        if requested.lower() == "all":
            return available
        requested = [requested]
    chosen, missing = [], []
    for want in requested:
        # instance number: 17 or "17" -> the instance whose index is 17
        idx = None
        if isinstance(want, (int, np.integer)) and not isinstance(want, bool):
            idx = int(want)
        elif isinstance(want, str) and want.isdigit():
            idx = int(want)
        if idx is not None:
            hits = [nm for nm in available if _index_of(nm) == idx]
            if len(hits) == 1:
                chosen.append(hits[0])
            elif len(hits) > 1:
                raise ValueError(f"instance number {idx} is ambiguous across "
                                 f"families: {hits}; pass a name or a prefix")
            else:
                missing.append(idx)
            continue
        want_h5 = want[:-3] if want.endswith(".h5") else want
        if want_h5 in available:
            chosen.append(want_h5)
            continue
        # family-index shorthand: "MM_002", "IMRT_lung_2" -> that family's
        # instance with index 2
        m = _SHORT_RE.match(want_h5)
        if m:
            prefix, idx = m.group(1), int(m.group(2))
            hits = [nm for nm in available
                    if nm.startswith(prefix) and _index_of(nm) == idx]
            if len(hits) == 1:
                chosen.append(hits[0])
            else:
                missing.append(want)
            continue
        # match on the part after the family prefix ("002_Lung") or after
        # the family+index prefix ("AUG2D", "CVXQP1_L")
        hits = [nm for nm in available
                if nm.split("_", 1)[-1] == want_h5
                or nm.split("_", 2)[-1] == want_h5
                or nm.endswith("_" + want_h5)]
        if len(hits) == 1:
            chosen.append(hits[0])
        elif len(hits) > 1:
            raise ValueError(f"ambiguous problem name {want!r}: {hits}")
        else:
            missing.append(want)
    if missing:
        raise ValueError(f"unknown problem(s): {missing}")
    # keep the caller's order, drop duplicates
    seen, ordered = set(), []
    for nm in chosen:
        if nm not in seen:
            seen.add(nm)
            ordered.append(nm)
    return ordered
