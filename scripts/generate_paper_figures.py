"""
Every benchmark figure in the paper, regenerated from the CSVs alone --
plus the shared plotting/reporting engine used at the end of each
benchmark run (print_runtime_summary, print_kkt_summary).

Reads the four family CSVs written by the benchmark scripts
(run_benchmark_MPC / _MM / _Synthetic / _IMRT) from ../results and
produces one PDF per panel in ../results/figures, sized for two-across
\\subfloat inclusion, in the paper style (STIX serif, identity solver
colors, log-time boxes with Failed-% bars, log-kkt boxes with
No-result-% bars, black outlier circles beyond the whiskers). No
re-solving is ever needed.

    python generate_paper_figures.py

The per-instance plots (ADMM/IP convergence, residual vs time) live in
the benchmark scripts: they need per-iteration histories that exist only
during a run.
"""

import os

import numpy as np
import pandas as pd

# Family -> default CSV name and display name used in plot titles/files
FAMILIES = {
    'MM':   {'csv': 'MM_solver_benchmark.csv',   'display': 'MM'},
    'MPC':  {'csv': 'MPC_solver_benchmark.csv',  'display': 'MPC'},
    'IMRT': {'csv': 'IMRT_solver_benchmark.csv', 'display': 'IMRT'},
    'Synthetic': {'csv': 'Synthetic_solver_benchmark.csv',
                  'display': 'Synthetic'},
}

# ---------------------------------------------------------------------------
# Paper styling. PRINT_WIDTH is the figure's FINAL width in the paper
# (inches); fonts below are true point sizes at that width. Serif/STIX
# matches the paper's body font.
# ---------------------------------------------------------------------------
SINGLE_WIDTH = 3.6   # standalone single-panel box plots (time / quality)
ITERS_WIDTH = 2.8    # standalone iterations box plot (two solvers)
_RC = {
    'font.family': 'serif',
    'hatch.linewidth': 0.5,   # fine stipple on the dotted %-bars
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Times',
                   'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
}

# fixed identity -> color mapping, identical in EVERY figure regardless
# of which solvers ran. Classic solid palette matching the paper's
# original figures (Warm-IP blue, MOSEK orange, Clarabel green, PIQP red,
# Gurobi purple, OSQP brown); IP gets dark gray so Warm-IP vs IP reads as
# method vs baseline.
SOLVER_COLORS = {'Warm-IP': '#1f77b4', 'IP': '#595959',
                 'MOSEK': '#ff7f0e', 'Clarabel': '#2ca02c',
                 'PIQP': '#d62728', 'Gurobi': '#9467bd',
                 'OSQP': '#8c564b'}
INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'

# mean indicator drawn on every box: a dashed horizontal line (matplotlib
# meanline style). The solid box line stays the median; the numeric label
# reports this mean.
MEAN_PROPS = dict(linestyle='--', linewidth=1.1, color=INK, zorder=4)


def _fmt_stat(v):
    """Compact 2-significant-digit label: plain for everyday magnitudes
    (110, 44, 0.055), short e-notation (4.4e-3, 1.4e3) outside
    [0.01, 1000)."""
    v = float(v)
    if v != 0 and (abs(v) < 0.01 or abs(v) >= 1000):
        m, e = f"{v:.1e}".split('e')
        return f"{m}e{int(e)}"
    s = f"{v:.2g}"
    if 'e' in s:                       # .2g fell back to exponent (e.g. 110)
        s = f"{float(s):.10g}"
    return s


def _cap_linear_axis(ax, positions, data, pct, show_counts=True):
    """Give linear panels breathing room: cap the y-axis at the pct-th
    percentile of all plotted values (never below any whisker top or
    mean, so boxes stay fully visible). With show_counts, each solver
    whose outlier circles were clipped gets a small 'up-arrow k' count;
    without it the clipping is silent (note the cap in the caption)."""
    vals = np.concatenate([v for v in data if v.size]) if data else None
    if vals is None or not vals.size:
        return
    floor = 0.0
    for v in data:
        if not v.size:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        inside = v[v <= q3 + 1.5 * (q3 - q1)]
        hi = inside.max() if inside.size else v.max()
        floor = max(floor, hi, float(np.mean(v)))
    cap = max(np.percentile(vals, pct), floor) * 1.05
    if cap <= 0:
        return
    ax.set_ylim(0, cap)
    if not show_counts:
        return
    for p0, v in zip(positions, data):
        k = int((v > cap).sum())
        if k:
            ax.annotate(f"$\\uparrow${k}", xy=(p0, cap), xytext=(0, -1),
                        textcoords='offset points', ha='center', va='top',
                        fontsize=6, color=INK2, zorder=5)


def _matplotlib():
    import matplotlib
    if os.environ.get("WARMIP_NOSHOW") == "1":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def print_runtime_summary(df):
    time_cols = [c for c in df.columns if c.endswith("_time")]
    if not time_cols:
        print("No runtimes recorded.")
        return
    summary = df[["Problem"] + time_cols].copy()
    summary.columns = ["Problem"] + [c[:-len("_time")] for c in time_cols]
    print()
    print("=" * 90)
    print("Runtime summary (seconds)")
    print("=" * 90)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(summary) > 1:
        means = summary.drop(columns=["Problem"]).mean(numeric_only=True)
        print("-" * 90)
        print("Mean:   " + "  ".join(f"{k}={v:.3f}" for k, v in means.items()))
    print("=" * 90)


def print_kkt_summary(df):
    kkt_cols = [c for c in df.columns if c.endswith("_kkt_max")]
    if not kkt_cols:
        return
    summary = df[["Problem"] + kkt_cols].copy()
    summary.columns = ["Problem"] + [c[:-len("_kkt_max")] for c in kkt_cols]
    print()
    print("=" * 90)
    print("Solution quality summary: max(primal_res, dual_res, dual_gap)")
    print("=" * 90)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    print("=" * 90)


# How points beyond the whiskers are drawn is set by
# OUTLIER_STYLE in the configuration block below.
_TAIL_GRAY = '#a5a49f'


def _tiered_scatter(ax, x0, v, lo, hi, color, rng, jw):
    """Outlier rendering beyond the whiskers, per OUTLIER_STYLE (no
    jitter, no in-box cloud) -- quiet, so the boxes read first."""
    v = np.asarray(v, dtype=float)
    out = v[(v < lo) | (v > hi)]
    if not out.size or OUTLIER_STYLE == 'none':
        return
    if OUTLIER_STYLE == 'circles':
        ax.scatter(np.full(out.size, x0), out, s=4, facecolors='none',
                   edgecolors=_TAIL_GRAY, linewidths=0.5, zorder=4)
        return
    # 'tail': one stroke per breached side, whisker tip -> extreme
    for tip, ext in ((hi, out.max()) if (out > hi).any() else (None, None),
                     (lo, out.min()) if (out < lo).any() else (None, None)):
        if tip is None or not np.isfinite(tip):
            continue
        ax.plot([x0, x0], [tip, ext], lw=0.8, color=_TAIL_GRAY,
                solid_capstyle='butt', zorder=3)
        ax.scatter([x0], [ext], s=6, facecolors='none',
                   edgecolors=INK2, linewidths=0.6, zorder=4)


def _resolve_solvers(df, solvers):
    """Solver list resolution: explicit arg > PLOT_SOLVERS > every solver
    with results in the CSV (identity-color order)."""
    available = [s for s in SOLVER_COLORS
                 if f'{s}_time' in df.columns
                 and pd.to_numeric(df[f'{s}_time'],
                                   errors='coerce').notna().any()]
    wanted = solvers if solvers is not None else PLOT_SOLVERS
    if wanted is not None:
        return [s for s in wanted if s in available]
    return available


def _prep_box_data(df, solvers, fail_kkt_tol):
    """Per-solver returned-result sets and failure counts (the population
    semantics shared by every box figure)."""
    n_all = len(df)
    times, quals, iters, no_result, fails = {}, {}, {}, {}, {}
    for s in solvers:
        t = pd.to_numeric(df[f'{s}_time'], errors='coerce')
        k = pd.to_numeric(df.get(f'{s}_kkt_max'), errors='coerce')
        it_col = df.get(f'{s}_iters')
        returned = t.notna() & k.notna()
        times[s] = t[returned].to_numpy(float)      # own returned-result set
        quals[s] = k[returned].to_numpy(float)      # incl. over-tolerance
        iters[s] = (pd.to_numeric(it_col, errors='coerce')[returned]
                    .to_numpy(float) if it_col is not None else np.zeros(0))
        no_result[s] = int(n_all - returned.sum())
        fails[s] = int(n_all - (returned & (k <= fail_kkt_tol)).sum())
    return n_all, times, quals, iters, no_result, fails


def _draw_box_panel(ax, P, solvers, times, n_all, rng):
    """One box panel: optional companion bar on a secondary % axis, boxes
    (median line plus a white-diamond mean marker), outlier circles, mean
    labels (unless the spec sets mean_labels=False), angled solver names
    on the x-axis (the problem count belongs in the caller's title).

    n_all may be an int (every solver attempted every problem) or a
    per-solver dict of attempted counts (pooled figures where a solver ran
    on only part of the set)."""
    from matplotlib.patches import Patch
    colors = [SOLVER_COLORS[s] for s in solvers]
    xs = np.arange(1, len(solvers) + 1)
    has_bar = P['counts'] is not None
    pos = xs - (0.18 if has_bar else 0.0)
    if has_bar:
        axp = ax.twinx()
        pct = [100.0 * P['counts'][s]
               / (n_all[s] if isinstance(n_all, dict) else n_all)
               for s in solvers]
        _hatch = P.get('bar_hatch')
        axp.bar(xs + 0.24, pct, width=0.18, color=colors, alpha=1.0,
                edgecolor=INK if _hatch else 'none',
                linewidth=0.5 if _hatch else 0.0, hatch=_hatch, zorder=2)
        for x, p in zip(xs + 0.24, pct):
            # vertical, reading bottom-up: one line-height wide, so it
            # stays inside the bar's footprint instead of reaching
            # into the neighboring box; an exact zero is labeled '0%'
            axp.annotate("0%" if p == 0 else f"{p:.1f}%", xy=(x, p),
                         xytext=(0, 2),
                         textcoords='offset points', rotation=90,
                         ha='center', va='bottom',
                         fontsize=6, color=INK2)
        axp.set_ylim(0, 100)
        axp.set_ylabel(P['bar_ylabel'], fontsize=8, color=INK2)
        axp.tick_params(colors=INK2, labelsize=7)
        axp.spines['top'].set_visible(False)
        axp.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        # legend ABOVE the axes (never occludes data); the panel title
        # sits on its own line above it
        from matplotlib.legend_handler import HandlerTuple
        from matplotlib.lines import Line2D
        bar_patch = Patch(facecolor=INK2, alpha=0.6, hatch=_hatch,
                          edgecolor=INK if _hatch else 'none',
                          linewidth=0.5 if _hatch else 0.0)
        handles, labels = [bar_patch], [P['legend']]
        if P.get('box_legend'):
            # box-styled handle: gray box with a median line through it
            handles.insert(0, (Patch(facecolor='0.85', edgecolor=INK,
                                     linewidth=0.7),
                               Line2D([], [], color=INK, lw=1.2)))
            labels.insert(0, P['box_legend'])
        ax.legend(handles=handles, labels=labels,
                  handler_map={tuple: HandlerTuple(ndivide=1)},
                  loc='lower right', bbox_to_anchor=(1.0, 0.99),
                  ncol=len(handles),
                  frameon=False, borderpad=0, handlelength=1.0,
                  handleheight=0.8, fontsize=7,
                  handletextpad=0.4, columnspacing=0.9)

    data = []
    for s in solvers:
        v = np.asarray(P['source'][s], dtype=float)
        if P['log']:
            v = v[np.isfinite(v) & (v > 0)]
        else:
            v = v[np.isfinite(v)]
        data.append(v)

    whisk = {}
    if max((v.size for v in data), default=0) >= 2:
        bp = ax.boxplot(data, positions=pos,
                        widths=0.34 if has_bar else 0.45,
                        patch_artist=True, showfliers=False,
                        showmeans=P.get('show_means', True),
                        meanline=True, meanprops=MEAN_PROPS,
                        medianprops=dict(color=INK, lw=1.6),
                        whiskerprops=dict(color=INK, lw=0.9),
                        capprops=dict(color=INK, lw=0.9))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)          # solid, classic-paper style
            patch.set_edgecolor(INK)
            patch.set_linewidth(0.9)
        for i in range(len(data)):
            los = bp['whiskers'][2 * i].get_ydata()
            his = bp['whiskers'][2 * i + 1].get_ydata()
            whisk[i] = (min(los), max(his))

    jw = 0.085 if has_bar else 0.12
    for i, (p0, v, color) in enumerate(zip(pos, data, colors)):
        lo, hi = whisk.get(i, (-np.inf, np.inf))
        _tiered_scatter(ax, p0, v, lo, hi, color, rng, jw)
    if P.get('show_means', True) and P.get('mean_labels', True):
        for p0, v in zip(pos, data):
            if v.size:
                mu = float(np.mean(v))
                ax.annotate(_fmt_stat(mu), xy=(p0, mu), xytext=(0, 8),
                            textcoords='offset points', ha='center',
                            fontsize=6.5, color=INK, zorder=5,
                            bbox=dict(boxstyle='round,pad=0.15',
                                      fc='white', ec='none', alpha=0.85))

    ax.set_title(P['title'], loc='left', color=INK,
                 pad=15 if has_bar else 6)
    if P['log']:
        ax.set_yscale('log')
    elif P.get('ycap_pct'):
        _cap_linear_axis(ax, pos, data, P['ycap_pct'],
                         show_counts=P.get('clip_counts', True))
    ax.set_xticks(xs)
    ax.set_xticklabels(solvers, rotation=30, ha='right',
                       rotation_mode='anchor')
    ax.set_ylabel(P['ylabel'], color=INK2)
    ax.grid(True, axis='y', ls=':', alpha=0.35)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2)


def _mean_stat(v, kind):
    """Arithmetic or geometric mean of a sample (geometric over the
    positive values)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    if kind == 'geometric':
        w = v[v > 0]
        return float(np.exp(np.log(w).mean())) if w.size else np.nan
    return float(v.mean())


def _draw_meanbar_panel(ax, P, solvers, n_all, mean_kind='arithmetic'):
    """Main-body summary version of a runtime box panel: one solid bar per
    solver showing the mean over its returned results (value labeled on
    top), plus hatched failed-% companion bars on a secondary % axis. The
    full distributions live in the appendix box figures.
    mean_kind: 'arithmetic' or 'geometric'."""
    from matplotlib.patches import Patch
    colors = [SOLVER_COLORS[s] for s in solvers]
    xs = np.arange(1, len(solvers) + 1)
    means = [_mean_stat(P['source'][s], mean_kind) for s in solvers]

    ax.bar(xs - 0.18, means, width=0.34, color=colors, edgecolor=INK,
           linewidth=0.9, zorder=3)
    top = max([m for m in means if np.isfinite(m)], default=1.0)
    ax.set_ylim(0, 1.18 * top if top > 0 else 1.0)
    for x, mu in zip(xs - 0.18, means):
        if np.isfinite(mu):
            label = _fmt_stat(mu)
            # a WIDE value text on a SHORT bar reaches into the rotated
            # %-label column of the neighboring bar; raise it just above
            # that text (long label + bar top below ~20% of the axis)
            dy = 14 if (len(label) > 4 and mu < 0.2 * top) else 2
            ax.annotate(label, xy=(x, mu), xytext=(0, dy),
                        textcoords='offset points', ha='center',
                        va='bottom', fontsize=6.5, color=INK, zorder=5)

    axp = ax.twinx()
    pct = [100.0 * P['counts'][s]
           / (n_all[s] if isinstance(n_all, dict) else n_all)
           for s in solvers]
    axp.bar(xs + 0.24, pct, width=0.18, color=colors, edgecolor=INK,
            linewidth=0.5, hatch='.....', zorder=2)
    for x, p in zip(xs + 0.24, pct):
        axp.annotate("0%" if p == 0 else f"{p:.1f}%", xy=(x, p),
                     xytext=(0, 2),
                     textcoords='offset points', rotation=90,
                     ha='center', va='bottom', fontsize=6, color=INK2)
    axp.set_ylim(0, 100)
    axp.set_ylabel(P['bar_ylabel'], fontsize=8, color=INK2)
    axp.tick_params(colors=INK2, labelsize=7)
    axp.spines['top'].set_visible(False)
    axp.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    mean_word = 'Geo. mean' if mean_kind == 'geometric' else 'Mean'
    ax.legend(handles=[
        Patch(facecolor='0.82', edgecolor=INK, linewidth=0.5,
              label=f'{mean_word} runtime'),
        Patch(facecolor='0.82', edgecolor=INK, linewidth=0.5, hatch='.....',
              label=P['legend']),
    ], loc='lower right', bbox_to_anchor=(1.0, 0.99), ncol=2,
        frameon=False, borderpad=0, handlelength=1.0, handleheight=0.8,
        fontsize=7, handletextpad=0.4, columnspacing=0.9)

    ax.set_title(P['title'], loc='left', color=INK, pad=15)
    ax.set_xticks(xs)
    ax.set_xticklabels(solvers, rotation=30, ha='right',
                       rotation_mode='anchor')
    ax.set_ylabel(P['ylabel'], color=INK2)
    ax.grid(True, axis='y', ls=':', alpha=0.35)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2)


def _panel_specs(times, quals, iters, fails, no_result, iters_panel):
    panels = [
        dict(source=times, ylabel='runtime (s)', title='Runtime', log=True,
             counts=fails, bar_ylabel='failed instances (%)',
             legend='Failed (%)', key='time', bar_hatch='.....',
             box_legend='Runtime (box)'),
    ]
    if iters_panel:
        panels.append(dict(source=iters, ylabel='IP iterations',
                           title='IP iterations', log=False, counts=None,
                           bar_ylabel=None, legend=None, key='iters'))
    panels.append(
        dict(source=quals, ylabel='max(primal, dual, gap)',
             title='Solution quality', log=True, counts=no_result,
             bar_ylabel='no result (%)', legend='No result (%)',
             key='quality', mean_labels=False, bar_hatch='.....',
             box_legend='Quality (box)'))
    return panels



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, '..', 'results')

# One finished-run CSV per benchmark family (edit to point at other runs)
FAMILY_CSVS = {
    'MPC':       os.path.join(_OUT, 'MPC_solver_benchmark.csv'),
    'MM':        os.path.join(_OUT, 'MM_solver_benchmark.csv'),
    'Synthetic': os.path.join(_OUT, 'Synthetic_solver_benchmark.csv'),
    'IMRT':      os.path.join(_OUT, 'IMRT_solver_benchmark.csv'),
}
# n,m for the IMRT problems (their benchmark CSV carries no size columns)
INSTANCES_CSV = os.path.join(_HERE, '..', 'data', 'instances_metadata.csv')

OUTPUT_DIR = os.path.join(_OUT, 'figures')
SAVE_FIGS = True
SHOW_FIGS = True

# Solvers shown in the box figures (Figs 5 and 6): edit this list to
# toggle solvers on/off (order = plot order). None -> every solver with
# results. IP is left out by default (it appears only in the vs-IP
# figure); add 'IP' back here to include it.
PLOT_SOLVERS = ['Warm-IP', 'Gurobi', 'MOSEK', 'Clarabel', 'PIQP',  'OSQP']
# Solvers considered for the performance profiles (Fig 7). None -> same
# as PLOT_SOLVERS. Independently of this list, a solver is DROPPED from a
# profile when it was not attempted on every problem in the category
# (Dolan-More needs a common problem set); drops are printed.
PROFILE_SOLVERS = None

FAIL_KKT_TOL = 1e-3      # failed = no result or kkt_max above this
MEAN_KIND = 'arithmetic'  # bar figures (main body): 'arithmetic' or
                          # 'geometric' mean of each solver's returned
                          # runtimes (labels/legend adapt automatically)
TIME_YSCALE = 'linear'      # y-scale of the runtime box panels: 'log' or
                         # 'linear' (regenerating overwrites the same PDFs)
LINEAR_YCAP_PCT = 95     # linear runtime panels only: cap the y-axis at
                         # this percentile of the plotted values (never
                         # below a whisker or mean)
SHOW_CLIP_COUNTS = False  # True: mark each solver whose outliers were
                          # clipped by the cap with a small up-arrow count
                          # (off for the paper; the caption notes the cap)
OUTLIER_STYLE = 'circles'  # beyond-whisker points: 'circles' (every
                           # outlier as a faint small circle), 'tail' (one
                           # thin line to the most extreme value), 'none'
SIZE_SPLIT = 100_000     # small/medium iff n + m + p <= this
TAU_MAX = 100            # linear tau range of the performance profiles
VSIP_LAYOUTS = ('family', 'pooled')   # which Warm-IP-vs-IP variants to draw

PANEL_WIDTH = 3.2        # inches; two across fit the paper's 6.5" text width
PANEL_HEIGHT = 2.6
FIG5_HEIGHT = 1.8        # shorter Fig 5 panels so the 4x2 grid plus its
                         # caption fits one 9-inch text page
FIG6_HEIGHT = 2.0        # pooled Fig 6 panels: shorter than the generic
                         # panel but taller than Fig 5 (their quality
                         # axes span many more decades)
PROFILE_HEIGHT = 2.4
FIG2_WIDTH, FIG2_HEIGHT = 3.5, 3.2   # problem-size scatter (Figure 2)
FIG2_MIN_COUNT_LABEL = 5  # label markers covering >= this many instances

# Figure 2 is about benchmark FAMILIES, not solvers: classic palette kept
# from the paper's original size figure, names spelled out with their
# short versions (the short version is what the rest of the paper uses).
FAMILY_COLORS = {'MM': '#1f77b4', 'MPC': '#ff7f0e',
                 'Synthetic': '#2ca02c', 'IMRT': '#d62728'}
FIG2_NAMES = {'MM': 'Maros–Mészáros (MM: n={c})',
              'MPC': 'Model predictive control (MPC: n={c})',
              'Synthetic': 'Synthetic (n={c})',
              'IMRT': 'Radiotherapy (IMRT: n={c})'}

CATEGORIES = {'small_medium': 'small/medium', 'large': 'large'}
_FAMILY_ORDER = ('MPC', 'MM', 'Synthetic', 'IMRT')


# ---------------------------------------------------------------------------
# Data loading and size categorization
# ---------------------------------------------------------------------------
def _imrt_nm():
    """Problem-name -> (n, m) for the IMRT (lung) family, from the
    instance manifest data/instances_metadata.csv."""
    nm = {}
    try:
        man = pd.read_csv(INSTANCES_CSV)
    except OSError:
        return nm
    for _, r in man[man['family'] == 'IMRT_lung'].iterrows():
        n, m = pd.to_numeric(r.get('n'), errors='coerce'), \
            pd.to_numeric(r.get('m'), errors='coerce')
        if np.isfinite(n) and np.isfinite(m):
            nm[str(r['name'])] = (float(n), float(m))
    return nm


def _imrt_sizes():
    """Problem-name -> n + m (see _imrt_nm)."""
    return {k: n + m for k, (n, m) in _imrt_nm().items()}


def load_families():
    """{family: DataFrame} with a _size column (n + m + p) on every row."""
    dfs = {}
    for fam in _FAMILY_ORDER:
        df = pd.read_csv(FAMILY_CSVS[fam])
        if {'n', 'm'}.issubset(df.columns):
            size = (pd.to_numeric(df['n'], errors='coerce')
                    + pd.to_numeric(df['m'], errors='coerce'))
            if 'p' in df.columns:
                size = size + pd.to_numeric(df['p'],
                                            errors='coerce').fillna(0)
        elif fam == 'IMRT':
            sizes = _imrt_sizes()
            size = df['Problem'].map(sizes)
            miss = df['Problem'][size.isna()]
            if len(miss):
                print(f"[sizes] {len(miss)} IMRT problems not in the "
                      f"manifest -- counted as large: "
                      f"{', '.join(miss[:5])}{'...' if len(miss) > 5 else ''}")
            size = size.fillna(np.inf)     # IMRT problems are large
        else:
            raise ValueError(f"{fam} CSV has no n/m columns; cannot "
                             f"size-categorize")
        df['_size'] = size
        dfs[fam] = df
    return dfs


def _split_by_size(dfs):
    """{category: {family: sub-DataFrame}} using the paper's size rule."""
    out = {'small_medium': {}, 'large': {}}
    for fam, df in dfs.items():
        small = df['_size'] <= SIZE_SPLIT
        if small.any():
            out['small_medium'][fam] = df[small]
        if (~small).any():
            out['large'][fam] = df[~small]
    for cat, parts in out.items():
        comp = ", ".join(f"{fam} {len(d)}" for fam, d in parts.items())
        total = sum(len(d) for d in parts.values())
        print(f"[sizes] {CATEGORIES[cat]}: {total} problems ({comp})")
    return out


def _attempted(df, s):
    return (f'{s}_time' in df.columns
            and pd.to_numeric(df[f'{s}_time'], errors='coerce').notna().any())


def _pooled_solvers(parts, wanted):
    """Solvers attempted in at least one contributing family, identity
    order, honoring the on/off list."""
    avail = [s for s in SOLVER_COLORS
             if any(_attempted(df, s) for df in parts.values())]
    if wanted is not None:
        avail = [s for s in wanted if s in avail]
    return avail


def _pooled_box_data(parts, solvers, tol):
    """_prep_box_data generalized across families: each solver's box and
    percentages cover only the problems where it was ATTEMPTED (its family
    CSV carries a column), so partial-coverage solvers stay honest."""
    times, quals, iters, no_result, fails, n_att = {}, {}, {}, {}, {}, {}
    for s in solvers:
        T, K, I = [], [], []
        att = nores = fail = 0
        for df in parts.values():
            if f'{s}_time' not in df.columns:
                continue
            t = pd.to_numeric(df[f'{s}_time'], errors='coerce')
            k = pd.to_numeric(df.get(f'{s}_kkt_max'), errors='coerce')
            returned = t.notna() & k.notna()
            att += len(df)
            nores += int(len(df) - returned.sum())
            fail += int(len(df) - (returned & (k <= tol)).sum())
            T.append(t[returned].to_numpy(float))
            K.append(k[returned].to_numpy(float))
            it = df.get(f'{s}_iters')
            if it is not None:
                I.append(pd.to_numeric(it, errors='coerce')[returned]
                         .to_numpy(float))
        times[s] = np.concatenate(T) if T else np.zeros(0)
        quals[s] = np.concatenate(K) if K else np.zeros(0)
        iters[s] = np.concatenate(I) if I else np.zeros(0)
        no_result[s], fails[s], n_att[s] = nores, fail, att
    return n_att, times, quals, iters, no_result, fails


# ---------------------------------------------------------------------------
# Shared single-panel renderer
# ---------------------------------------------------------------------------
def _render_panel(P, solvers, times, n_all, fname,
                  width=PANEL_WIDTH, height=PANEL_HEIGHT):
    """One standalone PDF around _draw_box_panel, with the
    dashed tolerance line on quality panels."""
    plt = _matplotlib()
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(width, height), facecolor='white')
        _draw_box_panel(ax, P, solvers, times, n_all,
                        np.random.default_rng(0))
        if P.get('key') == 'quality':
            ax.axhline(FAIL_KKT_TOL, ls='--', lw=0.8, color=INK2,
                       alpha=0.8, zorder=1)
        fig.tight_layout()
        if SAVE_FIGS:
            out = os.path.join(OUTPUT_DIR, fname)
            fig.savefig(out, dpi=300, facecolor='white')
            print(f"saved {fname}", flush=True)
        if os.environ.get("WARMIP_NOSHOW") != "1":
            plt.show()
        plt.close(fig)


def _render_meanbar(P, solvers, n_all, fname, width=PANEL_WIDTH,
                    height=PANEL_HEIGHT):
    """One standalone mean-runtime bar PDF (main-body summary of a runtime
    box panel) around _draw_meanbar_panel."""
    plt = _matplotlib()
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(width, height), facecolor='white')
        _draw_meanbar_panel(ax, P, solvers, n_all, mean_kind=MEAN_KIND)
        fig.tight_layout()
        if SAVE_FIGS:
            out = os.path.join(OUTPUT_DIR, fname)
            fig.savefig(out, dpi=300, facecolor='white')
            print(f"saved {fname}", flush=True)
        if os.environ.get("WARMIP_NOSHOW") != "1":
            plt.show()
        plt.close(fig)


def _style_axes(ax):
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2)


# ---------------------------------------------------------------------------
# Figure 2: problem-size scatter over every benchmarked instance
# ---------------------------------------------------------------------------
def figure_problem_size(dfs):
    """Distribution of benchmark instances by size (paper Figure 2):
    every benchmarked instance at (n, total constraints), log-log.
    Instances with identical dimensions merge into one marker whose AREA
    grows with sqrt(count); markers covering >= FIG2_MIN_COUNT_LABEL
    instances get an explicit x-count label (MPC's 2x30, MM's x12).
    Dashed curve: the m + n = SIZE_SPLIT boundary between the size
    regimes, with the regions lightly shaded as in the original figure."""
    from collections import Counter
    from matplotlib.lines import Line2D
    plt = _matplotlib()

    pts = {}
    for fam in _FAMILY_ORDER:
        df = dfs[fam]
        if {'n', 'm'}.issubset(df.columns):
            n = pd.to_numeric(df['n'], errors='coerce')
            m = pd.to_numeric(df['m'], errors='coerce')
            if 'p' in df.columns:
                m = m + pd.to_numeric(df['p'], errors='coerce').fillna(0)
            pairs = list(zip(n, m))
        else:                                    # IMRT: manifest join
            nm = _imrt_nm()
            pairs = [nm[p] for p in df['Problem'] if p in nm]
            if len(pairs) < len(df):
                print(f"[fig2] {len(df) - len(pairs)} {fam} problems not "
                      f"in the manifest -- omitted from the size figure")
        pts[fam] = Counter((float(a), float(b)) for a, b in pairs
                           if np.isfinite(a) and np.isfinite(b))

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(FIG2_WIDTH, FIG2_HEIGHT),
                               facecolor='white')
        lo, hi = 1.0, 1e6
        ns = np.logspace(0, np.log10(SIZE_SPLIT - 1), 400)
        ms = SIZE_SPLIT - ns
        ax.fill_between(ns, lo, ms, color='#fdf6e3', zorder=0, lw=0)
        ax.fill_between(ns, ms, hi, color='#fdeae6', zorder=0, lw=0)
        ax.fill_between([SIZE_SPLIT, hi], lo, hi, color='#fdeae6',
                        zorder=0, lw=0)
        ax.plot(ns, ms, ls='--', lw=1.1, color=INK2, zorder=1)

        for fam in ('MM', 'MPC', 'Synthetic', 'IMRT'):
            color = FAMILY_COLORS[fam]
            xs = [k[0] for k in pts[fam]]
            ys = [k[1] for k in pts[fam]]
            ss = [16.0 * np.sqrt(c) for c in pts[fam].values()]
            ax.scatter(xs, ys, s=ss, color=color, alpha=0.85,
                       edgecolors=INK, linewidths=0.5, zorder=3)
            for (x, y), c in pts[fam].items():
                if c >= FIG2_MIN_COUNT_LABEL:
                    ax.annotate(f"$\\times${c}", xy=(x, y), xytext=(7, 5),
                                textcoords='offset points', fontsize=7,
                                color=INK, zorder=6,
                                bbox=dict(boxstyle='round,pad=0.15',
                                          fc='white', ec='none',
                                          alpha=0.85))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        # slight y headroom so markers sitting at m = 1e6 are not clipped
        ax.set_ylim(lo, hi * 2.0)
        ax.set_yticks([10.0 ** k for k in range(0, 7)])
        ax.set_xlabel('Number of Variables (n) (log scale)')
        ax.set_ylabel('Number of Constraints (m) (log scale)')
        ax.text(3.5, 3.5e4, 'Small/Medium', fontsize=10,
                fontweight='bold', color=INK)
        ax.text(3.5, 3.5e5, 'Large', fontsize=10,
                fontweight='bold', color=INK)
        ax.text(1.35e5, 2.5e3, 'm+n=100000', rotation=90, fontsize=7.5,
                color=INK2, ha='left', va='bottom')
        _style_axes(ax)
        handles = [Line2D([], [], marker='o', ls='', markersize=5,
                          markerfacecolor=FAMILY_COLORS[f],
                          markeredgecolor=INK, markeredgewidth=0.5,
                          label=FIG2_NAMES[f].format(
                              c=sum(pts[f].values())))
                   for f in ('MM', 'MPC', 'Synthetic', 'IMRT')]
        ax.legend(handles=handles, loc='lower right', frameon=True,
                  framealpha=0.92, facecolor='white', edgecolor='#cccccc')
        fig.tight_layout()
        if SAVE_FIGS:
            fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_problem_size.pdf'),
                        dpi=300, facecolor='white')
            print("saved fig2_problem_size.pdf", flush=True)
        if os.environ.get("WARMIP_NOSHOW") != "1":
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figures 4 and 8: per-family mean-runtime bars + runtime boxes
# ---------------------------------------------------------------------------
def figure_per_family(dfs):
    for fam in _FAMILY_ORDER:
        df, disp = dfs[fam], FAMILIES[fam]['display']
        solvers = _resolve_solvers(df, PLOT_SOLVERS)
        if len(solvers) < 2:
            print(f"[fig5] {fam}: fewer than 2 solvers with results, "
                  f"skipped")
            continue
        n_all, times, quals, iters, no_result, fails = _prep_box_data(
            df, solvers, FAIL_KKT_TOL)
        for P in _panel_specs(times, quals, iters, fails, no_result,
                              iters_panel=False):
            P = dict(P)
            P['title'] = f"{P['title']} — {disp} (n={n_all})"
            if P['key'] == 'time':
                P['log'] = TIME_YSCALE == 'log'
                if not P['log']:
                    P['ycap_pct'] = LINEAR_YCAP_PCT
                    P['clip_counts'] = SHOW_CLIP_COUNTS
                # appendix version: full distributions, no mean marks
                P['show_means'] = False
                # main-body version: mean bars (MEAN_KIND) + failed-% bars
                _render_meanbar(dict(P, ylabel='mean runtime (s)'),
                                solvers, n_all,
                                f"fig4_runtime_mean_{fam}.pdf", height=FIG5_HEIGHT)
            fname = (f"fig8_runtime_box_{fam}.pdf" if P['key'] == 'time'
                     else f"quality_box_{fam}.pdf")   # extra, not in paper
            _render_panel(P, solvers, times, n_all, fname,
                          height=FIG5_HEIGHT)


# ---------------------------------------------------------------------------
# Figures 5 and 9: pooled by problem size (mean bars, quality, boxes)
# ---------------------------------------------------------------------------
def figure_pooled(by_size):
    for cat, parts in by_size.items():
        if not parts:
            print(f"[fig6] no {CATEGORIES[cat]} problems, skipped")
            continue
        solvers = _pooled_solvers(parts, PLOT_SOLVERS)
        if len(solvers) < 2:
            print(f"[fig6] {CATEGORIES[cat]}: fewer than 2 solvers, "
                  f"skipped")
            continue
        n_att, times, quals, iters, no_result, fails = _pooled_box_data(
            parts, solvers, FAIL_KKT_TOL)
        solvers = [s for s in solvers if n_att[s] > 0]
        total = sum(len(d) for d in parts.values())
        for P in _panel_specs(times, quals, iters, fails, no_result,
                              iters_panel=False):
            P = dict(P)
            P['title'] = f"{P['title']} — {CATEGORIES[cat]} (n={total})"
            if P['key'] == 'time':
                P['log'] = TIME_YSCALE == 'log'
                if not P['log']:
                    P['ycap_pct'] = LINEAR_YCAP_PCT
                    P['clip_counts'] = SHOW_CLIP_COUNTS
                P['show_means'] = False
                _render_meanbar(dict(P, ylabel='mean runtime (s)'),
                                solvers, n_att,
                                f"fig5_runtime_mean_{cat}.pdf", height=FIG6_HEIGHT)
            fname = (f"fig9_runtime_box_{cat}.pdf" if P['key'] == 'time'
                     else f"fig5_quality_{cat}.pdf")
            _render_panel(P, solvers, times, n_att, fname,
                          height=FIG6_HEIGHT)


# ---------------------------------------------------------------------------
# Figure 6: performance profiles by size category (linear tau)
# ---------------------------------------------------------------------------
def _profile_ratios(parts, solvers, tol):
    """Per-solver Dolan-More performance ratios over every problem in the
    category. A solve counts only when it returned a result within the
    accuracy tolerance; otherwise its ratio is +inf (the paper's
    'sufficiently large ratio'), which keeps the problem in the
    denominator."""
    ratios = {s: [] for s in solvers}
    for df in parts.values():
        tm = pd.DataFrame(index=df.index)
        for s in solvers:
            t = pd.to_numeric(df[f'{s}_time'], errors='coerce')
            k = pd.to_numeric(df.get(f'{s}_kkt_max'), errors='coerce')
            ok = t.notna() & k.notna() & (k <= tol)
            tm[s] = t.where(ok).replace(0, 1e-10)
        best = tm.min(axis=1)
        for s in solvers:
            r = np.array((tm[s] / best).to_numpy(float), copy=True)
            r[~np.isfinite(r)] = np.inf
            ratios[s].append(r)
    return {s: np.concatenate(v) for s, v in ratios.items()}


# Fixed marker per solver in the profile panels: sparse hollow markers keep
# the six curves distinguishable in black-and-white print.
PROFILE_MARKERS = {'Warm-IP': 'o', 'Gurobi': 's', 'MOSEK': '^',
                   'Clarabel': 'D', 'PIQP': 'v', 'OSQP': 'X'}


def figure_profiles(by_size):
    plt = _matplotlib()
    for cat, parts in by_size.items():
        if not parts:
            continue
        wanted = PROFILE_SOLVERS if PROFILE_SOLVERS is not None \
            else PLOT_SOLVERS
        candidates = _pooled_solvers(parts, wanted)
        # strict Dolan-More coverage: attempted on EVERY problem in the set
        solvers, skipped = [], []
        for s in candidates:
            (solvers if all(f'{s}_time' in df.columns
                            for df in parts.values())
             else skipped).append(s)
        if skipped:
            print(f"[fig7] {CATEGORIES[cat]}: skipped "
                  f"{', '.join(skipped)} (not attempted on every problem "
                  f"in the category)")
        if len(solvers) < 2:
            print(f"[fig7] {CATEGORIES[cat]}: fewer than 2 fully-covering "
                  f"solvers, skipped")
            continue
        ratios = _profile_ratios(parts, solvers, FAIL_KKT_TOL)
        n_prob = len(next(iter(ratios.values())))
        taus = np.linspace(1.0, float(TAU_MAX), 400)
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PROFILE_HEIGHT),
                                   facecolor='white')
            for j, s in enumerate(solvers):
                rho = [(ratios[s] <= tau).mean() for tau in taus]
                ax.plot(taus, rho, lw=1.7, label=s,
                        color=SOLVER_COLORS.get(s, INK2),
                        marker=PROFILE_MARKERS.get(s),
                        markevery=(10 + 13 * j, 85),
                        ms=3.6, mfc='white', mew=0.9)
            ax.set_xlabel(r"Performance Ratio $\tau$")
            ax.set_ylabel(r"Fraction of Problems $\rho_s(\tau)$")
            ax.set_title(f"Performance profile — {CATEGORIES[cat]}",
                         loc='left', color=INK)
            ax.set_xlim(1, TAU_MAX)
            ax.set_xticks([1] + list(range(20, int(TAU_MAX) + 1, 20)))
            ax.set_ylim(-0.02, 1.05)
            _style_axes(ax)
            ax.legend(loc="lower right", frameon=False)
            fig.tight_layout()
            if SAVE_FIGS:
                fname = f"fig6_profile_{cat}.pdf"
                fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
                print(f"saved {fname} ({len(solvers)} solvers, "
                      f"{n_prob} problems)", flush=True)
            if os.environ.get("WARMIP_NOSHOW") != "1":
                plt.show()
            plt.close(fig)


# ---------------------------------------------------------------------------
# Figures 7 and 10: Warm-IP vs IP over all problems (means + boxes)
# ---------------------------------------------------------------------------
_VSIP = ('Warm-IP', 'IP')


def _vsip_collect(dfs):
    """{family: {solver: {'time': arr, 'iters': arr}}} over returned
    results, plus per-family problem counts."""
    data, counts = {}, {}
    for fam in _FAMILY_ORDER:
        df = dfs[fam]
        if not all(f'{s}_time' in df.columns for s in _VSIP):
            print(f"[vsip] {fam}: missing Warm-IP/IP columns, skipped")
            continue
        data[fam], counts[fam] = {}, len(df)
        for s in _VSIP:
            t = pd.to_numeric(df[f'{s}_time'], errors='coerce')
            k = pd.to_numeric(df.get(f'{s}_kkt_max'), errors='coerce')
            it = pd.to_numeric(df.get(f'{s}_iters'), errors='coerce')
            returned = t.notna() & k.notna()
            data[fam][s] = {
                'time': t[returned].to_numpy(float),
                'iters': (it[returned].to_numpy(float)
                          if f'{s}_iters' in df.columns else np.zeros(0)),
                'failed': int(len(df) - (returned & (k <= FAIL_KKT_TOL))
                              .sum()),
            }
    return data, counts


def _vsip_family_panel(ax, data, counts, key):
    """Grouped boxes into ``ax``: x-axis benchmark family, a Warm-IP and an
    IP box in each group. Called once per panel of the combined
    two-panel figure (single PDF, so the panels stay aligned)."""
    from matplotlib.patches import Patch
    fams = list(data)
    log = key == 'time' and TIME_YSCALE == 'log'
    if True:
        xs = np.arange(1, len(fams) + 1)
        rng = np.random.default_rng(0)
        cap_pos, cap_arrs = [], []      # for the linear-scale axis cap
        for s, off in zip(_VSIP, (-0.19, 0.19)):
            color = SOLVER_COLORS[s]
            arrs = []
            for fam in fams:
                v = np.asarray(data[fam][s][key], dtype=float)
                v = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
                arrs.append(v)
            pos = xs + off
            cap_pos.extend(pos)
            cap_arrs.extend(arrs)
            whisk = {}
            if max((v.size for v in arrs), default=0) >= 2:
                bpx = ax.boxplot(arrs, positions=pos, widths=0.3,
                                 patch_artist=True, showfliers=False,
                                 showmeans=False,
                                 medianprops=dict(color=INK, lw=1.4),
                                 whiskerprops=dict(color=INK, lw=0.9),
                                 capprops=dict(color=INK, lw=0.9))
                for patch in bpx['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(1.0)   # solid, classic-paper style
                    patch.set_edgecolor(INK)
                    patch.set_linewidth(0.9)
                for i in range(len(arrs)):
                    los = bpx['whiskers'][2 * i].get_ydata()
                    his = bpx['whiskers'][2 * i + 1].get_ydata()
                    whisk[i] = (min(los), max(his))
            for i, (p0, v) in enumerate(zip(pos, arrs)):
                lo, hi = whisk.get(i, (-np.inf, np.inf))
                _tiered_scatter(ax, p0, v, lo, hi, color, rng, 0.07)

        ax.set_title("Runtime — Warm-IP vs IP" if key == 'time'
                     else "IP iterations — Warm-IP vs IP",
                     loc='left', color=INK, pad=15)
        if log:
            ax.set_yscale('log')
        elif key == 'time':
            _cap_linear_axis(ax, cap_pos, cap_arrs, LINEAR_YCAP_PCT,
                                show_counts=SHOW_CLIP_COUNTS)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{FAMILIES[f]['display']}\n(n={counts[f]})"
                            for f in fams])
        ax.set_ylabel('runtime (s)' if key == 'time' else 'IP iterations',
                      color=INK2)
        _style_axes(ax)
        ax.legend(handles=[Patch(facecolor=SOLVER_COLORS[s], alpha=1.0,
                                 edgecolor=INK, linewidth=0.5, label=s)
                           for s in _VSIP],
                  loc='lower right', bbox_to_anchor=(1.0, 0.99), ncol=2,
                  frameon=False, borderpad=0, handlelength=1.0,
                  handleheight=0.8, fontsize=7, handletextpad=0.4,
                  columnspacing=0.9)


def _vsip_meanbar_panel(ax, data, counts, key):
    """Mean bars into ``ax``: one family group per x position, a solid
    Warm-IP bar and a diagonally hatched IP bar (means over each method's
    returned results, MEAN_KIND statistic, value labeled on top)."""
    from matplotlib.patches import Patch
    fams = list(data)
    xs = np.arange(1, len(fams) + 1)
    handles = []
    for solver, off, hatch in ((_VSIP[0], -0.19, None),
                               (_VSIP[1], 0.19, '///')):
        color = SOLVER_COLORS[solver]
        means = [_mean_stat(data[fam][solver][key], MEAN_KIND)
                 for fam in fams]
        ax.bar(xs + off, means, width=0.34, color=color, edgecolor=INK,
               linewidth=0.9, hatch=hatch, zorder=3)
        for x, mu in zip(xs + off, means):
            if np.isfinite(mu):
                ax.annotate(_fmt_stat(mu), xy=(x, mu), xytext=(0, 2),
                            textcoords='offset points', ha='center',
                            va='bottom', fontsize=6.5, color=INK, zorder=5)
        handles.append(Patch(facecolor=color, edgecolor=INK, linewidth=0.5,
                             hatch=hatch, label=solver))
    tops = [_mean_stat(data[f][sv][key], MEAN_KIND)
            for f in fams for sv in _VSIP]
    top = max([t for t in tops if np.isfinite(t)], default=1.0)
    ax.set_ylim(0, 1.18 * top if top > 0 else 1.0)
    ax.set_title("Runtime — Warm-IP vs IP" if key == 'time'
                 else "IP iterations — Warm-IP vs IP",
                 loc='left', color=INK, pad=15)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{FAMILIES[f]['display']}\n(n={counts[f]})"
                        for f in fams])
    ax.set_ylabel('mean runtime (s)' if key == 'time'
                  else 'mean IP iterations', color=INK2)
    _style_axes(ax)
    ax.legend(handles=handles, loc='lower right',
              bbox_to_anchor=(1.0, 0.99), ncol=2, frameon=False,
              borderpad=0, handlelength=1.0, handleheight=0.8, fontsize=7,
              handletextpad=0.4, columnspacing=0.9)


def _vsip_meanbar_figure(data, counts, fname='fig7_warmip_vs_ip_mean.pdf'):
    """Main-body version of the family comparison: mean bars for both
    panels in one figure -> one PDF (the box version moves to the
    appendix)."""
    plt = _matplotlib()
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_WIDTH + 0.1,
                                                PANEL_HEIGHT),
                                 facecolor='white')
        _vsip_meanbar_panel(axes[0], data, counts, 'time')
        _vsip_meanbar_panel(axes[1], data, counts, 'iters')
        fig.tight_layout(w_pad=1.6)
        if SAVE_FIGS:
            fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300,
                        facecolor='white')
            print(f"saved {fname}", flush=True)
        if os.environ.get("WARMIP_NOSHOW") != "1":
            plt.show()
        plt.close(fig)


def _vsip_family_figure(data, counts, fname='fig10_warmip_vs_ip_box.pdf'):
    """BOTH family panels (runtime, IP iterations) in one figure -> one
    PDF, so the two panels are aligned by construction in the paper."""
    plt = _matplotlib()
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_WIDTH + 0.1,
                                                PANEL_HEIGHT),
                                 facecolor='white')
        _vsip_family_panel(axes[0], data, counts, 'time')
        _vsip_family_panel(axes[1], data, counts, 'iters')
        fig.tight_layout(w_pad=1.6)
        if SAVE_FIGS:
            fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300,
                        facecolor='white')
            print(f"saved {fname}", flush=True)
        if os.environ.get("WARMIP_NOSHOW") != "1":
            plt.show()
        plt.close(fig)


def figure_vsip(dfs):
    data, counts = _vsip_collect(dfs)
    if not data:
        return
    if 'family' in VSIP_LAYOUTS:
        _vsip_meanbar_figure(data, counts)  # main body: mean bars
        _vsip_family_figure(data, counts)   # appendix: box distributions
    if 'pooled' in VSIP_LAYOUTS:
        n_all = sum(counts.values())
        times = {s: np.concatenate([data[f][s]['time'] for f in data])
                 for s in _VSIP}
        iters = {s: np.concatenate([data[f][s]['iters'] for f in data])
                 for s in _VSIP}
        fails = {s: sum(data[f][s]['failed'] for f in data) for s in _VSIP}
        P = dict(source=times, ylabel='runtime (s)',
                 title=f'Runtime — Warm-IP vs IP (n={n_all})',
                 log=TIME_YSCALE == 'log',
                 counts=fails, bar_ylabel='failed instances (%)',
                 legend='Failed (%)', key='time', bar_hatch='.....',
                 box_legend='Runtime (box)')
        if not P['log']:
            P['ycap_pct'] = LINEAR_YCAP_PCT
            P['clip_counts'] = SHOW_CLIP_COUNTS
        _render_panel(P, list(_VSIP), times, n_all, 'warmip_vs_ip_time_pooled.pdf')
        if all(iters[s].size for s in _VSIP):
            P = dict(source=iters, ylabel='IP iterations',
                     title=f'IP iterations — Warm-IP vs IP (n={n_all})',
                     log=False, counts=None, bar_ylabel=None, legend=None,
                     key='iters')
            _render_panel(P, list(_VSIP), times, n_all,
                          'warmip_vs_ip_iters_pooled.pdf')


# ---------------------------------------------------------------------------
def main():
    if not SHOW_FIGS:
        os.environ["WARMIP_NOSHOW"] = "1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[generate_paper_figures] output -> {OUTPUT_DIR}")
    dfs = load_families()
    by_size = _split_by_size(dfs)
    figure_problem_size(dfs)       # Fig 2
    figure_per_family(dfs)         # Figs 4 and 8
    figure_pooled(by_size)         # Figs 5 and 9
    figure_profiles(by_size)       # Fig 6
    figure_vsip(dfs)               # Figs 7 and 10
    print("[generate_paper_figures] done")


if __name__ == "__main__":
    main()
