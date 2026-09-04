"""
Run all four solver benchmarks sequentially and report progress.

Order: MPC -> Synthetic -> MM -> IMRT.

Each benchmark script is executed as its own subprocess (its own
__main__, its own fault-isolation workers), so a crash or Ctrl-C in one
benchmark never takes down the others: the runner records the failure
and moves on. All solver output streams to this console as usual; the
runner adds [runner] banner lines so you can always see where in the
process it is, and prints a final summary with per-benchmark status,
elapsed time, and the CSV each run wrote.

Interactive figure windows are suppressed for the child runs
(WARMIP_NOSHOW=1): a plt.show() popup would otherwise pause an
unattended run after the first benchmark until someone closed it.

    python run_all_benchmarks.py
"""

import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

# (label, script, CSV the run writes)
BENCHMARKS = [
    ("MPC",       "run_benchmark_MPC.py",
     os.path.join("..", "results", "MPC_solver_benchmark.csv")),
    ("Synthetic", "run_benchmark_Synthetic.py",
     os.path.join("..", "results", "Synthetic_solver_benchmark.csv")),
    ("MM",        "run_benchmark_MM.py",
     os.path.join("..", "results", "MM_solver_benchmark.csv")),
    ("IMRT",      "run_benchmark_IMRT.py",
     os.path.join("..", "results", "IMRT_solver_benchmark.csv")),
]


def _hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _banner(text):
    line = "=" * 79
    print(f"\n{line}\n[runner] {text}\n{line}", flush=True)


def main():
    env = dict(os.environ)
    env["WARMIP_NOSHOW"] = "1"     # no blocking figure windows overnight

    results = []
    t_all = time.time()
    _banner(f"Benchmark suite started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            f" -- {len(BENCHMARKS)} benchmarks: "
            + " -> ".join(label for label, *_ in BENCHMARKS))

    for k, (label, script, csv_rel) in enumerate(BENCHMARKS, 1):
        _banner(f"[{k}/{len(BENCHMARKS)}] {label}: starting {script} "
                f"at {time.strftime('%H:%M:%S')}")
        t0 = time.time()
        try:
            rc = subprocess.call([sys.executable, "-u", script],
                                 cwd=_HERE, env=env)
        except KeyboardInterrupt:
            _banner(f"[{k}/{len(BENCHMARKS)}] {label}: interrupted by user "
                    f"after {_hms(time.time() - t0)} -- moving on")
            results.append((label, "INTERRUPTED", time.time() - t0, csv_rel))
            continue
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        _banner(f"[{k}/{len(BENCHMARKS)}] {label}: {status} in "
                f"{_hms(elapsed)}")
        results.append((label, status, elapsed, csv_rel))

    _banner(f"Benchmark suite finished at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            f" (total {_hms(time.time() - t_all)})")
    print(f"{'Benchmark':<12} {'Status':<20} {'Elapsed':>9}   Results CSV",
          flush=True)
    print("-" * 79, flush=True)
    for label, status, elapsed, csv_rel in results:
        path = os.path.join(_HERE, csv_rel)
        note = csv_rel if os.path.exists(path) else f"{csv_rel} (NOT FOUND)"
        print(f"{label:<12} {status:<20} {_hms(elapsed):>9}   {note}",
              flush=True)
    return 0 if all(s == "OK" for _, s, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
