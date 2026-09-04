"""
Download the benchmark data up front.

Every benchmark instance downloads automatically the first time it is
requested, so running this script is never required -- but pre-fetching
avoids download pauses interleaved with a long benchmark run. Set
FAMILIES below and run

    python download_data.py

Re-running is always safe: files already present are skipped, and an
interrupted download resumes where it left off. The MPC and MM families
are small; the IMRT_lung family is by far the largest (several GB).

The files come from the Hugging Face dataset
https://huggingface.co/datasets/Radiotherapy-Optimization/QP-Benchmark
into data/<Family>_data/. The synthetic family has no data files (its
instances are generated in memory by run_benchmark_Synthetic.py).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))

# ---------------------------------------------------------------------------
# Which families to download: "all", or a list from
# {"MPC", "MM", "IMRT_lung"} (e.g. FAMILIES = ["MPC", "MM"] to skip the
# large radiotherapy set).
# ---------------------------------------------------------------------------
FAMILIES = "all"

import os

from data_loader import HF_DATASET, _DATA_ROOT, _FAMILY_DIRS


def main():
    from huggingface_hub import snapshot_download

    wanted = ([d.rstrip('_') for d in _FAMILY_DIRS]
              if FAMILIES == "all" else list(FAMILIES))
    bad = [f for f in wanted if f + '_' not in _FAMILY_DIRS]
    if bad:
        raise SystemExit(f"unknown families {bad}; choose from "
                         f"{[d.rstrip('_') for d in _FAMILY_DIRS]}")

    for fam in wanted:
        folder = _FAMILY_DIRS[fam + '_']
        local = os.path.join(_DATA_ROOT, folder)
        n_before = len([f for f in os.listdir(local)
                        if f.endswith('.h5')]) if os.path.isdir(local) else 0
        print(f"[{fam}] downloading into data/{folder}/ "
              f"({n_before} files already local) ...", flush=True)
        snapshot_download(repo_id=HF_DATASET, repo_type="dataset",
                          allow_patterns=f"{folder}/*",
                          local_dir=_DATA_ROOT)
        n_after = len([f for f in os.listdir(local) if f.endswith('.h5')])
        print(f"[{fam}] done: {n_after} files "
              f"({n_after - n_before} new)", flush=True)

    print("[download_data] complete", flush=True)


if __name__ == '__main__':
    main()
