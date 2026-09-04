# Benchmark data

The benchmark instances are HDF5 (`.h5`) files hosted on the Hugging Face
dataset
[Radiotherapy-Optimization/QP-Benchmark](https://huggingface.co/datasets/Radiotherapy-Optimization/QP-Benchmark),
organized in three folders mirrored locally under this directory:

    data/MPC_data/        64 model-predictive-control instances  MPC_001_....h5
    data/MM_data/        137 Maros-Meszaros instances            MM_001_....h5
    data/IMRT_lung_data/  60 radiotherapy (lung IMRT) instances  IMRT_lung_001.h5

Radiotherapy data is organized one folder per (modality, site) family --
IMRT_lung today; future releases may add sibling families such as
IMRT_prostate or VMAT_lung -- and radiotherapy instance names are exactly
<family>_<index>. Everything else about an instance (patient, treatment
protocol, provenance) lives in the manifest (instances_metadata.csv) and
inside each file's ``data_note``, never in folder or file names.

The synthetic suite has no data files: `scripts/run_benchmark_Synthetic.py`
generates its 21 instances in memory from a fixed seed.

<p align="center">
  <img src="../assets/fig2_problem_size.png" width="55%">
</p>
<p align="center"><em>The 282 benchmark instances (stored families plus
the synthetic suite) by number of variables and constraints.</em></p>

## Automatic download

Nothing needs to be fetched by hand. `src/data_loader.py` resolves an
instance name by first looking under the local family folder
(e.g. `data/MM_data/MM_001_AUG2D.h5`,
`data/IMRT_lung_data/IMRT_lung_001.h5`) and, when the file is absent,
downloading it from the Hugging Face dataset into that folder (this
requires the `huggingface_hub` package, installed by requirements.txt).

To download whole families (or everything) up front, use the provided
script -- set `FAMILIES` at its top and run it (re-running is safe;
existing files are skipped):

    python scripts/download_data.py

The benchmark scripts also prefetch whatever their selected problem set
is missing at startup. Under the hood both use `huggingface_hub`
(`snapshot_download` / `hf_hub_download`) into `data/<Family>_data/`.

## File format

Every file stores a convex QP with objective `0.5*x'Qx + q'x + c`.
`Q` (n x n, full symmetric, not a triangle) and the constraint matrix `G`
(m x n) are stored in CSR form as four datasets each
(`Q_data`, `Q_indices`, `Q_indptr`, `Q_shape`, and likewise for `G`);
`q` and `c` are stored directly. Two constraint forms are used:

* **Canonical (radiotherapy files)** -- dataset `h` is present and the problem is

      minimize    0.5 x'Qx + q'x + c
      subject to  G x <= h

* **Two-sided (MM and MPC files)** -- datasets `l_h`, `u_h`, `l_x`, `u_x`
  are present and the problem is

      minimize    0.5 x'Qx + q'x + c
      subject to  l_h <= G x <= u_h,    l_x <= x <= u_x

  with infinite entries encoding absent bounds; a row with
  `l_h[i] == u_h[i]` is an equality constraint. `source_file` records the
  originating file in the source repository.

Every file also carries `format_note` (this format description) and
`data_note` (provenance). MM and MPC files may additionally carry a
reference solution (`x_opt`, `dual`, `obj_val`, `solver`); the radiotherapy files
do not.

`data_loader.load_data` returns the stored form; `to_canonical` converts
either form to one-sided `(Q, q, c, G, h)` (equality rows doubled as +/-
pairs) and `to_split` to `(Q, q, c, G, h, A, b)` with the equality rows
kept separate for solvers that support `Ax = b` natively.

## Families

* **MPC** -- the 64 model-predictive-control instances of the qpbenchmark
  MPC test set (Caron et al., 2024): LIPMWALK, QUADCMPC, WHLIPBAL and
  related control problems, downloaded from
  [qpsolvers/mpc_qpbenchmark](https://github.com/qpsolvers/mpc_qpbenchmark)
  and converted from the qpsolvers-form `.npz` sources into the two-sided
  HDF5 container. Small, mostly well-conditioned; QUADCMPC carries
  equality constraints and variable bounds.

* **MM** -- the 137 instances of the Maros-Meszaros convex QP test set,
  introduced by Maros and Meszaros (1999); the files were downloaded from
  the qpbenchmark distribution
  ([qpsolvers/maros_meszaros_qpbenchmark](https://github.com/qpsolvers/maros_meszaros_qpbenchmark),
  Caron et al., 2024) and converted from its MATLAB (`.mat`) sources.
  Widely varying size and conditioning; many instances are
  ill-conditioned, and some are equality-only.

* **IMRT_lung** -- 60 fluence-map-optimization QPs built from the public
  lung-patient data of [PortPy](https://github.com/PortPy-Project/PortPy)
  (Jhanwar et al., 2023); one instance per patient under the
  `Lung_2Gy_30Fx` protocol, with the manifest's `source` column mapping
  each instance to its PortPy patient (e.g. `IMRT_lung_007` <-
  `PortPy Lung_Patient_8`). Large (up to roughly 900,000 constraints),
  dense rows, inequality-only.

## Manifest

`instances_metadata.csv` lists every stored instance with its
essentials -- one row per instance:

    family        MPC | MM | IMRT_lung
    name          instance name (= file name without .h5)
    source        original problem (MPC/MM); dataset + patient for
                  radiotherapy (e.g. "PortPy Lung_Patient_8")
    n, m          number of variables / constraint rows as stored
    n_equalities  rows with l_h == u_h (0 for canonical radiotherapy files)
    nnz_Q, nnz_G  nonzero counts
    protocol      treatment protocol (radiotherapy only, e.g.
                  "Lung_2Gy_30Fx"); blank for MPC/MM
    formulation   the instance's optimization problem, spelled out (e.g.
                  "min 0.5*x'Qx + q'x + c  s.t.  Gx <= h"), so every row
                  is self-describing
    note          free-form remarks; blank when there is nothing to say

`scripts/generate_paper_figures.py` reads it to size-categorize the IMRT
problems (their benchmark CSV carries no size columns), and
`src/data_loader.py` uses it to expand family-plus-index shorthands
(`"MM_002"`, `"IMRT_lung_1"`) to full instance names -- so the shorthand
works before any data has been downloaded. A copy of the manifest is also
kept at the root of the Hugging Face dataset, so the dataset is
self-describing on its own.

## References

* Caron, S., Zaki, A., Otta, P., Arnstrom, D., Carpentier, J., Yang, F.,
  Leziart, P.-A. (2024). *qpbenchmark: Benchmark for quadratic
  programming solvers available in Python* (Version 2.5.0).
  https://github.com/qpsolvers/qpbenchmark

* Maros, I., Meszaros, C. (1999). A repository of convex quadratic
  programming problems. *Optimization Methods and Software*, 11(1-4),
  671-681.

* Jhanwar, G., Tefagh, M., Taasti, V. T., Alam, S. R., Tuomaala, S.,
  Nadeem, S., Zarepisheh, M. (2023). PortPy: An open-source Python
  package for planning and optimization in radiation therapy including
  benchmark data and algorithms. *AAPM 65th Annual Meeting & Exhibition*.
  https://github.com/PortPy-Project/PortPy
