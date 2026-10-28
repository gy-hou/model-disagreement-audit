# Reproducing the research code

This guide accompanies the `kbs-final-r5` public release. `VALIDATION.json` records which commands were executed in an isolated copy and the observed outcomes. The source and aggregate results were selected from the validated final-r5 research archive; `SOURCE_MANIFEST.json` preserves their hashes.

## Scope and environment

The public checkout can reproduce exact calculations, inspect completed synthetic runs, redraw displays from aggregate results and run the four implementation suites. Reconstructing the mortgage and discovery analyses from individual cases requires the excluded source data, row mappings, prediction caches and fitted model objects. The repository is a retrospective mechanism/diagnostic research record, not an independent prospective replication or an implementation of human/agent review.

Use this repository's `requirements.txt`, which contains the recorded Python 3.14.3 environment pins. The current local `.venv` agrees with NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9, pytest 9.0.3, interpret 0.7.8 and xgboost 3.2.0. These are the versions used by the bounded release checks.

The r1 presentation builder additionally needs Pillow; the inspected local version is 12.2.0. PDF-only QA scripts that import `fitz` need PyMuPDF, which is not installed in this `.venv`; those scripts are optional and must not be included in a minimal validation promise. The r3 Figure 1 builder requires **Arial** (including bold/italic) and the Poppler `pdffonts` executable. Its assertion intentionally rejects a fallback font. Byte-identical rendering across different fonts or Matplotlib versions is not guaranteed.

Suggested environment setup from the public checkout root:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install Pillow==12.2.0
```

The Pillow pin is also provided in `requirements-render.txt`. Fonts and Poppler remain system dependencies; do not distribute the proprietary Arial font.

## Bounded validation commands

```sh
.venv/bin/python src/mechanism_v1_10.py design
.venv/bin/python idea-update/scripts/00_exact_construction.py
.venv/bin/python idea-update/scripts/08_theory_family.py
.venv/bin/python -m pytest -q tests/test_mechanism_v1_9.py tests/test_audit_revision_v1_9.py tests/test_candidate_rebuild_v1_9.py tests/test_mechanism_v1_10.py
```

The first three commands calculate exact finite-support quantities and print them. They do not read empirical row data or run the completed 300-seed experiments. The four suites contain 40 implementation tests. The tests exercise synthetic fixtures, small numerical fits and resampling, leakage/target/queue rules, and overwrite/freeze guards using temporary outputs; they do not retrain production credit models or regenerate the reported scientific experiments. See `VALIDATION.json` for the actual test outcome.

Tests import the training interfaces even when they only check configurations. Therefore keep all archived `src/*.py` dependencies and these three configuration files:

```text
configs/m1_pool_wave2_5.json
configs/m2_pool_wave2_5.json
configs/m7_pool_wave2_5.json
```

The synthetic protocol tests also hash recorded historical inputs. Keep the archived aggregate `results/tables/v1_4_score_control/`, `results/tables/v1_9_mechanism/` and `results/tables/v1_10_mechanism/` lineage files; they are not raw borrower data.

## Rebuild the current figures from completed results

Use a disposable clone/worktree because the original builders write fixed legacy output paths.

```sh
.venv/bin/python docs/revisions/kbs_final_r1/rebuild_figures.py
.venv/bin/python docs/revisions/kbs_final_r3/rebuild_figure1.py
```

The first command redraws all seven r1-style main figures from completed aggregates and checks that its newly exported plotted-value tables exactly equal the archived tables. The current r5 Figure 1 comes from the **second** command and supersedes the r1 design. r5 Figures 2–7 are the r1 artwork unchanged. Output mapping:

| Current r5 figure | Builder output |
|---|---|
| 1 | `paper/kbs_final_r3/figures/protocol_v1_10.{pdf,svg,png}` |
| 2 | `paper/kbs_final_r1/figures/historical_targets_v1_10.{pdf,svg,png}` |
| 3 | `paper/kbs_final_r1/figures/score_geometry_v1_10.{pdf,svg,png}` |
| 4 | `paper/kbs_final_r1/figures/final_mechanisms.{pdf,svg,png}` |
| 5 | `paper/kbs_final_r1/figures/incremental_capture_v1_10.{pdf,svg,png}` |
| 6 | `paper/kbs_final_r1/figures/frozen_selection_v1_10.{pdf,svg,png}` |
| 7 | `paper/kbs_final_r1/figures/candidate_stability_v1_10.{pdf,svg,png}` |

These `paper/` directories hold figure artifacts; the builders do not need manuscript drafts. Keep the three original `paper/kbs_final_r1/figures/final_mechanisms.{pdf,svg,png}` baseline files: the construction renderer hashes them during the r1 rebuild. They are included alongside the authoritative current r5 figures. The r3 builder also writes an identical `paper/kbs_final_r3_zh/figures/` copy. It writes `FIGURE1_QA.json` under its existing script directory. The r1 builder writes QA, plotted values and a montage under `docs/revisions/kbs_final_r1/`.

The r1 builder dynamically imports `src/render_revision_v1_10.py` and `docs/revisions/kbs_final/final_construction_figure.py`. It also imports `docs/revisions/kbs_final_r1/nature_qa/audit_panel_alignment.py`; retain that utility's Apache-2.0 license. Keep `docs/revisions/kbs_final/final_refit_summary.csv`: it is a numeric equality reference for Figure 4, not optional editorial prose. The exact aggregate input list and archive presence checks are recorded in `REPRODUCTION_DEPENDENCIES.json`.

**Use the recommended builders above instead of the old direct command `python src/render_revision_v1_10.py`.** Its default `protocol()` hashes `paper/v1_9/design_details.tex`, which would be omitted. The r1 builder replaces that function and uses `figure1_data_roles.csv`, so the recommended route has no such manuscript dependency. Similarly, archived package-building commands expect earlier manuscript source/archive locations and are not public reproduction entry points.

## Supplementary displays and discovery summary

```sh
.venv/bin/python src/render_revision_v1_11.py
.venv/bin/python idea-update/scripts/15_discovery_report.py
```

The first command renders completed extension tables/numeric definitions and three supplementary figures to `paper/v1_11/generated/` and `paper/v1_11/figures/`. It calculates exact rational construction identities but performs no model estimation. It creates the required output directories. Keep all aggregate `idea-update/results/` inputs, `idea-update/protocol_audit_discovery_v1.json`, `idea-update/scripts/00_exact_construction.py`, `idea-update/scripts/14_discovery_eval.py` (its source hash is reported), the v1.9/v1.10 result inputs and `docs/revisions/kbs_v1_9/outcome_audit.csv`.

The second command rewrites `idea-update/results/14_report.md` from `13_discovery_roles.csv`, `14_methods.csv`, `14_contrasts.csv`, `14_logloss.csv`, `14_status.csv` and the frozen protocol; it imports `idea-update/scripts/mdtable.py`. It applies the recorded decision rules to aggregate records, without re-estimation. The discovery datasets had already been inspected; the later frozen protocol does not make them untouched confirmation data.

## Synthetic reruns are separate work

The source provides complete synthetic designs. `src/mechanism_v1_10.py freeze` / `run` refuse existing completed production output. A scientific rerun should be made in a separate clone with an intentionally new output record; do not delete or silently replace the released results. `idea-update/scripts/09_postgate_refit_sim.py` and `11_identity_positive_control.py` execute simulations/fits at module top level and write fixed result paths. Importing these modules therefore executes scientific work; they are not part of the bounded release checks. Listing these sources does not mean a new run has been performed.

## Restricted-data commands

`src/audit_revision_v1_9.py`, `src/candidate_rebuild_v1_9.py`, the preprocessing/training programs, empirical `idea-update/scripts/02_*`–`07_*`, `10_*`, `12_*`–`14_*` require some combination of private inputs. In particular, the seven-dataset evaluator's `--smoke` still loads individual validation/test rows, the `results/tables/v1_4_mechanism/local_cache/*.npz` prediction/label cache, member records and a saved deployment model. It is **not** a public-data-free smoke check.

Mortgage reconstruction also requires authorized provider files, row identifiers and split mappings, original candidate/reference `.joblib` models, and frozen development artifacts. These intentionally excluded caches and models prevent end-to-end empirical reproduction from the public aggregates alone.

No credential, raw loan data, model artifact, manuscript draft, human/agent review material or private correspondence is needed for the recommended public exact/aggregate/test commands.
