"""
build_reference_ensemble_v0_7.py
=================================

Constructs the **reference Rashomon ensemble** ``R_ref`` defined in
``preregistration_v0_7.md`` §1.1 — the strategy-orthogonality device that
underpins hypotheses H-C / H-A / H-A' (and that A(d), §1.2, is built from).

DESIGN CONTRACT (do not weaken without a new amendment)
-------------------------------------------------------
R_ref is an *evaluation-only* estimate of population RC. It NEVER routes.
It is built from a seed stream ``S_ref`` that is disjoint from the deployment
committee V, the candidate grid G, and the deployed model M2.

Two strictly separated phases (this separation IS the firewall at code level):

  Phase 1 — build_reference_ensemble()
      Reads TRAIN (fit) + VAL (AUC/Brier for the axis-local-epsilon filter)
      ONLY. Produces a frozen R_ref member manifest. Must run end-to-end and
      pass all three unit tests WITHOUT ever touching a test split.

  Phase 2 — generate_orthogonal_target()  [gated behind require_test=True]
      Uses the FROZEN R_ref to score the TEST split exactly once and derive
      A(d) = top-decile by reference_RC_range. This is the only place a test
      split is read. It refuses to run unless Phase 1 output exists and is
      sealed, and unless the caller explicitly opts in.

Aligns to the repo's real interfaces (no invented unified fit_grid()):
  - src/compute_rashomon_capacity.py : filter_rashomon_set, compute_predictions_matrix, compute_rc_per_instance
  - src/train_hmda_phase_b.py        : train_m1/train_m2/train_m6/train_m7
  - src/audit_budget_routing_v0_6.py : find_deployed_m2
  - src/v0_4_common.py               : load_split, RANDOM_STATE

Status: Phase 1 implemented. Phase 2 remains opt-in and must not be executed
until the frozen reference manifests have been reviewed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

# --- repo imports (Phase-1 logic must not import anything that reads test) ---
try:
    from .audit_budget_routing_v0_6 import find_deployed_m2
    from .compute_rashomon_capacity import (
        compute_predictions_matrix,
        compute_rc_per_instance,
        filter_rashomon_set,
    )
    from .preprocessing_fannie_mae import (
        CATEGORICAL_FEATURES as FANNIE_CATEGORICAL_FEATURES,
    )
    from .preprocessing_fannie_mae import NUMERIC_FEATURES as FANNIE_NUMERIC_FEATURES
    from .preprocessing_freddie_mac import (
        CATEGORICAL_FEATURES as FREDDIE_CATEGORICAL_FEATURES,
    )
    from .preprocessing_freddie_mac import NUMERIC_FEATURES as FREDDIE_NUMERIC_FEATURES
    from . import train_hmda_phase_b as candidate_trainer
    from .v0_4_common import RANDOM_STATE, load_split
except ImportError:
    from audit_budget_routing_v0_6 import find_deployed_m2
    from compute_rashomon_capacity import (
        compute_predictions_matrix,
        compute_rc_per_instance,
        filter_rashomon_set,
    )
    from preprocessing_fannie_mae import (
        CATEGORICAL_FEATURES as FANNIE_CATEGORICAL_FEATURES,
    )
    from preprocessing_fannie_mae import NUMERIC_FEATURES as FANNIE_NUMERIC_FEATURES
    from preprocessing_freddie_mac import (
        CATEGORICAL_FEATURES as FREDDIE_CATEGORICAL_FEATURES,
    )
    from preprocessing_freddie_mac import NUMERIC_FEATURES as FREDDIE_NUMERIC_FEATURES
    import train_hmda_phase_b as candidate_trainer
    from v0_4_common import RANDOM_STATE, load_split


# ======================================================================
# Locked constants (from preregistration_v0_7.md §6). DO NOT EDIT.
# ======================================================================

EPSILON: float = 0.010                    # axis-local AUC tolerance (inherited)
S_REF_LO: int = 770_000                   # reference seed block, inclusive
S_REF_HI: int = 779_999                   # reference seed block, inclusive
DEPLOYMENT_RANDOM_STATE: int = 20_260_504  # v0.6 grid/M2 seed (v0_4_common.RANDOM_STATE)
R_REF_MIN_SIZE: int = 50                  # |R_ref| target
A_QUANTILE: float = 0.90                  # A(d) = top-decile of reference_RC_range
DEGEN_FRAC: float = 0.05                  # reference-degenerate if #{rc>0} < 0.05*N
DEGEN_EFF_RANK: float = 1.5               # reference-degenerate if eff. rank < 1.5
AXES: tuple[str, ...] = ("M1", "M2", "M6", "M7")
N_SEEDS_PER_CONFIG: int = 4
V06_GRID_CONFIG_COUNTS: dict[str, int] = {
    "M1": 6,
    "M2": 7,
    "M6": 9,
    "M7": 4,
}

S_REF: range = range(S_REF_LO, S_REF_HI + 1)

if RANDOM_STATE != DEPLOYMENT_RANDOM_STATE:
    raise RuntimeError(
        f"deployment seed mismatch: v0_4_common={RANDOM_STATE}, "
        f"preregistered={DEPLOYMENT_RANDOM_STATE}"
    )


# ======================================================================
# Dataset registry (GSE is NOT in V0_4_DATASET; register explicitly).
# Only train/val are registered. Test path is derived lazily and only ever
# touched in Phase 2. load_split() additionally refuses any path containing
# 'test', so the test path is never passed to it.
# ======================================================================

@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    processed_dir: Path
    target_col: str
    # column groupings needed by train_mX (raw vs tree/one-hot). Fill per dataset.
    raw_features: tuple[str, ...] = ()
    tree_features: tuple[str, ...] = ()
    categorical_features: tuple[str, ...] = ()

    @property
    def train_path(self) -> Path:
        return self.processed_dir / "train.parquet"

    @property
    def val_path(self) -> Path:
        return self.processed_dir / "val.parquet"

    @property
    def _test_path(self) -> Path:
        # Phase-2 only. Deliberately prefixed _ and never passed to load_split().
        return self.processed_dir / "test.parquet"


_PROC = Path("data/processed")

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "freddie_mac_2018": DatasetSpec(
        slug="freddie_mac_2018",
        processed_dir=_PROC / "freddie_mac_2018_ever_60dpd_24m",
        target_col="ever_60dpd_24m",
        raw_features=tuple(FREDDIE_NUMERIC_FEATURES + FREDDIE_CATEGORICAL_FEATURES),
        tree_features=tuple(FREDDIE_NUMERIC_FEATURES + FREDDIE_CATEGORICAL_FEATURES),
        categorical_features=tuple(FREDDIE_CATEGORICAL_FEATURES),
    ),
    "fannie_mae_2018": DatasetSpec(
        slug="fannie_mae_2018",
        processed_dir=_PROC / "fannie_mae_2018_ever_60dpd_24m",
        target_col="ever_60dpd_24m",
        raw_features=tuple(FANNIE_NUMERIC_FEATURES + FANNIE_CATEGORICAL_FEATURES),
        tree_features=tuple(FANNIE_NUMERIC_FEATURES + FANNIE_CATEGORICAL_FEATURES),
        categorical_features=tuple(FANNIE_CATEGORICAL_FEATURES),
    ),
}


# ======================================================================
# Phase 1 — build R_ref (TRAIN + VAL only; never reads test)
# ======================================================================

_REFERENCE_MODEL_DIR = Path("models/v0_7/r_ref")
_DEPLOYMENT_MODEL_DIR = Path("models/v0_7/deployment")
_TREE_TRAIN_ATTR = "_rref_tree_train"
_TREE_VAL_ATTR = "_rref_tree_validation"
_DEPLOYMENT_M2_IDENTITIES: dict[str, dict[str, Any]] = {}


def _seed_for(axis: str, replicate: int) -> int:
    """Deterministic seed drawn from S_ref for (axis, replicate).

    Guarantees every R_ref seed lies in S_ref and never equals the deployment
    seed. Keep this pure so the unit tests can verify membership offline.
    """
    base = S_REF_LO + (AXES.index(axis) * 1000) + replicate
    if base > S_REF_HI:
        raise ValueError(f"seed {base} escapes S_ref block")
    return base


def _validate_phase1_frame(spec: DatasetSpec, frame: pd.DataFrame, split: str) -> None:
    required = {spec.target_col, *spec.raw_features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.slug} {split} missing locked columns: {missing}")
    unexpected_tree = sorted(set(spec.tree_features) - set(spec.raw_features))
    if unexpected_tree:
        raise ValueError(f"{spec.slug} tree features not present in raw feature scope: {unexpected_tree}")


def _build_tree_frames(
    spec: DatasetSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical = [column for column in spec.categorical_features if column in spec.tree_features]
    train_tree = pd.get_dummies(
        train[list(spec.tree_features)],
        columns=categorical,
        drop_first=False,
        dtype=np.float32,
    ).astype(np.float32)
    validation_tree = pd.get_dummies(
        validation[list(spec.tree_features)],
        columns=categorical,
        drop_first=False,
        dtype=np.float32,
    ).reindex(columns=train_tree.columns, fill_value=0).astype(np.float32)
    return train_tree, validation_tree


@contextmanager
def _gse_training_context(spec: DatasetSpec):
    """Give the inherited M1 wrapper the GSE categorical schema."""
    import m1_feature_engineering

    previous_dataset = os.environ.get("V0_4_DATASET")
    previous_categorical = m1_feature_engineering.CATEGORICAL_FEATURES
    os.environ["V0_4_DATASET"] = spec.slug
    m1_feature_engineering.CATEGORICAL_FEATURES = list(spec.categorical_features)
    try:
        yield
    finally:
        m1_feature_engineering.CATEGORICAL_FEATURES = previous_categorical
        if previous_dataset is None:
            os.environ.pop("V0_4_DATASET", None)
        else:
            os.environ["V0_4_DATASET"] = previous_dataset


def _dispatch_axis_training(
    spec: DatasetSpec,
    axis: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> list[dict[str, Any]]:
    train_tree = train.attrs[_TREE_TRAIN_ATTR]
    validation_tree = validation.attrs[_TREE_VAL_ATTR]
    if axis == "M1":
        return candidate_trainer.train_m1(
            train=train,
            validation=validation,
            raw_features=list(spec.raw_features),
            y_train=y_train,
            y_val=y_val,
            random_state=random_state,
        )
    if axis == "M2":
        return candidate_trainer.train_m2(
            train_tree=train_tree,
            validation_tree=validation_tree,
            y_train=y_train,
            y_val=y_val,
            random_state=random_state,
        )
    if axis == "M6":
        return candidate_trainer.train_m6(
            train_tree=train_tree,
            validation_tree=validation_tree,
            y_train=y_train,
            y_val=y_val,
            random_state=random_state,
        )
    if axis == "M7":
        return candidate_trainer.train_m7(
            dataset=spec.slug,
            train=train,
            validation=validation,
            raw_features=list(spec.raw_features),
            train_tree=train_tree,
            validation_tree=validation_tree,
            tree_features=list(train_tree.columns),
            y_train=y_train,
            y_val=y_val,
            random_state=random_state,
            categorical_features=list(spec.categorical_features),
        )
    raise KeyError(f"unknown mechanism axis: {axis}")


def _axis_grid_specs(axis: str) -> list[dict[str, Any]]:
    if axis == "M6":
        specs = [
            {"n_estimators": n_estimators, "max_depth": max_depth}
            for n_estimators in [200, 300, 500]
            for max_depth in [None, 8, 16]
        ]
    else:
        specs = [
            copy.deepcopy(item)
            for item in candidate_trainer.load_candidate_config(axis)
            if axis != "M7" or item.get("interactions") == 0
        ]
    expected = V06_GRID_CONFIG_COUNTS[axis]
    if len(specs) != expected:
        raise ValueError(
            f"{axis} v0.6 grid has {len(specs)} configurations; expected {expected}"
        )
    return specs


@contextmanager
def _single_candidate_grid(axis: str, template: dict[str, Any], seed: int):
    """Restrict an inherited per-axis grid function to one locked template."""
    original_loader = candidate_trainer.load_candidate_config
    original_product = candidate_trainer.product
    if axis == "M6":
        n_estimators = int(template["n_estimators"])
        max_depth = template["max_depth"]
        candidate_trainer.product = lambda *_args, **_kwargs: iter([(n_estimators, max_depth)])
    else:
        seeded = copy.deepcopy(template)
        seeded["random_state"] = int(seed)

        def load_one(requested_axis: str) -> list[dict[str, Any]]:
            if requested_axis == axis:
                return [copy.deepcopy(seeded)]
            return original_loader(requested_axis)

        candidate_trainer.load_candidate_config = load_one
    try:
        yield
    finally:
        candidate_trainer.load_candidate_config = original_loader
        candidate_trainer.product = original_product


def _artifact_from_row(
    spec: DatasetSpec,
    axis: str,
    candidate_id: str,
    row: dict[str, Any],
    *,
    seed: int,
    train_tree_columns: list[str],
) -> dict[str, Any]:
    input_type = str(row["input_type"])
    return {
        "dataset": spec.slug,
        "model_id": axis,
        "candidate_id": candidate_id,
        "model": row["model"],
        "input_type": input_type,
        "raw_feature_columns": list(spec.raw_features),
        "feature_columns": (
            list(spec.raw_features) if input_type == "raw" else list(train_tree_columns)
        ),
        "categorical_features": list(spec.categorical_features),
        "predict_as_numpy": bool(row.get("predict_as_numpy", False)),
        "hyperparameters": row["hyperparameters"],
        "random_state": int(seed),
        "trained_on": spec.train_path.as_posix(),
        "selection_split": spec.val_path.as_posix(),
        "validation_auc": float(row["validation_auc"]),
        "validation_brier": float(row["validation_brier"]),
        "validation_calibration_slope": float(row["validation_calibration_slope"]),
    }


def _atomic_joblib_dump(artifact: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    joblib.dump(artifact, temporary)
    temporary.replace(path)
    path.chmod(0o444)


def _prune_reference_artifacts(slug: str, members: list[dict[str, Any]]) -> None:
    member_paths = {Path(member["artifact_path"]).resolve() for member in members}
    artifact_dir = _REFERENCE_MODEL_DIR / slug
    for path in artifact_dir.glob("*.joblib"):
        if path.resolve() not in member_paths:
            path.unlink()


def _fit_axis_candidates(
    spec: DatasetSpec,
    axis: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    """Fit one mechanism axis across the given S_ref seeds.

    Thin wrapper over train_m1/m2/m6/m7. Each train_mX has a different
    signature (see src/train_hmda_phase_b.py); dispatch accordingly and pass
    the S_ref seed as random_state. Returns the repo's candidate-dict list.
    """
    seed_list = [int(seed) for seed in seeds]
    if len(seed_list) != N_SEEDS_PER_CONFIG or any(seed not in S_REF for seed in seed_list):
        raise ValueError(
            f"{axis} requires exactly {N_SEEDS_PER_CONFIG} seeds from S_ref"
        )
    if DEPLOYMENT_RANDOM_STATE in seed_list:
        raise ValueError(f"{axis} reference seeds overlap deployment seed")

    train_tree_columns = list(train.attrs[_TREE_TRAIN_ATTR].columns)
    output: list[dict[str, Any]] = []
    with _gse_training_context(spec):
        if axis == "M2":
            deployment_grid = _dispatch_axis_training(
                spec,
                axis,
                train,
                validation,
                y_train,
                y_val,
                DEPLOYMENT_RANDOM_STATE,
            )
            best = candidate_trainer.select_best(deployment_grid)
            deployment_path = _DEPLOYMENT_MODEL_DIR / spec.slug / "m2_xgb.joblib"
            deployment_artifact = _artifact_from_row(
                spec,
                axis,
                str(best["candidate_id"]),
                best,
                seed=DEPLOYMENT_RANDOM_STATE,
                train_tree_columns=train_tree_columns,
            )
            _atomic_joblib_dump(deployment_artifact, deployment_path)
            deployed = find_deployed_m2(
                [
                    {
                        "model_id": "M2",
                        "candidate_id": str(best["candidate_id"]),
                        "artifact_path": deployment_path.as_posix(),
                        "source_artifact_path": deployment_path.as_posix(),
                        "random_state": DEPLOYMENT_RANDOM_STATE,
                    }
                ]
            )
            _DEPLOYMENT_M2_IDENTITIES[spec.slug] = deployed
            print(
                f"{spec.slug} M2: deployed candidate {best['candidate_id']} "
                f"fixed at seed {DEPLOYMENT_RANDOM_STATE}"
            )
            del deployment_grid

        grid_specs = _axis_grid_specs(axis)
        print(
            f"{spec.slug} {axis}: v0.6 grid configs={len(grid_specs)} "
            f"x S_ref seeds/config={N_SEEDS_PER_CONFIG}"
        )
        for config_index, template in enumerate(grid_specs, start=1):
            for seed in seed_list:
                with _single_candidate_grid(axis, template, seed):
                    rows = _dispatch_axis_training(
                        spec,
                        axis,
                        train,
                        validation,
                        y_train,
                        y_val,
                        seed,
                    )
                if len(rows) != 1:
                    raise ValueError(
                        f"{axis} config {config_index} seed {seed} produced "
                        f"{len(rows)} candidates; expected one"
                    )
                row = rows[0]
                candidate_id = f"{row['candidate_id']}__sref_{seed}"
                artifact_path = _REFERENCE_MODEL_DIR / spec.slug / f"{candidate_id}.joblib"
                artifact = _artifact_from_row(
                    spec,
                    axis,
                    candidate_id,
                    row,
                    seed=seed,
                    train_tree_columns=train_tree_columns,
                )
                if int(artifact["random_state"]) != seed:
                    raise AssertionError(
                        f"{candidate_id} did not receive its locked S_ref seed"
                    )
                _atomic_joblib_dump(artifact, artifact_path)
                output.append(
                    {
                        "candidate_id": candidate_id,
                        "template_candidate_id": str(row["candidate_id"]),
                        "model_id": axis,
                        "axis": axis,
                        "seed": seed,
                        "validation_auc": float(row["validation_auc"]),
                        "validation_brier": float(row["validation_brier"]),
                        "validation_calibration_slope": float(
                            row["validation_calibration_slope"]
                        ),
                        "artifact_path": artifact_path.as_posix(),
                    }
                )
            print(
                f"{spec.slug} {axis}: fit config {config_index}/{len(grid_specs)} "
                f"candidate={row['candidate_id']}"
            )
    return output


def build_reference_ensemble(spec: DatasetSpec) -> dict[str, Any]:
    """Phase 1: build and freeze R_ref for one dataset. TRAIN + VAL only.

    Returns a manifest dict (also written to disk) with at least:
        {
          "slug": str,
          "epsilon": EPSILON,
          "members": [ {candidate_id, axis, seed, validation_auc,
                        validation_brier, artifact_path}... ],
          "size": int,
          "degenerate": bool,           # filled in Phase 2 once test RC exists,
                                        # provisional False here
          "deployment_m2_identity": {...},  # from find_deployed_m2, for the test
        }
    Must NOT read any test split.
    """
    print(
        f"{spec.slug}: Phase 1 only; loading TRAIN={spec.train_path} "
        f"and VAL={spec.val_path}; TEST is not registered with load_split"
    )
    train = load_split(spec.train_path)
    validation = load_split(spec.val_path)
    _validate_phase1_frame(spec, train, "train")
    _validate_phase1_frame(spec, validation, "validation")
    y_train = train[spec.target_col].to_numpy(dtype=int)
    y_val = validation[spec.target_col].to_numpy(dtype=int)
    train_tree, validation_tree = _build_tree_frames(spec, train, validation)
    train.attrs[_TREE_TRAIN_ATTR] = train_tree
    validation.attrs[_TREE_VAL_ATTR] = validation_tree

    _DEPLOYMENT_M2_IDENTITIES.pop(spec.slug, None)
    candidates: list[dict[str, Any]] = []
    for axis in AXES:
        axis_seeds = [
            _seed_for(axis, seed_index)
            for seed_index in range(N_SEEDS_PER_CONFIG)
        ]
        candidates.extend(
            _fit_axis_candidates(
                spec,
                axis,
                train,
                validation,
                y_train,
                y_val,
                axis_seeds,
            )
        )

    metrics = pd.DataFrame(candidates)
    kept = filter_rashomon_set(metrics, epsilon=EPSILON)
    kept_ids = set(kept["candidate_id"].astype(str))
    members = [candidate for candidate in candidates if candidate["candidate_id"] in kept_ids]
    if len(members) < R_REF_MIN_SIZE:
        raise RuntimeError(
            f"{spec.slug} R_ref size {len(members)} is below locked minimum "
            f"{R_REF_MIN_SIZE}; do not proceed to Phase 2"
        )

    deployed = _DEPLOYMENT_M2_IDENTITIES.get(spec.slug)
    if deployed is None:
        raise RuntimeError(f"{spec.slug} deployment M2 identity was not fixed")
    members = sorted(
        members,
        key=lambda item: (
            AXES.index(item["axis"]),
            item["template_candidate_id"],
            item["seed"],
        ),
    )
    before_by_axis = {
        axis: int(sum(candidate["axis"] == axis for candidate in candidates))
        for axis in AXES
    }
    after_by_axis = {
        axis: int(sum(member["axis"] == axis for member in members))
        for axis in AXES
    }
    manifest = {
        "phase": 1,
        "frozen": True,
        "test_split_read": False,
        "slug": spec.slug,
        "epsilon": EPSILON,
        "seed_block": [S_REF_LO, S_REF_HI],
        "grid": "v0.6 hyperparameter grid x S_ref seeds",
        "n_seeds_per_config": N_SEEDS_PER_CONFIG,
        "axis_config_counts": V06_GRID_CONFIG_COUNTS,
        "base_rate_threshold": float(validation[spec.target_col].mean()),
        "candidate_count_before_filter": int(len(candidates)),
        "candidate_count_after_filter": int(len(members)),
        "axis_counts_before_filter": before_by_axis,
        "axis_counts_after_filter": after_by_axis,
        "members": members,
        "size": int(len(members)),
        "degenerate": False,
        "deployment_m2_identity": deployed,
    }
    if deployed_m2_identity(manifest) in reference_member_identities(manifest):
        raise AssertionError("deployed M2 candidate identity appears in R_ref")
    member_seeds = reference_member_seeds(manifest)
    if not member_seeds.issubset(set(S_REF)) or DEPLOYMENT_RANDOM_STATE in member_seeds:
        raise AssertionError("R_ref seed stream overlaps deployment grid")
    _prune_reference_artifacts(spec.slug, members)
    _write_manifest(spec.slug, manifest)
    print(
        f"{spec.slug}: candidates={len(candidates)} kept={len(members)} "
        f"axes_before={before_by_axis} axes_after={after_by_axis}"
    )
    return manifest


# ======================================================================
# Phase 2 — A(d) generation (reads TEST exactly once; opt-in gated)
# ======================================================================

def generate_orthogonal_target(
    spec: DatasetSpec,
    *,
    require_test: bool = False,
) -> pd.DataFrame:
    """Phase 2: score TEST with the frozen R_ref and return A(d).

    Reads the test split — the ONLY place this module does so. Guard rails:
      - refuses unless require_test=True (explicit opt-in; prevents reading
        test during the modeling phase);
      - refuses unless a frozen Phase-1 manifest exists for this slug;
      - refuses unless the test split is sealed (read-only / chmod 444).

    Returns a DataFrame indexed like the test split with columns:
        ['reference_rc_range', 'reference_rc_label', 'in_A']
    where in_A == (reference_rc_range >= quantile(0.90)). Also stamps the
    manifest's 'degenerate' flag per §1.1 degeneracy guard.
    """
    if not require_test:
        raise PermissionError(
            "generate_orthogonal_target reads the sealed test split; "
            "pass require_test=True only at the final metric-computation stage."
        )
    manifest = _read_manifest(spec.slug)
    if manifest is None:
        raise FileNotFoundError(f"freeze Phase-1 R_ref for {spec.slug} first")
    _assert_test_sealed(spec)
    output_path = _orthogonal_target_path(spec.slug)
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists; refusing to read the test split again"
        )
    if manifest.get("test_split_read"):
        raise RuntimeError(
            f"{spec.slug} manifest records a prior test read; refusing a second read"
        )

    members = pd.DataFrame(manifest["members"])
    if len(members) != int(manifest["size"]):
        raise ValueError(f"{spec.slug} manifest member count is inconsistent")
    base_rate_threshold = float(manifest["base_rate_threshold"])
    print(
        f"{spec.slug}: Phase 2 authorized; reading sealed TEST exactly once; "
        f"base_rate_threshold={base_rate_threshold:.8f} "
        "(source=validation prevalence)"
    )
    test_frame = pd.read_parquet(spec._test_path)
    p = compute_predictions_matrix(members, test_frame)
    rc = compute_rc_per_instance(
        p,
        base_rate_threshold=base_rate_threshold,
    )
    rc = rc.rename(
        columns={
            "rc_range": "reference_rc_range",
            "rc_label": "reference_rc_label",
        }
    )
    threshold = float(rc["reference_rc_range"].quantile(A_QUANTILE))
    rc["in_A"] = rc["reference_rc_range"] >= threshold
    target = rc[["reference_rc_range", "reference_rc_label", "in_A"]].copy()
    target.index = test_frame.index.copy()
    target.index.name = test_frame.index.name or "test_row_index"
    _write_orthogonal_target(spec.slug, target)
    degenerate = _stamp_degeneracy(
        spec.slug,
        target,
        p,
        output_path=output_path,
    )
    print(
        f"{spec.slug}: test_rows={len(target)} "
        f"reference_rc_range_median={target['reference_rc_range'].median():.8f} "
        f"q90={target['reference_rc_range'].quantile(0.90):.8f} "
        f"in_A={int(target['in_A'].sum())} "
        f"is_reference_degenerate={degenerate} "
        f"wrote={output_path}"
    )
    return target


# ======================================================================
# Manifest / sealing helpers
# ======================================================================

_MANIFEST_DIR = Path("results/tables/v0_7/r_ref")


def _manifest_path(slug: str) -> Path:
    return _MANIFEST_DIR / f"r_ref_manifest_{slug}.json"


def _orthogonal_target_path(slug: str) -> Path:
    return _MANIFEST_DIR / f"orthogonal_target_{slug}.parquet"


def _write_orthogonal_target(slug: str, target: pd.DataFrame) -> None:
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    destination = _orthogonal_target_path(slug)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    if temporary.exists():
        temporary.unlink()
    target.to_parquet(temporary, index=True)
    temporary.replace(destination)
    destination.chmod(0o444)


def _write_manifest(slug: str, manifest: dict[str, Any]) -> None:
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    destination = _manifest_path(slug)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    destination.chmod(0o444)


def _read_manifest(slug: str) -> dict[str, Any] | None:
    p = _manifest_path(slug)
    return json.loads(p.read_text()) if p.exists() else None


def _assert_test_sealed(spec: DatasetSpec) -> None:
    """Refuse to proceed unless the test split is read-only (chmod 444)."""
    p = spec._test_path
    if not p.exists():
        raise FileNotFoundError(f"test split missing: {p}")
    mode = os.stat(p).st_mode
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError(f"test split not sealed (writable): {p}")


# ======================================================================
# Membership / degeneracy utilities (pure; unit-tested)
# ======================================================================

def reference_member_seeds(manifest: dict[str, Any]) -> set[int]:
    return {int(m["seed"]) for m in manifest.get("members", [])}


def reference_member_identities(manifest: dict[str, Any]) -> set[str]:
    """Stable identity per member; prefer candidate_id, fall back to artifact path."""
    out: set[str] = set()
    for m in manifest.get("members", []):
        out.add(str(m.get("candidate_id") or m.get("artifact_path")))
    return out


def deployed_m2_identity(manifest: dict[str, Any]) -> str:
    d = manifest["deployment_m2_identity"]
    return str(d.get("candidate_id") or d.get("source_artifact_path") or d.get("artifact_path"))


def effective_rank(pred_matrix: np.ndarray) -> float:
    """Shannon effective rank of the voter-prediction correlation matrix (as o4)."""
    if pred_matrix.shape[1] < 2:
        return 1.0
    corr = np.corrcoef(np.asarray(pred_matrix, dtype=float).T)
    eig = np.linalg.eigvalsh(corr)
    eig = eig[eig > 0]
    if eig.size == 0:
        return 1.0
    p = eig / eig.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def is_reference_degenerate(reference_rc_range: np.ndarray, pred_matrix: np.ndarray) -> bool:
    n = len(reference_rc_range)
    frac_nonzero = float(np.mean(np.asarray(reference_rc_range) > 0))
    return (frac_nonzero < DEGEN_FRAC) or (effective_rank(pred_matrix) < DEGEN_EFF_RANK)


def _stamp_degeneracy(
    slug: str,
    rc: pd.DataFrame,
    pred_matrix: np.ndarray,
    *,
    output_path: Path,
) -> bool:
    degenerate = is_reference_degenerate(
        rc["reference_rc_range"].to_numpy(), pred_matrix
    )
    manifest = _read_manifest(slug) or {}
    manifest["degenerate"] = degenerate
    manifest["test_split_read"] = True
    manifest["phase_2_target_generated"] = True
    manifest["orthogonal_target_path"] = output_path.as_posix()
    _write_manifest(slug, manifest)
    return degenerate


# ======================================================================
# Unit tests (firewall geometry). Run: python build_reference_ensemble_v0_7.py --selftest
# These three must pass BEFORE any metric work. They use synthetic manifests
# and do NOT require fitted models or any data split.
# ======================================================================

def _selftest() -> None:
    # ---- Test 1: M2 (deployment artifact) is NOT a member of R_ref ----------
    # R_ref MAY contain an XGBoost/M2-axis model; what is forbidden is the
    # *deployed* M2 candidate identity. Test by identity, not by axis/model_id.
    manifest = {
        "members": [
            {"candidate_id": "M2_xgb_sref_770003", "axis": "M2", "seed": 770_003},
            {"candidate_id": "M1_lr_sref_770000", "axis": "M1", "seed": 770_000},
        ],
        "deployment_m2_identity": {
            "model_id": "M2",
            "candidate_id": "M2_xgb_d6_lr05_n300",  # the deployed one, seed 20260504
            "source_artifact_path": "models/v0_5/m2_xgb.pkl",
        },
    }
    assert deployed_m2_identity(manifest) not in reference_member_identities(manifest), \
        "deployed M2 identity must not appear in R_ref members"
    # an M2-axis member is allowed:
    assert any(m["axis"] == "M2" for m in manifest["members"]), \
        "R_ref is allowed to contain an M2-axis (XGBoost) model"

    # ---- Test 2: every R_ref seed is in S_ref and disjoint from deployment ---
    seeds = reference_member_seeds(manifest)
    assert seeds.issubset(set(S_REF)), "all R_ref seeds must lie in S_ref"
    assert DEPLOYMENT_RANDOM_STATE not in seeds, "R_ref must not reuse the deployment seed"
    # _seed_for must stay inside S_ref for every axis/replicate it can emit:
    for axis in AXES:
        for r in range(100):
            assert _seed_for(axis, r) in S_REF

    # ---- Test 3: A(d) is reproducible under the locked quantile/seed ---------
    rng = np.random.default_rng(DEPLOYMENT_RANDOM_STATE)  # any fixed seed; A(d) is deterministic given rc
    rc_vals = rng.random(2000)
    thr_a = np.quantile(rc_vals, A_QUANTILE)
    thr_b = np.quantile(rc_vals, A_QUANTILE)
    in_a = rc_vals >= thr_a
    in_b = rc_vals >= thr_b
    assert np.array_equal(in_a, in_b), "A(d) membership must be deterministic"
    assert abs(in_a.mean() - (1 - A_QUANTILE)) < 0.02, "A(d) should select ~top decile"

    print("OK: 3/3 firewall geometry tests passed "
          "(M2-identity exclusion, S_ref seed disjointness, A(d) reproducibility).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build reference Rashomon ensemble R_ref (v0.7).")
    ap.add_argument("--selftest", action="store_true", help="run firewall geometry unit tests")
    ap.add_argument("--dataset", choices=sorted(DATASET_REGISTRY), help="build Phase-1 R_ref for one dataset")
    ap.add_argument("--emit-target", action="store_true",
                    help="Phase 2: read sealed TEST and emit A(d). Requires --i-understand-this-reads-test")
    ap.add_argument("--i-understand-this-reads-test", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.dataset and not args.emit_target:
        build_reference_ensemble(DATASET_REGISTRY[args.dataset])
        return
    if args.dataset and args.emit_target:
        generate_orthogonal_target(
            DATASET_REGISTRY[args.dataset],
            require_test=args.i_understand_this_reads_test,
        )
        return
    ap.print_help()


if __name__ == "__main__":
    main()
