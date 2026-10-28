"""Op4/Op5 candidate-pool training for v0.5 credit datasets.

This script deliberately avoids UCI/FICO/synthetic artifacts and never loads a
test split. It reuses the v0.4 candidate-grid definitions for M1/M2/M6/M7,
then writes one canonical artifact per axis for later RC phases.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = Path("data/processed/hmda/train.parquet")
DEFAULT_VAL = Path("data/processed/hmda/val.parquet")
DEFAULT_MODEL_DIR = Path("models/hmda")
DEFAULT_METRICS_OUT = Path("results/tables/v0_5/wave_metrics_hmda.csv")
DATASET_DEFAULTS = {
    "hmda": {
        "train_split": DEFAULT_TRAIN,
        "val_split": DEFAULT_VAL,
        "model_dir": DEFAULT_MODEL_DIR,
        "metrics_out": DEFAULT_METRICS_OUT,
    },
    "lendingclub": {
        "train_split": Path("data/processed/lendingclub/train.parquet"),
        "val_split": Path("data/processed/lendingclub/val.parquet"),
        "model_dir": Path("models/lendingclub"),
        "metrics_out": Path("results/tables/v0_5/wave_metrics_lendingclub.csv"),
    },
    "gmc": {
        "train_split": Path("data/processed/gmc_train.csv"),
        "val_split": Path("data/processed/gmc_validation.csv"),
        "model_dir": Path("models/v0_4_gmc"),
        "metrics_out": Path("results/tables/v0_4/candidate_pool_metrics_gmc.csv"),
    },
    "home_credit": {
        "train_split": Path("data/processed/home_credit_train.csv"),
        "val_split": Path("data/processed/home_credit_validation.csv"),
        "model_dir": Path("models/v0_4_home_credit"),
        "metrics_out": Path("results/tables/v0_4/candidate_pool_metrics_home_credit.csv"),
    },
    "home_credit_no_ext": {
        "train_split": Path("data/processed/home_credit_no_ext_train.csv"),
        "val_split": Path("data/processed/home_credit_no_ext_validation.csv"),
        "model_dir": Path("models/v0_4_home_credit_no_ext"),
        "metrics_out": Path("results/tables/v0_4/candidate_pool_metrics_home_credit_no_ext.csv"),
    },
    "hmeq": {
        "train_split": Path("data/processed/hmeq_train.csv"),
        "val_split": Path("data/processed/hmeq_validation.csv"),
        "model_dir": Path("models/v0_4_hmeq"),
        "metrics_out": Path("results/tables/v0_4/candidate_pool_metrics_hmeq.csv"),
    },
    # n_g accrual step registered in preregistration_v0_7.md §4/§5.
    "german_credit": {
        "train_split": Path("data/processed/german_credit_train.csv"),
        "val_split": Path("data/processed/german_credit_validation.csv"),
        "model_dir": Path("models/v0_7_german_credit"),
        "metrics_out": Path("results/tables/v0_7/rho_g_accrual/candidate_pool_metrics_german_credit.csv"),
    },
    "australian_credit": {
        "train_split": Path("data/processed/australian_credit_train.csv"),
        "val_split": Path("data/processed/australian_credit_validation.csv"),
        "model_dir": Path("models/v0_7_australian_credit"),
        "metrics_out": Path("results/tables/v0_7/rho_g_accrual/candidate_pool_metrics_australian_credit.csv"),
    },
    "synthetic_v2": {
        "train_split": Path("data/processed/synthetic_v2_train.csv"),
        "val_split": Path("data/processed/synthetic_v2_validation.csv"),
        "model_dir": Path("models/v0_4_synthetic_v2"),
        "metrics_out": Path("results/tables/v0_4/candidate_pool_metrics_synthetic_v2.csv"),
    },
    "synthetic_strong": {
        "train_split": Path("data/processed/synthetic_strong_train.csv"),
        "val_split": Path("data/processed/synthetic_strong_validation.csv"),
        "model_dir": Path("models/v0_4_synthetic_strong"),
        "metrics_out": Path("results/tables/v0_4/candidate_pool_metrics_synthetic_strong.csv"),
    },
}

MODEL_FILES = {
    "M1": "m1_lr_engineered.joblib",
    "M2": "m2_xgb.joblib",
    "M6": "m6_rf.joblib",
    "M7": "m7_ebm.joblib",
}

CONFIG_FILES = {
    "M1": Path("configs/m1_pool_wave2_5.json"),
    "M2": Path("configs/m2_pool_wave2_5.json"),
    "M7": Path("configs/m7_pool_wave2_5.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hmda", choices=sorted(DATASET_DEFAULTS))
    parser.add_argument("--axis", choices=["M1", "M2", "M6", "M7"])
    parser.add_argument("--train-split", type=Path)
    parser.add_argument("--val-split", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-metrics", action="store_true")
    args = parser.parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    args.train_split = args.train_split or defaults["train_split"]
    args.val_split = args.val_split or defaults["val_split"]
    args.model_dir = args.model_dir or defaults["model_dir"]
    args.metrics_out = args.metrics_out or defaults["metrics_out"]
    return args


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_dataset(dataset: str) -> dict[str, Any]:
    os.environ["V0_4_DATASET"] = dataset
    from data_preprocessing import CATEGORICAL_FEATURES, ID_COL, TARGET, one_hot_tree_frame
    from v0_4_common import AXES, RANDOM_STATE

    return {
        "CATEGORICAL_FEATURES": CATEGORICAL_FEATURES,
        "ID_COL": ID_COL,
        "TARGET": TARGET,
        "one_hot_tree_frame": one_hot_tree_frame,
        "AXES": AXES,
        "RANDOM_STATE": RANDOM_STATE,
    }


def read_frame(path: Path) -> pd.DataFrame:
    resolved = project_path(path)
    if path.name.lower().startswith("test") or "_test" in path.name.lower():
        raise ValueError(f"Phase B must not load the test split: {path}")
    if resolved.suffix == ".csv":
        return pd.read_csv(resolved)
    if resolved.suffix == ".parquet":
        return pd.read_parquet(resolved)
    raise ValueError(f"Unsupported split format: {path}")


def load_candidate_config(axis: str) -> list[dict[str, Any]]:
    data = json.loads(project_path(CONFIG_FILES[axis]).read_text(encoding="utf-8"))
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"No candidates in {CONFIG_FILES[axis]}")
    return candidates


def model_feature_columns(frame: pd.DataFrame, *, id_col: str, target: str) -> list[str]:
    return [col for col in frame.columns if col not in {id_col, target}]


def build_xgb(params: dict[str, Any], *, random_state: int) -> XGBClassifier:
    clean = {key: value for key, value in params.items() if key != "name"}
    clean.setdefault("random_state", random_state)
    clean.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
            "tree_method": "hist",
        }
    )
    return XGBClassifier(**clean)


def calibration_slope(y_true: np.ndarray, probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, y_true)
    return float(model.coef_[0, 0])


def evaluate_probs(y_val: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "validation_auc": float(roc_auc_score(y_val, probs)),
        "validation_brier": float(brier_score_loss(y_val, probs)),
        "validation_calibration_slope": calibration_slope(y_val, probs),
    }


def artifact_row(
    *,
    model_id: str,
    candidate_id: str,
    hyperparameters: dict[str, Any],
    metrics: dict[str, float],
    artifact_path: Path,
    axis_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    axis = axis_meta[model_id]
    return {
        "candidate_id": candidate_id,
        "model_id": model_id,
        "axis": axis["axis"],
        "axis_name": axis["axis_name"],
        "model_label": axis["model_label"],
        "hyperparameters_json": json.dumps(hyperparameters, sort_keys=True),
        "validation_auc": metrics["validation_auc"],
        "validation_brier": metrics["validation_brier"],
        "validation_calibration_slope": metrics["validation_calibration_slope"],
        "artifact_path": artifact_path.as_posix(),
    }


def train_m1(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    raw_features: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> list[dict[str, Any]]:
    from m1_feature_engineering import M1EngineeredLogistic

    rows: list[dict[str, Any]] = []
    for spec in load_candidate_config("M1"):
        candidate_id = str(spec["name"])
        feature_engineering = bool(spec.get("feature_engineering", False))
        logistic_params = {
            key: value
            for key, value in spec.items()
            if key not in {"name", "feature_engineering"}
        }
        logistic_params.setdefault("random_state", random_state)
        model = M1EngineeredLogistic(
            feature_engineering=feature_engineering,
            logistic_params=logistic_params,
        )
        model.fit(train[raw_features], y_train)
        probs = model.predict_proba(validation[raw_features])[:, 1]
        rows.append(
            {
                "candidate_id": candidate_id,
                "model": model,
                "input_type": "raw",
                "raw_feature_columns": raw_features,
                "feature_columns": raw_features,
                "hyperparameters": {
                    "model": "LogisticRegression",
                    "feature_engineering": feature_engineering,
                    **logistic_params,
                },
                **evaluate_probs(y_val, probs),
            }
        )
    return rows


def train_m2(
    *,
    train_tree: pd.DataFrame,
    validation_tree: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in load_candidate_config("M2"):
        candidate_id = str(spec["name"])
        model = build_xgb(spec, random_state=random_state)
        model.fit(train_tree.to_numpy(), y_train)
        probs = model.predict_proba(validation_tree.to_numpy())[:, 1]
        rows.append(
            {
                "candidate_id": candidate_id,
                "model": model,
                "input_type": "one_hot",
                "predict_as_numpy": True,
                "hyperparameters": {
                    "model": "XGBClassifier",
                    **{key: value for key, value in spec.items() if key != "name"},
                },
                **evaluate_probs(y_val, probs),
            }
        )
    return rows


def train_m6(
    *,
    train_tree: pd.DataFrame,
    validation_tree: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_estimators, max_depth in product([200, 300, 500], [None, 8, 16]):
        candidate_id = f"M6_rf_n{n_estimators}_depth{'none' if max_depth is None else max_depth}"
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=1,
            class_weight=None,
        )
        model.fit(train_tree, y_train)
        probs = model.predict_proba(validation_tree)[:, 1]
        rows.append(
            {
                "candidate_id": candidate_id,
                "model": model,
                "input_type": "one_hot",
                "hyperparameters": {
                    "model": "RandomForestClassifier",
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                },
                **evaluate_probs(y_val, probs),
            }
        )
    return rows


def train_m7(
    *,
    dataset: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    raw_features: list[str],
    train_tree: pd.DataFrame,
    validation_tree: pd.DataFrame,
    tree_features: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
    categorical_features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    use_native_categorical = dataset in {"freddie_mac_2018", "fannie_mae_2018"}
    use_one_hot = bool(categorical_features) and not use_native_categorical
    fit_train = train_tree if use_one_hot else train[raw_features]
    fit_validation = validation_tree if use_one_hot else validation[raw_features]
    fit_features = tree_features if use_one_hot else raw_features
    if use_native_categorical:
        categorical_set = set(categorical_features)
        feature_types = [
            "nominal" if feature in categorical_set else "continuous"
            for feature in fit_features
        ]
    else:
        feature_types = ["continuous"] * len(fit_features)
    for spec in load_candidate_config("M7"):
        if spec.get("interactions") != 0:
            continue
        candidate_id = str(spec["name"])
        params = {
            key: value
            for key, value in spec.items()
            if key != "name"
        }
        params.update(
            {
                "feature_names": fit_features,
                "feature_types": feature_types,
                "random_state": random_state,
                "n_jobs": -2,
            }
        )
        model = ExplainableBoostingClassifier(**params)
        model.fit(fit_train, y_train)
        probs = model.predict_proba(fit_validation)[:, 1]
        rows.append(
            {
                "candidate_id": candidate_id,
                "model": model,
                "input_type": "one_hot" if use_one_hot else "raw",
                "raw_feature_columns": raw_features,
                "feature_columns": fit_features,
                "hyperparameters": {
                    "model": "ExplainableBoostingClassifier",
                    **{key: value for key, value in spec.items() if key != "name"},
                },
                **evaluate_probs(y_val, probs),
            }
        )
    if not rows:
        raise ValueError("M7 grid contains no interactions=0 candidates.")
    return rows


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(rows, key=lambda row: (row["validation_auc"], -row["validation_brier"]), reverse=True)[0]


def write_axis_outputs(
    *,
    dataset: str,
    axis: str,
    model_rows: list[dict[str, Any]],
    train_split: Path,
    val_split: Path,
    model_dir: Path,
    raw_features: list[str],
    tree_features: list[str],
    validation_base_rate_threshold: float,
    wall_clock_seconds: float,
    axis_meta: dict[str, dict[str, Any]],
    random_state: int,
) -> None:
    resolved_model_dir = project_path(model_dir)
    metrics_dir = resolved_model_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = resolved_model_dir / MODEL_FILES[axis]
    artifact_rel = artifact_path.relative_to(PROJECT_ROOT)

    best = select_best(model_rows)
    input_type = best["input_type"]
    artifact = {
        "dataset": dataset,
        "model_id": axis,
        "candidate_id": best["candidate_id"],
        "model": best["model"],
        "input_type": input_type,
        "raw_feature_columns": raw_features,
        "feature_columns": raw_features if input_type == "raw" else tree_features,
        "predict_as_numpy": bool(best.get("predict_as_numpy", False)),
        "hyperparameters": best["hyperparameters"],
        "random_state": random_state,
        "trained_on": train_split.as_posix(),
        "selection_split": val_split.as_posix(),
        "validation_base_rate_threshold": validation_base_rate_threshold,
        "validation_auc": best["validation_auc"],
        "validation_brier": best["validation_brier"],
        "validation_calibration_slope": best["validation_calibration_slope"],
        "training_wall_clock_seconds": wall_clock_seconds,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    grid_rows = []
    for row in model_rows:
        grid_rows.append(
            artifact_row(
                model_id=axis,
                candidate_id=row["candidate_id"],
                hyperparameters=row["hyperparameters"],
                metrics=row,
                artifact_path=artifact_rel,
                axis_meta=axis_meta,
            )
        )
    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["axis", "validation_auc", "validation_brier"],
        ascending=[True, False, True],
    )
    grid_df.to_csv(metrics_dir / f"{axis.lower()}_grid_metrics.csv", index=False)

    best_row = artifact_row(
        model_id=axis,
        candidate_id=best["candidate_id"],
        hyperparameters=best["hyperparameters"],
        metrics=best,
        artifact_path=artifact_rel,
        axis_meta=axis_meta,
    )
    best_row["validation_base_rate_threshold"] = validation_base_rate_threshold
    best_row["training_wall_clock_seconds"] = wall_clock_seconds
    (metrics_dir / f"{axis.lower()}_metrics.json").write_text(
        json.dumps(best_row, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"{axis}: selected {best['candidate_id']} "
        f"AUC={best['validation_auc']:.6f} "
        f"Brier={best['validation_brier']:.6f} "
        f"slope={best['validation_calibration_slope']:.6f} "
        f"time={wall_clock_seconds:.2f}s"
    )


def write_metrics_csv(model_dir: Path, metrics_out: Path, axis_meta: dict[str, dict[str, Any]]) -> None:
    metrics_dir = project_path(model_dir) / "metrics"
    rows = []
    for axis in ["M1", "M2", "M6", "M7"]:
        path = metrics_dir / f"{axis.lower()}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing axis metrics for {axis}: {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))

    frame = pd.DataFrame(rows).sort_values("axis")
    expected_columns = [
        "candidate_id",
        "model_id",
        "axis",
        "axis_name",
        "model_label",
        "hyperparameters_json",
        "validation_auc",
        "validation_brier",
        "artifact_path",
    ]
    missing = [col for col in expected_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing metrics columns: {missing}")
    out = project_path(metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame[expected_columns].to_csv(out, index=False)
    print(frame[["model_id", "candidate_id", "validation_auc", "validation_brier", "artifact_path"]].to_string(index=False))
    print(f"Saved {out.relative_to(PROJECT_ROOT)}")


def dry_run_report(args: argparse.Namespace, context: dict[str, Any]) -> None:
    train = read_frame(args.train_split)
    validation = read_frame(args.val_split)
    target = context["TARGET"]
    id_col = context["ID_COL"]
    categorical_features = context["CATEGORICAL_FEATURES"]
    raw_features = model_feature_columns(train, id_col=id_col, target=target)
    one_hot_tree_frame = context["one_hot_tree_frame"]
    train_tree = one_hot_tree_frame(train, raw_features)
    print(f"{args.dataset} candidate-pool dry run")
    print(f"train_rows={len(train)} val_rows={len(validation)}")
    print(f"feature_count={len(raw_features)} target={target} id_col={id_col}")
    print(f"categorical_features={categorical_features}")
    print(f"one_hot_feature_count={len(train_tree.columns)}")
    print(f"validation_base_rate_threshold={validation[target].mean():.6f}")
    print(f"M1_candidates={len(load_candidate_config('M1'))}")
    print(f"M2_candidates={len(load_candidate_config('M2'))}")
    print("M6_candidates=9")
    print(f"M7_interactions0_candidates={sum(1 for spec in load_candidate_config('M7') if spec.get('interactions') == 0)}")


def main() -> None:
    args = parse_args()
    context = configure_dataset(args.dataset)

    if args.dry_run:
        dry_run_report(args, context)
        return

    if args.write_metrics:
        write_metrics_csv(args.model_dir, args.metrics_out, context["AXES"])
        return

    if not args.axis:
        raise SystemExit("--axis is required unless --dry-run or --write-metrics is used")

    train = read_frame(args.train_split)
    validation = read_frame(args.val_split)
    target = context["TARGET"]
    id_col = context["ID_COL"]
    categorical_features = context["CATEGORICAL_FEATURES"]
    random_state = context["RANDOM_STATE"]
    raw_features = model_feature_columns(train, id_col=id_col, target=target)
    validation_base_rate_threshold = float(validation[target].mean())
    y_train = train[target].to_numpy()
    y_val = validation[target].to_numpy()

    one_hot_tree_frame = context["one_hot_tree_frame"]
    train_tree = one_hot_tree_frame(train, raw_features)
    validation_tree = one_hot_tree_frame(validation, raw_features, list(train_tree.columns))
    tree_features = list(train_tree.columns)

    start = time.perf_counter()
    if args.axis == "M1":
        rows = train_m1(
            train=train,
            validation=validation,
            raw_features=raw_features,
            y_train=y_train,
            y_val=y_val,
            random_state=random_state,
        )
    elif args.axis == "M2":
        rows = train_m2(
            train_tree=train_tree,
            validation_tree=validation_tree,
            y_train=y_train,
            y_val=y_val,
            random_state=random_state,
        )
    elif args.axis == "M6":
        rows = train_m6(
            train_tree=train_tree,
            validation_tree=validation_tree,
            y_train=y_train,
            y_val=y_val,
            random_state=random_state,
        )
    else:
        rows = train_m7(
            dataset=args.dataset,
            train=train,
            validation=validation,
            raw_features=raw_features,
            train_tree=train_tree,
            validation_tree=validation_tree,
            tree_features=tree_features,
            y_train=y_train,
            y_val=y_val,
            random_state=random_state,
            categorical_features=categorical_features,
        )
    wall_clock_seconds = time.perf_counter() - start

    write_axis_outputs(
        dataset=args.dataset,
        axis=args.axis,
        model_rows=rows,
        train_split=args.train_split,
        val_split=args.val_split,
        model_dir=args.model_dir,
        raw_features=raw_features,
        tree_features=tree_features,
        validation_base_rate_threshold=validation_base_rate_threshold,
        wall_clock_seconds=wall_clock_seconds,
        axis_meta=context["AXES"],
        random_state=random_state,
    )


if __name__ == "__main__":
    main()
