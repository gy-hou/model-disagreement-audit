"""Compute Rashomon Capacity per plan6_amendment_04.md R2.

Wave 4 is evaluation-only: this module loads existing candidate artifacts,
filters the axis-local epsilon-AUC Rashomon set, opens the test split for
prediction, and writes per-instance RC signals plus a dataset summary.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20260512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "uci",
            "fico",
            "synthetic",
            "lendingclub",
            "hmda",
            "gmc",
            "home_credit",
            "home_credit_no_ext",
            "home_credit_h2",
            "hmeq",
            "synthetic_v2",
            "synthetic_strong",
            # n_g accrual step registered in preregistration_v0_7.md §4/§5.
            "german_credit",
            "australian_credit",
        ],
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path)
    parser.add_argument("--epsilon", type=float, default=0.010)
    parser.add_argument("--scores-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported split format: {path}")


def infer_validation_split(test_split: Path) -> Path:
    path = resolve_path(test_split)
    name = path.name
    if "_test" not in name:
        raise ValueError(f"Cannot infer validation split from test split name: {test_split}")
    return path.with_name(name.replace("_test", "_validation"))


def filter_rashomon_set(metrics: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    required = {"axis", "validation_auc", "artifact_path"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Candidate metrics missing required columns: {sorted(missing)}")
    filtered = metrics.copy()
    filtered["axis_auc_best"] = filtered.groupby("axis")["validation_auc"].transform("max")
    filtered["axis_floor"] = filtered["axis_auc_best"] - epsilon
    filtered["in_rashomon"] = filtered["validation_auc"] + 1e-6 >= filtered["axis_floor"]
    return filtered[filtered["in_rashomon"]].reset_index(drop=True)


def compute_predictions_matrix(rashomon_df: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
    from v0_4_common import load_artifact, predict_artifact

    preds: list[np.ndarray] = []
    for _, row in rashomon_df.iterrows():
        artifact_path = resolve_path(Path(row["artifact_path"]))
        if not artifact_path.exists():
            raise FileNotFoundError(f"Candidate artifact not found: {artifact_path}")
        artifact = load_artifact(artifact_path)
        p = np.asarray(predict_artifact(artifact, test_frame), dtype=float)
        if p.ndim != 1:
            raise ValueError(f"Prediction vector must be 1D for {artifact_path}; got shape {p.shape}")
        if np.any(~np.isfinite(p)) or np.any((p < -1e-9) | (p > 1 + 1e-9)):
            raise ValueError(f"Predictions outside [0, 1] for {artifact_path}")
        preds.append(np.clip(p, 0.0, 1.0))
    if not preds:
        raise ValueError("Rashomon set is empty")
    return np.column_stack(preds)


def compute_rc_per_instance(p: np.ndarray, base_rate_threshold: float) -> pd.DataFrame:
    rc_range = p.max(axis=1) - p.min(axis=1)
    rc_var = p.std(axis=1)
    labels = (p >= base_rate_threshold).astype(int)
    pos_frac = labels.mean(axis=1)
    rc_label = 2 * np.minimum(pos_frac, 1.0 - pos_frac)
    return pd.DataFrame({"rc_range": rc_range, "rc_var": rc_var, "rc_label": rc_label})


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        medians[i] = np.median(sample)
    lo = float(np.quantile(medians, alpha / 2))
    hi = float(np.quantile(medians, 1 - alpha / 2))
    if lo == hi and np.ptp(values) > 0:
        raise ValueError("Bootstrap CI degenerated despite non-constant input values")
    return lo, hi


def format_float(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_summary(
    *,
    dataset: str,
    metrics_path: Path,
    test_split: Path,
    val_split: Path,
    epsilon: float,
    validation_base_rate: float,
    rashomon_df: pd.DataFrame,
    rc: pd.DataFrame,
    ci: tuple[float, float],
    n_bootstrap: int,
) -> str:
    q1 = float(rc["rc_range"].quantile(0.25))
    q3 = float(rc["rc_range"].quantile(0.75))
    rows = [
        {
            "candidate_id": row.get("candidate_id", ""),
            "model_id": row.get("model_id", ""),
            "axis": int(row["axis"]),
            "validation_auc": float(row["validation_auc"]),
            "axis_floor": float(row["axis_floor"]),
            "artifact_path": row["artifact_path"],
        }
        for _, row in rashomon_df.sort_values(["axis", "validation_auc"], ascending=[True, False]).iterrows()
    ]
    return (
        f"# Rashomon Capacity Summary — {dataset}\n\n"
        "> Generated by `src/compute_rashomon_capacity.py`. Rashomon set uses axis-local epsilon-AUC with no Brier filter.\n\n"
        "## Inputs\n\n"
        f"- Dataset: {dataset}\n"
        f"- Candidate metrics: `{metrics_path.as_posix()}`\n"
        f"- Test split: `{test_split.as_posix()}`\n"
        f"- Validation split for base-rate threshold: `{val_split.as_posix()}`\n"
        f"- Epsilon: {epsilon:.3f}\n"
        f"- Validation base-rate threshold: {validation_base_rate:.6f}\n\n"
        "## Summary\n\n"
        f"- rashomon_set_size: {len(rashomon_df)}\n"
        f"- test_size: {len(rc)}\n"
        f"- rc_range_median: {rc['rc_range'].median():.6f}\n"
        f"- rc_range_q1: {q1:.6f}\n"
        f"- rc_range_q3: {q3:.6f}\n"
        f"- rc_range_iqr: {(q3 - q1):.6f}\n"
        f"- rc_var_median: {rc['rc_var'].median():.6f}\n"
        f"- rc_label_mean: {rc['rc_label'].mean():.6f}\n"
        f"- bootstrap_ci_lower: {ci[0]:.6f}\n"
        f"- bootstrap_ci_upper: {ci[1]:.6f}\n"
        f"- bootstrap_resamples: {n_bootstrap}\n\n"
        "## Rashomon Set Inventory\n\n"
        f"{markdown_table(rows, ['candidate_id', 'model_id', 'axis', 'validation_auc', 'axis_floor', 'artifact_path'])}\n"
    )


def main() -> None:
    args = parse_args()
    os.environ["V0_4_DATASET"] = args.dataset

    from data_preprocessing import ID_COL, TARGET

    metrics_path = resolve_path(args.metrics)
    test_split = resolve_path(args.test_split)
    val_split = resolve_path(args.val_split) if args.val_split else infer_validation_split(args.test_split)
    scores_out = resolve_path(args.scores_out)
    summary_out = resolve_path(args.summary_out)

    metrics = pd.read_csv(metrics_path)
    rashomon_df = filter_rashomon_set(metrics, args.epsilon)
    if len(rashomon_df) < 4:
        raise SystemExit(f"Rashomon set size {len(rashomon_df)} < 4. Halting per amendment 04 R2.1.")

    validation = read_frame(val_split)
    test = read_frame(test_split)
    if TARGET not in validation.columns or TARGET not in test.columns:
        raise ValueError(f"Target column {TARGET!r} missing for dataset {args.dataset}")
    if ID_COL not in test.columns:
        test = test.copy()
        test.insert(0, ID_COL, np.arange(len(test)))
    validation_base_rate = float(validation[TARGET].mean())

    p = compute_predictions_matrix(rashomon_df, test)
    rc = compute_rc_per_instance(p, validation_base_rate)
    if not rc[["rc_range", "rc_var", "rc_label"]].apply(lambda col: col.between(0, 1).all()).all():
        raise ValueError("RC columns must all be in [0, 1]")

    ci = bootstrap_ci(rc["rc_range"].to_numpy(), n_boot=args.n_bootstrap)
    out = pd.concat([test[[ID_COL]].reset_index(drop=True), rc], axis=1)
    out = out.rename(columns={ID_COL: "row_id"})
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(scores_out, index=False)

    summary = build_summary(
        dataset=args.dataset,
        metrics_path=args.metrics,
        test_split=args.test_split,
        val_split=val_split.relative_to(PROJECT_ROOT) if val_split.is_relative_to(PROJECT_ROOT) else val_split,
        epsilon=args.epsilon,
        validation_base_rate=validation_base_rate,
        rashomon_df=rashomon_df,
        rc=rc,
        ci=ci,
        n_bootstrap=args.n_bootstrap,
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(summary, encoding="utf-8")

    print(
        f"{args.dataset}: R(epsilon)={len(rashomon_df)}, "
        f"rc_range_median={rc['rc_range'].median():.6f}, "
        f"CI=[{ci[0]:.6f}, {ci[1]:.6f}]"
    )
    print(f"Saved {scores_out.relative_to(PROJECT_ROOT)}")
    print(f"Saved {summary_out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
