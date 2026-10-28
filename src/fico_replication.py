from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_preprocessing import PROJECT_ROOT


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "fico" / "heloc.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "fico"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PAPER_DIR = PROJECT_ROOT / "paper"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models" / "fico"

TARGET = "is_at_risk"
RANDOM_STATE = 20260504
RANK_STD_MAX_THREE_MODELS = math.sqrt(2 / 9)


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, TABLES_DIR, PAPER_DIR, DOCS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def labels_from_top_fraction(probs: np.ndarray, fraction: float, ids: np.ndarray) -> tuple[np.ndarray, float]:
    n_flag = max(1, int(round(float(fraction) * len(probs))))
    labels = np.zeros(len(probs), dtype=int)
    order = np.lexsort((ids, -probs))
    labels[order[:n_flag]] = 1
    cutoff = float(np.min(probs[order[:n_flag]]))
    return labels, cutoff


def entropy(pi: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pi, dtype=float)
    mask = (pi > 0) & (pi < 1)
    out[mask] = -(pi[mask] * np.log(pi[mask]) + (1 - pi[mask]) * np.log(1 - pi[mask])) / np.log(2)
    return out


def risk_percentile(probs: np.ndarray) -> np.ndarray:
    return pd.Series(probs).rank(method="average", pct=True).to_numpy()


def ks_statistic(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, probs)
    return float(np.max(tpr - fpr))


def selected_mask(score: np.ndarray, ids: np.ndarray, budget: float) -> np.ndarray:
    n_select = max(1, int(round(float(budget) * len(score))))
    order = np.lexsort((ids, -score))
    out = np.zeros(len(score), dtype=bool)
    out[order[:n_select]] = True
    return out


def capture_rate(selected: np.ndarray, target: np.ndarray) -> float:
    return float((selected & target).sum() / target.sum()) if target.sum() else float("nan")


def bootstrap_gap(
    selected_a: np.ndarray,
    selected_b: np.ndarray,
    target: np.ndarray,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(target)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    weights = np.vstack([np.bincount(row, minlength=n) for row in indices])
    target_counts = weights @ target.astype(int)
    valid = target_counts > 0
    a_counts = weights @ (selected_a & target).astype(int)
    b_counts = weights @ (selected_b & target).astype(int)
    diffs = np.full(n_bootstrap, np.nan)
    diffs[valid] = (a_counts[valid] - b_counts[valid]) / target_counts[valid]
    return float(np.nanquantile(diffs, 0.025)), float(np.nanquantile(diffs, 0.975))


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(RAW_PATH)
    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val[TARGET],
        random_state=RANDOM_STATE,
    )
    for split_name, split_df in [("train", train), ("validation", validation), ("test", test)]:
        split_df.reset_index(drop=True).to_csv(PROCESSED_DIR / f"fico_{split_name}.csv", index=False)
    feature_columns = [col for col in df.columns if col not in {"row_id", TARGET}]
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), feature_columns


def build_lr(feature_columns: list[str]) -> Pipeline:
    preprocess = ColumnTransformer([("num", StandardScaler(), feature_columns)])
    model = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=RANDOM_STATE)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def build_xgb(seed: int, subsample: float = 1.0, colsample_bytree: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )


def headline_for_seed(df: pd.DataFrame, seed: int) -> dict[str, Any]:
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df[TARGET],
        random_state=seed,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val[TARGET],
        random_state=seed,
    )
    feature_columns = [col for col in df.columns if col not in {"row_id", TARGET}]
    x_train = train[feature_columns]
    y_train = train[TARGET].to_numpy()
    y_val = validation[TARGET].to_numpy()
    x_test = test[feature_columns]
    y_test = test[TARGET].to_numpy()
    ids = test["row_id"].to_numpy()
    review_fraction = float(y_val.mean())

    m1 = build_lr(feature_columns)
    m2 = build_xgb(seed)
    m3 = build_xgb(seed + 1, subsample=0.80, colsample_bytree=0.80)
    m1.fit(x_train, y_train)
    m2.fit(x_train, y_train)
    m3.fit(x_train, y_train)

    probs = {
        "M1": m1.predict_proba(x_test)[:, 1],
        "M2": m2.predict_proba(x_test)[:, 1],
        "M3": m3.predict_proba(x_test)[:, 1],
    }
    labels: dict[str, np.ndarray] = {}
    cutoffs: dict[str, float] = {}
    for model_id, p in probs.items():
        labels[model_id], cutoffs[model_id] = labels_from_top_fraction(p, review_fraction, ids)

    vote_matrix = np.vstack([labels["M1"], labels["M2"], labels["M3"]]).T
    h = entropy(vote_matrix.mean(axis=1))
    ranks = np.vstack([risk_percentile(probs["M1"]), risk_percentile(probs["M2"]), risk_percentile(probs["M3"])]).T
    r = np.clip(np.std(ranks, axis=1, ddof=0) / RANK_STD_MAX_THREE_MODELS, 0, 1)
    dgs_core = 0.5 * h + 0.5 * r
    fn_target = (y_test == 1) & (labels["M2"] == 0)
    dgs_selected = selected_mask(dgs_core, ids, 0.10)
    threshold_selected = selected_mask(-np.abs(probs["M2"] - cutoffs["M2"]), ids, 0.10)
    dgs_capture = capture_rate(dgs_selected, fn_target)
    threshold_capture = capture_rate(threshold_selected, fn_target)
    gap = dgs_capture - threshold_capture
    return {
        "seed": seed,
        "test_rows": int(len(test)),
        "fn_total": int(fn_target.sum()),
        "dgs_core_capture": dgs_capture,
        "thresholdprob_capture": threshold_capture,
        "gap": gap,
        "m2_auc": float(roc_auc_score(y_test, probs["M2"])),
    }


def main() -> None:
    ensure_dirs()
    raw_df = pd.read_csv(RAW_PATH).reset_index(drop=False).rename(columns={"index": "row_id"})
    train, validation, test, feature_columns = load_and_split()
    x_train = train[feature_columns]
    y_train = train[TARGET].to_numpy()
    x_val = validation[feature_columns]
    y_val = validation[TARGET].to_numpy()
    x_test = test[feature_columns]
    y_test = test[TARGET].to_numpy()
    ids = test["row_id"].to_numpy()
    review_fraction = float(y_val.mean())

    m1 = build_lr(feature_columns)
    m2 = build_xgb(RANDOM_STATE)
    m3 = build_xgb(RANDOM_STATE + 1, subsample=0.80, colsample_bytree=0.80)
    m1.fit(x_train, y_train)
    m2.fit(x_train, y_train)
    m3.fit(x_train, y_train)

    probs = {
        "M1": m1.predict_proba(x_test)[:, 1],
        "M2": m2.predict_proba(x_test)[:, 1],
        "M3": m3.predict_proba(x_test)[:, 1],
    }
    labels: dict[str, np.ndarray] = {}
    cutoffs: dict[str, float] = {}
    for model_id, p in probs.items():
        labels[model_id], cutoffs[model_id] = labels_from_top_fraction(p, review_fraction, ids)

    vote_matrix = np.vstack([labels["M1"], labels["M2"], labels["M3"]]).T
    h = entropy(vote_matrix.mean(axis=1))
    ranks = np.vstack([risk_percentile(probs["M1"]), risk_percentile(probs["M2"]), risk_percentile(probs["M3"])]).T
    r = np.clip(np.std(ranks, axis=1, ddof=0) / RANK_STD_MAX_THREE_MODELS, 0, 1)
    dgs_core = 0.5 * h + 0.5 * r

    fn_target = (y_test == 1) & (labels["M2"] == 0)
    dgs_selected = selected_mask(dgs_core, ids, 0.10)
    threshold_selected = selected_mask(-np.abs(probs["M2"] - cutoffs["M2"]), ids, 0.10)
    dgs_capture = capture_rate(dgs_selected, fn_target)
    threshold_capture = capture_rate(threshold_selected, fn_target)
    gap = dgs_capture - threshold_capture
    ci_low, ci_high = bootstrap_gap(dgs_selected, threshold_selected, fn_target)

    metric_rows = []
    for model_id in ["M1", "M2", "M3"]:
        metric_rows.append(
            {
                "model_id": model_id,
                "auc": float(roc_auc_score(y_test, probs[model_id])),
                "ks": ks_statistic(y_test, probs[model_id]),
                "brier": float(brier_score_loss(y_test, probs[model_id])),
                "fn": int(((y_test == 1) & (labels[model_id] == 0)).sum()),
                "cutoff_probability": cutoffs[model_id],
            }
        )
    pair_rows = []
    for a, b in [("M1", "M2"), ("M2", "M3")]:
        pair_rows.append(
            {
                "pair": f"{a}-{b}",
                "label_disagreement_rate": float(np.mean(labels[a] != labels[b])),
                "spearman_rank_correlation": float(spearmanr(probs[a], probs[b]).statistic),
            }
        )

    headline = {
        "dataset": "FICO_HELOC_public_mirror",
        "target": "FN",
        "budget": 0.10,
        "dgs_core_capture": dgs_capture,
        "thresholdprob_capture": threshold_capture,
        "gap": gap,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "direction": "replicates_positive_direction" if gap > 0 else "does_not_replicate_positive_direction",
        "fn_total": int(fn_target.sum()),
        "test_rows": int(len(test)),
        "test_positive_rate": float(y_test.mean()),
        "notes": "M3 is an XGBoost seed/subsample challenger because HELOC features are anonymized and PAY_* monotonic constraints are unavailable.",
    }
    pd.DataFrame(metric_rows).to_csv(TABLES_DIR / "fico_baseline_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(TABLES_DIR / "fico_pairwise_disagreement.csv", index=False)
    pd.DataFrame([headline]).to_csv(TABLES_DIR / "fico_headline_replication.csv", index=False)
    sensitivity_rows = [headline_for_seed(raw_df, RANDOM_STATE + offset) for offset in range(20)]
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES_DIR / "fico_split_sensitivity.csv", index=False)
    sensitivity_summary = {
        "n_splits": int(len(sensitivity)),
        "median_gap": float(sensitivity["gap"].median()),
        "min_gap": float(sensitivity["gap"].min()),
        "max_gap": float(sensitivity["gap"].max()),
        "positive_gap_share": float((sensitivity["gap"] > 0).mean()),
        "median_dgs_capture": float(sensitivity["dgs_core_capture"].median()),
        "median_threshold_capture": float(sensitivity["thresholdprob_capture"].median()),
    }
    pd.DataFrame([sensitivity_summary]).to_csv(TABLES_DIR / "fico_split_sensitivity_summary.csv", index=False)

    score_frame = pd.DataFrame(
        {
            "row_id": ids,
            TARGET: y_test,
            "M1_prob": probs["M1"],
            "M2_prob": probs["M2"],
            "M3_prob": probs["M3"],
            "M1_label": labels["M1"],
            "M2_label": labels["M2"],
            "M3_label": labels["M3"],
            "label_entropy": h,
            "rank_dispersion_norm": r,
            "dgs_core": dgs_core,
            "m2_fn": fn_target.astype(int),
        }
    )
    score_frame.to_csv(RESULTS_DIR / "scores" / "fico_dgs_scores.csv", index=False)

    text = (
        "# FICO HELOC Headline Replication - Phase 7\n\n"
        "> Direction-only replication of the UCI H4 headline using a public Hugging Face mirror of the FICO HELOC xML Challenge dataset. "
        "Formal publication should confirm official FICO data-use permissions.\n\n"
        "## Source\n\n"
        "- Public mirror used for this run: https://huggingface.co/datasets/mstz/heloc/tree/main/risk\n"
        "- FICO xML Challenge announcement: https://investors.fico.com/news-releases/news-release-details/fico-announces-xml-challenge\n\n"
        "## Baseline Metrics\n\n"
        f"{markdown_table(metric_rows, list(metric_rows[0].keys()))}\n\n"
        "## Pairwise Disagreement\n\n"
        f"{markdown_table(pair_rows, list(pair_rows[0].keys()))}\n\n"
        "## Headline Result\n\n"
        f"{markdown_table([headline], list(headline.keys()))}\n\n"
        "## Split Sensitivity\n\n"
        f"{markdown_table([sensitivity_summary], list(sensitivity_summary.keys()))}\n"
    )
    (TABLES_DIR / "fico_headline_replication.md").write_text(text, encoding="utf-8")
    (DOCS_DIR / "fico_replication_note.md").write_text(text, encoding="utf-8")

    joblib.dump({"M1": m1, "M2": m2, "M3": m3, "feature_columns": feature_columns}, MODELS_DIR / "fico_models.joblib")
    print("Saved Phase 7 FICO headline replication")
    print(f"gap={gap:.4f}, ci=[{ci_low:.4f}, {ci_high:.4f}], direction={headline['direction']}")


if __name__ == "__main__":
    main()
