"""Audit-budget routing simulation for v0.6 section 5.6.

The module consumes sealed test/validation splits, sealed voter assignments,
and sealed per-instance Rashomon Capacity signals. It never trains or mutates
models. Bootstrap indices are paired across every strategy and metric within a
fixed (dataset, budget) cell, per the 2026-06-19 amendment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
BASE_SEED = 20260619
N_BOOTSTRAP = 1000
DATASET_ORDER = (
    "uci",
    "fico",
    "hmda",
    "lendingclub",
    "gmc",
    "hmeq",
    "home_credit",
    "synthetic_strong",
    "freddie_mac_2018",
    "fannie_mae_2018",
)
BUDGETS = (1, 5, 10, 15, 20)
METRICS = ("MultiplicityFN_capture", "FN_capture", "HighRC_capture")
MAIN_STRATEGIES = ("Random", "ThresholdProb", "RC-only", "Committee", "Joint(0.5)")
STRATEGY_VARIANTS = (
    "Random",
    "ThresholdProb",
    "RC-only",
    "Committee",
    "Joint(0.3)",
    "Joint(0.5)",
    "Joint(0.7)",
)
PAIRWISE_COMPARISONS = (
    ("ThresholdProb", "Random"),
    ("RC-only", "Random"),
    ("RC-only", "ThresholdProb"),
    ("Committee", "ThresholdProb"),
    ("Joint(0.5)", "RC-only"),
    ("Joint(0.5)", "Committee"),
    ("Joint(0.5)", "ThresholdProb"),
)


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    test_path: Path
    validation_path: Path
    voters_path: Path
    rc_path: Path
    target: str
    id_column: str
    categorical_features: tuple[str, ...]
    y_semantic: str
    m2_predictions_path: Path | None = None
    committee_range_path: Path | None = None
    deployed_m2_artifact_path: Path | None = None


@dataclass
class PreparedDataset:
    dataset: str
    row_ids: np.ndarray
    y_true: np.ndarray
    y_pred_deployed: np.ndarray
    rc_range: np.ndarray
    rc_label: np.ndarray
    committee_range: np.ndarray
    p_deployed: np.ndarray
    tau: float
    scores: dict[str, np.ndarray]
    deployed_candidate_id: str
    voters_json: str
    y_semantic: str


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "uci": DatasetConfig(
        key="uci",
        test_path=Path("data/processed/uci_test.csv"),
        validation_path=Path("data/processed/uci_validation.csv"),
        voters_path=Path("results/tables/v0_4/uci_control/selected_voters.json"),
        rc_path=Path("results/scores/v0_4_uci/rc_per_instance.csv"),
        target="default_payment_next_month",
        id_column="ID",
        categorical_features=("SEX", "EDUCATION", "MARRIAGE"),
        y_semantic="default",
    ),
    "fico": DatasetConfig(
        key="fico",
        test_path=Path("data/processed/fico/fico_test.csv"),
        validation_path=Path("data/processed/fico/fico_validation.csv"),
        voters_path=Path("results/tables/v0_4/selected_voters_fico.json"),
        rc_path=Path("results/scores/v0_4_fico/rc_per_instance.csv"),
        target="is_at_risk",
        id_column="row_id",
        categorical_features=(),
        y_semantic="default",
    ),
    "hmda": DatasetConfig(
        key="hmda",
        test_path=Path("data/processed/hmda/test.parquet"),
        validation_path=Path("data/processed/hmda/val.parquet"),
        voters_path=Path("results/tables/v0_5/selected_voters_hmda.json"),
        rc_path=Path("results/scores/v0_5_hmda/rc_per_instance.csv"),
        target="y",
        id_column="row_id",
        categorical_features=(),
        y_semantic="origination",
    ),
    "lendingclub": DatasetConfig(
        key="lendingclub",
        test_path=Path("data/processed/lendingclub/test.parquet"),
        validation_path=Path("data/processed/lendingclub/val.parquet"),
        voters_path=Path("results/tables/v0_5/selected_voters_lendingclub.json"),
        rc_path=Path("results/scores/v0_5_lendingclub/rc_per_instance.csv"),
        target="default",
        id_column="row_id",
        categorical_features=(
            "grade",
            "home_ownership",
            "verification_status",
            "purpose",
            "emp_length_binned",
        ),
        y_semantic="default",
    ),
    "gmc": DatasetConfig(
        key="gmc",
        test_path=Path("data/processed/gmc_test.csv"),
        validation_path=Path("data/processed/gmc_validation.csv"),
        voters_path=Path("results/tables/v0_4/selected_voters_gmc.json"),
        rc_path=Path("results/tables/v0_4/rashomon_label_per_instance_gmc.csv"),
        target="SeriousDlqin2yrs",
        id_column="row_id",
        categorical_features=(),
        y_semantic="default",
    ),
    "hmeq": DatasetConfig(
        key="hmeq",
        test_path=Path("data/processed/hmeq_test.csv"),
        validation_path=Path("data/processed/hmeq_validation.csv"),
        voters_path=Path("results/tables/v0_4/selected_voters_hmeq.json"),
        rc_path=Path("results/tables/v0_4/rashomon_label_per_instance_hmeq.csv"),
        target="BAD",
        id_column="row_id",
        categorical_features=("REASON", "JOB"),
        y_semantic="default",
    ),
    "home_credit": DatasetConfig(
        key="home_credit",
        test_path=Path("data/processed/home_credit_test.csv"),
        validation_path=Path("data/processed/home_credit_validation.csv"),
        voters_path=Path("results/tables/v0_4/selected_voters_home_credit.json"),
        rc_path=Path("results/tables/v0_4/rashomon_label_per_instance_home_credit.csv"),
        target="TARGET",
        id_column="row_id",
        categorical_features=(
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "WEEKDAY_APPR_PROCESS_START",
            "ORGANIZATION_TYPE",
            "EMERGENCYSTATE_MODE",
        ),
        y_semantic="default",
    ),
    "synthetic_strong": DatasetConfig(
        key="synthetic_strong",
        test_path=Path("data/processed/synthetic_strong_test.csv"),
        validation_path=Path("data/processed/synthetic_strong_validation.csv"),
        voters_path=Path("results/tables/v0_4/selected_voters_synthetic_strong.json"),
        rc_path=Path("results/tables/v0_4/rashomon_label_per_instance_synthetic_strong.csv"),
        target="y",
        id_column="row_id",
        categorical_features=(),
        y_semantic="default",
    ),
    "freddie_mac_2018": DatasetConfig(
        key="freddie_mac_2018",
        test_path=Path(
            "data/processed/freddie_mac_2018_ever_60dpd_24m/test.parquet"
        ),
        validation_path=Path(
            "data/processed/freddie_mac_2018_ever_60dpd_24m/val.parquet"
        ),
        voters_path=Path(
            "results/tables/v0_7/selected_voters_freddie_mac_2018.json"
        ),
        rc_path=Path(
            "results/scores/v0_7_freddie_mac_2018/rc_per_instance.csv"
        ),
        target="ever_60dpd_24m",
        id_column="row_id",
        categorical_features=(),
        y_semantic="default",
        m2_predictions_path=Path(
            "results/scores/v0_7_freddie_mac_2018/m2_predictions.csv"
        ),
        committee_range_path=Path(
            "results/scores/v0_7_freddie_mac_2018/committee_range.csv"
        ),
        deployed_m2_artifact_path=Path(
            "models/v0_7/deployment/freddie_mac_2018/m2_xgb.joblib"
        ),
    ),
    "fannie_mae_2018": DatasetConfig(
        key="fannie_mae_2018",
        test_path=Path(
            "data/processed/fannie_mae_2018_ever_60dpd_24m/test.parquet"
        ),
        validation_path=Path(
            "data/processed/fannie_mae_2018_ever_60dpd_24m/val.parquet"
        ),
        voters_path=Path(
            "results/tables/v0_7/selected_voters_fannie_mae_2018.json"
        ),
        rc_path=Path(
            "results/scores/v0_7_fannie_mae_2018/rc_per_instance.csv"
        ),
        target="ever_60dpd_24m",
        id_column="row_id",
        categorical_features=(),
        y_semantic="default",
        m2_predictions_path=Path(
            "results/scores/v0_7_fannie_mae_2018/m2_predictions.csv"
        ),
        committee_range_path=Path(
            "results/scores/v0_7_fannie_mae_2018/committee_range.csv"
        ),
        deployed_m2_artifact_path=Path(
            "models/v0_7/deployment/fannie_mae_2018/m2_xgb.joblib"
        ),
    ),
}


def resolve(path: Path) -> Path:
    if not path.is_absolute():
        return PROJECT_ROOT / path
    if path.exists():
        return path
    # Voter manifests store absolute artifact paths from the checkout that wrote
    # them, so they break when the project directory moves. Rebase onto the
    # longest trailing run of components that exists here. Paths that already
    # resolve return above, so this cannot redirect a live artifact.
    parts = path.parts
    for start in range(1, len(parts)):
        candidate = PROJECT_ROOT.joinpath(*parts[start:])
        if candidate.exists():
            return candidate
    return path


def load_frame(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported frame format: {path}")


def committee_range(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"Committee predictions must be a non-empty 2D matrix; got {values.shape}")
    return values.max(axis=1) - values.min(axis=1)


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    span = float(np.max(values) - lo)
    if span == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / span


def fixed_random_scores(row_ids: np.ndarray, seed: int = BASE_SEED) -> np.ndarray:
    """Map (seed, row_id) deterministically to [0, 1), independent of row order."""

    scores = np.empty(len(row_ids), dtype=np.float64)
    denominator = float(1 << 64)
    for index, row_id in enumerate(row_ids):
        if isinstance(row_id, (np.integer, int)):
            stable_id = str(int(row_id))
        elif isinstance(row_id, (np.floating, float)) and float(row_id).is_integer():
            stable_id = str(int(row_id))
        else:
            stable_id = str(row_id)
        digest = hashlib.blake2b(
            f"{seed}:{stable_id}".encode("utf-8"), digest_size=8
        ).digest()
        scores[index] = int.from_bytes(digest, "big") / denominator
    return scores


def threshold_probability_scores(probabilities: np.ndarray, tau: float) -> np.ndarray:
    return -np.abs(np.asarray(probabilities, dtype=float) - float(tau))


def strategy_order(scores: np.ndarray, row_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((np.asarray(row_ids), -np.asarray(scores, dtype=float)))


def selected_mask(scores: np.ndarray, row_ids: np.ndarray, budget_pct: int) -> np.ndarray:
    n_selected = max(1, int(round((budget_pct / 100.0) * len(scores))))
    order = strategy_order(scores, row_ids)
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:n_selected]] = True
    return mask


def compute_capture(selected: np.ndarray, target: np.ndarray) -> float:
    selected = np.asarray(selected, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(target.sum())
    if denominator == 0:
        return float("nan")
    return float(np.logical_and(selected, target).sum() / denominator)


def generate_bootstrap_indices(
    n_rows: int,
    *,
    dataset_idx: int,
    budget_idx: int,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BASE_SEED,
) -> np.ndarray:
    cell_seed = int(seed + dataset_idx * 100 + budget_idx)
    rng = np.random.default_rng(cell_seed)
    dtype = np.int32 if n_rows <= np.iinfo(np.int32).max else np.int64
    return rng.integers(0, n_rows, size=(n_bootstrap, n_rows), dtype=dtype)


def find_deployed_m2(voters: Iterable[dict[str, Any]]) -> dict[str, Any]:
    matches = [voter for voter in voters if str(voter.get("model_id")) == "M2"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one deployed M2 voter; found {len(matches)}")
    return matches[0]


def _predict_artifact(
    artifact: dict[str, Any],
    test_frame: pd.DataFrame,
    categorical_features: tuple[str, ...],
) -> np.ndarray:
    input_type = artifact["input_type"]
    if input_type == "raw":
        x = test_frame[artifact["feature_columns"]]
    elif input_type == "one_hot":
        raw_columns = list(artifact["raw_feature_columns"])
        artifact_categories = artifact.get("categorical_features")
        category_source = (
            artifact_categories
            if artifact_categories is not None
            else categorical_features
        )
        categories = [column for column in category_source if column in raw_columns]
        x = pd.get_dummies(
            test_frame[raw_columns],
            columns=categories,
            drop_first=False,
        ).reindex(columns=artifact["feature_columns"], fill_value=0)
        x = x.astype(float)
    else:
        raise ValueError(f"Unknown artifact input_type: {input_type}")
    if artifact.get("predict_as_numpy", False):
        x = x.to_numpy()
    probabilities = np.asarray(artifact["model"].predict_proba(x)[:, 1], dtype=float)
    if probabilities.shape != (len(test_frame),):
        raise ValueError(f"Prediction shape {probabilities.shape} != ({len(test_frame)},)")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite voter probabilities")
    return probabilities


def load_aligned_score(
    path: Path,
    *,
    score_column: str,
    row_ids: np.ndarray,
) -> np.ndarray:
    frame = load_frame(path)
    required = {"row_id", score_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} missing {sorted(required - set(frame.columns))}")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row IDs in {path}")
    aligned = frame.set_index("row_id").reindex(row_ids)
    if aligned[score_column].isna().any():
        raise ValueError(f"{score_column} join failed for {path}")
    values = aligned[score_column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{score_column} contains non-finite values in {path}")
    return values


def load_committee_predictions(
    config: DatasetConfig,
    test_frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    state = json.loads(resolve(config.voters_path).read_text(encoding="utf-8"))
    voters = list(state["selected_voters"])
    probabilities: dict[str, np.ndarray] = {}
    for voter in voters:
        model_id = str(voter["model_id"])
        artifact_path = resolve(Path(voter["artifact_path"]))
        artifact = joblib.load(artifact_path)
        probabilities[model_id] = _predict_artifact(
            artifact,
            test_frame,
            config.categorical_features,
        )
    return probabilities, voters


def build_strategy_scores(
    row_ids: np.ndarray,
    p_deployed: np.ndarray,
    tau: float,
    rc_range_values: np.ndarray,
    committee_range_values: np.ndarray,
    *,
    seed: int = BASE_SEED,
) -> dict[str, np.ndarray]:
    normalized_rc = minmax_normalize(rc_range_values)
    normalized_committee = minmax_normalize(committee_range_values)
    scores = {
        "Random": fixed_random_scores(row_ids, seed),
        "ThresholdProb": threshold_probability_scores(p_deployed, tau),
        "RC-only": np.asarray(rc_range_values, dtype=float).copy(),
        "Committee": np.asarray(committee_range_values, dtype=float).copy(),
    }
    for alpha in (0.3, 0.5, 0.7):
        scores[f"Joint({alpha:.1f})"] = (
            alpha * normalized_rc + (1.0 - alpha) * normalized_committee
        )
    return scores


def prepare_dataset(dataset_key: str, *, seed: int = BASE_SEED) -> PreparedDataset:
    if dataset_key not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset: {dataset_key}")
    config = DATASET_CONFIGS[dataset_key]
    test = load_frame(config.test_path)
    validation = load_frame(config.validation_path)
    required_test = {config.id_column, config.target}
    if not required_test.issubset(test.columns) or config.target not in validation.columns:
        raise ValueError(f"Missing ID/target columns for {dataset_key}")

    row_ids = test[config.id_column].to_numpy()
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError(f"Duplicate row IDs in {dataset_key} test split")
    y_true = test[config.target].to_numpy(dtype=int)
    tau = float(validation[config.target].mean())

    state = json.loads(resolve(config.voters_path).read_text(encoding="utf-8"))
    voters = list(state["selected_voters"])
    if config.deployed_m2_artifact_path is None:
        deployed = find_deployed_m2(voters)
    else:
        deployed_artifact = joblib.load(resolve(config.deployed_m2_artifact_path))
        deployed = {
            "model_id": "M2",
            "candidate_id": str(deployed_artifact["candidate_id"]),
            "artifact_path": config.deployed_m2_artifact_path.as_posix(),
        }
    probability_map: dict[str, np.ndarray] | None = None
    if config.m2_predictions_path is not None:
        p_deployed = load_aligned_score(
            config.m2_predictions_path,
            score_column="m2_probability",
            row_ids=row_ids,
        )
    else:
        probability_map, _ = load_committee_predictions(config, test)
        p_deployed = probability_map["M2"]

    if config.committee_range_path is not None:
        committee_values = load_aligned_score(
            config.committee_range_path,
            score_column="committee_range",
            row_ids=row_ids,
        )
    else:
        if probability_map is None:
            probability_map, _ = load_committee_predictions(config, test)
        probability_matrix = np.column_stack(
            [probability_map[str(voter["model_id"])] for voter in voters]
        )
        committee_values = committee_range(probability_matrix)

    rc = pd.read_csv(resolve(config.rc_path))
    required_rc = {"row_id", "rc_range", "rc_label"}
    if not required_rc.issubset(rc.columns):
        raise ValueError(f"{config.rc_path} missing {sorted(required_rc - set(rc.columns))}")
    if rc["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row IDs in {config.rc_path}")
    aligned = rc.set_index("row_id").reindex(row_ids)
    if aligned[["rc_range", "rc_label"]].isna().any().any():
        raise ValueError(f"RC join failed for {dataset_key}")
    rc_range_values = aligned["rc_range"].to_numpy(dtype=float)
    rc_label_values = aligned["rc_label"].to_numpy(dtype=float)
    y_pred = (p_deployed >= tau).astype(int)
    scores = build_strategy_scores(
        row_ids,
        p_deployed,
        tau,
        rc_range_values,
        committee_values,
        seed=seed,
    )
    return PreparedDataset(
        dataset=dataset_key,
        row_ids=row_ids,
        y_true=y_true,
        y_pred_deployed=y_pred,
        rc_range=rc_range_values,
        rc_label=rc_label_values,
        committee_range=committee_values,
        p_deployed=p_deployed,
        tau=tau,
        scores=scores,
        deployed_candidate_id=str(deployed["candidate_id"]),
        voters_json=str(config.voters_path),
        y_semantic=config.y_semantic,
    )


def metric_targets(prepared: PreparedDataset) -> dict[str, np.ndarray]:
    high_rc = selected_mask(prepared.rc_range, prepared.row_ids, 10)
    return {
        "MultiplicityFN_capture": np.logical_and(
            prepared.rc_label > 0,
            prepared.y_pred_deployed != prepared.y_true,
        ),
        "FN_capture": np.logical_and(
            prepared.y_pred_deployed == 0,
            prepared.y_true == 1,
        ),
        "HighRC_capture": high_rc,
    }


def _encode_targets(targets: dict[str, np.ndarray], metrics: tuple[str, ...]) -> tuple[np.ndarray, int]:
    n_rows = len(next(iter(targets.values())))
    base = n_rows + 1
    code = np.zeros(n_rows, dtype=np.int64)
    multiplier = 1
    for metric in metrics:
        code += targets[metric].astype(np.int64) * multiplier
        multiplier *= base
    if multiplier > np.iinfo(np.int64).max:
        raise OverflowError("Target encoding exceeds int64")
    return code, base


def _decode_counts(code: np.ndarray, *, base: int, n_metrics: int) -> list[np.ndarray]:
    remaining = np.asarray(code, dtype=np.int64).copy()
    decoded: list[np.ndarray] = []
    for _ in range(n_metrics):
        decoded.append(remaining % base)
        remaining //= base
    return decoded


def evaluate_bootstrap_cell(
    prepared: PreparedDataset,
    *,
    budget_pct: int,
    bootstrap_indices: np.ndarray,
    strategies: tuple[str, ...] = STRATEGY_VARIANTS,
    metrics: tuple[str, ...] = METRICS,
    chunk_size: int = 32,
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate all strategies/metrics using one shared bootstrap index matrix."""

    n_bootstrap, n_rows = bootstrap_indices.shape
    if n_rows != len(prepared.row_ids):
        raise ValueError("Bootstrap index width does not match prepared dataset")
    n_selected = max(1, int(round((budget_pct / 100.0) * n_rows)))
    targets = metric_targets(prepared)
    target_code, base = _encode_targets(targets, metrics)
    orders = {
        strategy: strategy_order(prepared.scores[strategy], prepared.row_ids)
        for strategy in strategies
    }
    result = {
        strategy: {
            metric: np.empty(n_bootstrap, dtype=float)
            for metric in metrics
        }
        for strategy in strategies
    }

    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        sampled = bootstrap_indices[start:stop]
        counts = np.stack(
            [np.bincount(row, minlength=n_rows) for row in sampled],
            axis=0,
        ).astype(np.int16, copy=False)
        denominator_code = counts @ target_code
        denominator_values = _decode_counts(
            denominator_code,
            base=base,
            n_metrics=len(metrics),
        )

        for strategy in strategies:
            order = orders[strategy]
            ordered_counts = counts[:, order]
            cumulative_count = np.cumsum(ordered_counts, axis=1, dtype=np.int32)
            boundary = np.argmax(cumulative_count >= n_selected, axis=1)
            row_index = np.arange(stop - start)
            previous_count = np.where(
                boundary > 0,
                cumulative_count[row_index, np.maximum(boundary - 1, 0)],
                0,
            )
            boundary_take = n_selected - previous_count

            ordered_target_code = target_code[order]
            cumulative_target = np.cumsum(
                ordered_counts * ordered_target_code,
                axis=1,
                dtype=np.int64,
            )
            previous_target = np.where(
                boundary > 0,
                cumulative_target[row_index, np.maximum(boundary - 1, 0)],
                0,
            )
            selected_code = (
                previous_target + boundary_take * ordered_target_code[boundary]
            )
            selected_values = _decode_counts(
                selected_code,
                base=base,
                n_metrics=len(metrics),
            )
            for metric_index, metric in enumerate(metrics):
                denominator = denominator_values[metric_index].astype(float)
                numerator = selected_values[metric_index].astype(float)
                values = np.divide(
                    numerator,
                    denominator,
                    out=np.full_like(numerator, np.nan, dtype=float),
                    where=denominator > 0,
                )
                result[strategy][metric][start:stop] = values
    return result


def _summary_statistics(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    if np.isnan(finite).all():
        return {"median": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    return {
        "median": float(np.nanmedian(finite)),
        "ci_lower": float(np.nanquantile(finite, 0.025)),
        "ci_upper": float(np.nanquantile(finite, 0.975)),
    }


def _alpha_for_strategy(strategy: str) -> float:
    if not strategy.startswith("Joint("):
        return float("nan")
    return float(strategy.removeprefix("Joint(").removesuffix(")"))


def paired_sign_p_value(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 1.0
    p_value = 2.0 * min(
        float(np.mean(nonzero < 0)),
        float(np.mean(nonzero > 0)),
    )
    return float(min(1.0, p_value))


def simulate_prepared_dataset(
    prepared: PreparedDataset,
    *,
    dataset_idx: int,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BASE_SEED,
    budgets: tuple[int, ...] = BUDGETS,
    strategies: tuple[str, ...] = STRATEGY_VARIANTS,
    metrics: tuple[str, ...] = METRICS,
) -> dict[str, pd.DataFrame]:
    routing_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []

    for budget_pct in budgets:
        budget_idx = BUDGETS.index(budget_pct)
        indices = generate_bootstrap_indices(
            len(prepared.row_ids),
            dataset_idx=dataset_idx,
            budget_idx=budget_idx,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        replicates = evaluate_bootstrap_cell(
            prepared,
            budget_pct=budget_pct,
            bootstrap_indices=indices,
            strategies=strategies,
            metrics=metrics,
        )
        del indices

        for strategy in strategies:
            summary_row: dict[str, Any] = {
                "dataset": prepared.dataset,
                "strategy": strategy,
                "alpha": _alpha_for_strategy(strategy),
                "budget_pct": budget_pct,
                "y_semantic": prepared.y_semantic,
                "deployed_candidate_id": prepared.deployed_candidate_id,
            }
            for metric in metrics:
                statistics = _summary_statistics(replicates[strategy][metric])
                metric_prefix = {
                    "MultiplicityFN_capture": "multiplicity_fn_capture",
                    "FN_capture": "fn_capture",
                    "HighRC_capture": "high_rc_capture",
                }[metric]
                for statistic, value in statistics.items():
                    routing_rows.append(
                        {
                            "dataset": prepared.dataset,
                            "strategy": strategy,
                            "alpha": _alpha_for_strategy(strategy),
                            "budget_pct": budget_pct,
                            "metric": metric,
                            "statistic": statistic,
                            "value": value,
                        }
                    )
                    summary_row[f"{metric_prefix}_{statistic}"] = value
            summary_rows.append(summary_row)

        available_pairs = [
            pair
            for pair in PAIRWISE_COMPARISONS
            if pair[0] in strategies and pair[1] in strategies
        ]
        for strategy_a, strategy_b in available_pairs:
            for metric in metrics:
                differences = (
                    replicates[strategy_a][metric] - replicates[strategy_b][metric]
                )
                statistics = _summary_statistics(differences)
                gap_rows.append(
                    {
                        "dataset": prepared.dataset,
                        "strategy_A": strategy_a,
                        "strategy_B": strategy_b,
                        "budget_pct": budget_pct,
                        "metric": metric,
                        "gap_median": statistics["median"],
                        "gap_ci_lower": statistics["ci_lower"],
                        "gap_ci_upper": statistics["ci_upper"],
                        "p_value_paired_sign_test": paired_sign_p_value(differences),
                    }
                )
        del replicates

    return {
        "routing": pd.DataFrame(routing_rows),
        "summary": pd.DataFrame(summary_rows),
        "pairwise_gaps": pd.DataFrame(gap_rows),
        "assignment": pd.DataFrame(
            [
                {
                    "dataset": prepared.dataset,
                    "deployed_model_id": "M2",
                    "candidate_id": prepared.deployed_candidate_id,
                    "voters_json": prepared.voters_json,
                }
            ]
        ),
    }


def run_routing_simulation(
    dataset_key: str,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BASE_SEED,
) -> dict[str, pd.DataFrame]:
    prepared = prepare_dataset(dataset_key, seed=seed)
    return simulate_prepared_dataset(
        prepared,
        dataset_idx=DATASET_ORDER.index(dataset_key),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASET_ORDER)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/tables/v0_6/audit_budget"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_routing_simulation(
        args.dataset,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    output_dir = resolve(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{args.dataset}_routing.csv"
    result["routing"].to_csv(path, index=False)
    print(
        f"dataset={args.dataset} rows={len(result['routing'])} "
        f"deployed={result['assignment'].iloc[0]['candidate_id']}"
    )
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
