from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import (
    CATEGORICAL_FEATURES,
    ID_COL,
    PROJECT_ROOT,
    TARGET,
    one_hot_tree_frame,
)


RANDOM_STATE = 20260504
TABLES_V04_DIR = PROJECT_ROOT / "results" / "tables" / "v0_4"
MODELS_V04_DIR = PROJECT_ROOT / "models" / "v0_4"
CANDIDATES_DIR = MODELS_V04_DIR / "candidates"

CANONICAL_MODEL_FILES = {
    "M1": "m1_lr.pkl",
    "M2": "m2_xgb.pkl",
    "M6": "m6_rf.pkl",
    "M7": "m7_ebm.pkl",
}

AXES = {
    "M1": {
        "axis": 1,
        "axis_name": "linear / additive-linear in raw features",
        "model_label": "Logistic Regression (L2)",
    },
    "M2": {
        "axis": 2,
        "axis_name": "gradient-boosted decision trees",
        "model_label": "XGBoost default",
    },
    "M6": {
        "axis": 3,
        "axis_name": "bagged decision trees",
        "model_label": "Random Forest",
    },
    "M7": {
        "axis": 4,
        "axis_name": "additive shape functions on individual features",
        "model_label": "Explainable Boosting Machine",
    },
}


def ensure_v0_4_dirs() -> None:
    TABLES_V04_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)


def load_split(path: Path) -> pd.DataFrame:
    if "test" in path.name.lower():
        raise ValueError(f"Phase 10 must not load the test split: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported split format: {path}")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in {ID_COL, TARGET}]


def labels_from_top_fraction(
    probs: np.ndarray,
    fraction: float,
    ids: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    n_flag = max(1, int(round(float(fraction) * len(probs))))
    labels = np.zeros(len(probs), dtype=int)
    if ids is None:
        order = np.argsort(-probs, kind="mergesort")
    else:
        order = np.lexsort((ids, -probs))
    selected = order[:n_flag]
    labels[selected] = 1
    return labels, float(np.min(probs[selected])), int(n_flag)


def top_fraction_ids(
    probs: np.ndarray,
    ids: np.ndarray,
    fraction: float,
) -> set[int]:
    n_flag = max(1, int(round(float(fraction) * len(probs))))
    order = np.lexsort((ids, -probs))
    return set(int(value) for value in ids[order[:n_flag]])


def predict_artifact(artifact: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    model = artifact["model"]
    input_type = artifact["input_type"]
    if input_type == "raw":
        x = frame[artifact["feature_columns"]]
    elif input_type == "one_hot":
        categorical_features = artifact.get("categorical_features")
        if categorical_features is None:
            x = one_hot_tree_frame(
                frame,
                artifact["raw_feature_columns"],
                artifact["feature_columns"],
            )
        else:
            raw_features = list(artifact["raw_feature_columns"])
            categorical = [
                column for column in categorical_features if column in raw_features
            ]
            x = pd.get_dummies(
                frame[raw_features],
                columns=categorical,
                drop_first=False,
            ).reindex(columns=artifact["feature_columns"], fill_value=0)
    else:
        raise ValueError(f"Unknown artifact input_type: {input_type}")
    if artifact.get("predict_as_numpy", False):
        x = x.to_numpy()
    return model.predict_proba(x)[:, 1]


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Unexpected model artifact format: {path}")
    return artifact


def save_artifact(artifact: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def copy_selected_artifact(candidate_path: Path, model_id: str, voters_dir: Path | None = None) -> Path:
    destination_root = voters_dir if voters_dir is not None else MODELS_V04_DIR
    destination = destination_root / CANONICAL_MODEL_FILES[model_id]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, destination)
    return destination


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, bool):
            return "PASS" if value else "FAIL"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


# v0.5 gate functions (pairwise_gate_metrics, pair_diagnostics) removed in v0.6
# cycle per plan8.md §6. Legacy snapshot preserved at
# legacy/v0_5_gate/v0_4_common_snapshot.py. v0.6 committee observability
# diagnostic is in src/v0_6_observability.py.
