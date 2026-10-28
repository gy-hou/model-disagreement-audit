"""Prepare the Op4 HMDA Boston-Cambridge-Newton 2018 splits.

Phase A only: raw LAR CSV -> financial-only model matrices plus a separate
protected-attribute hold-aside keyed by row_id.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "hmda" / "hmda_2018_msa14460_lar.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hmda"
RANDOM_STATE = 42

NUMERIC_FEATURES = {
    "loan_amount": "loan_amount",
    "applicant_income": "income",
}

CATEGORICAL_FEATURES = {
    "loan_type": "loan_type",
    "loan_purpose": "loan_purpose",
    "occupancy_type": "occupancy_type",
    "debt_to_income_ratio": "debt_to_income_ratio",
    "construction_method": "construction_method",
    "manufactured_home_status": "manufactured_home_secured_property_type",
    "total_units": "total_units",
    "county_code": "county_code",
}

# Caps are locked before any HMDA model training or RC computation. The first
# level is retained as the baseline category; retained non-baseline levels and
# "Other" are dummy-coded.
CATEGORY_LEVEL_LIMITS = {
    "loan_type": 3,
    "loan_purpose": 4,
    "occupancy_type": 2,
    "debt_to_income_ratio": 5,
    "construction_method": 2,
    "manufactured_home_status": 2,
    "total_units": 2,
    "county_code": 2,
}

PROTECTED_ATTRIBUTES = {
    "derived_race": "derived_race",
    "derived_ethnicity": "derived_ethnicity",
    "derived_sex": "derived_sex",
    "age": "applicant_age",
}

PROTECTED_TOKENS = ("race", "ethnic", "sex", "age")

REMOVED_FEATURES_AMENDMENT_04 = [
    "hoepa_status",
    "interest_rate",
    "loan_term",
    "property_value",
    "combined_loan_to_value_ratio",
    "lien_status",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"HMDA raw file missing required columns: {missing}")


def filter_property_scope(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "property_type" in df.columns:
        filtered = df[df["property_type"].astype("string").str.strip().eq("1")].copy()
        return filtered, "Filtered legacy property_type == 1."

    require_columns(df, ["derived_dwelling_category"])
    mask = df["derived_dwelling_category"].astype("string").str.startswith(
        "Single Family (1-4 Units)", na=False
    )
    filtered = df[mask].copy()
    return filtered, (
        "Filtered modern 2018 HMDA derived_dwelling_category values starting "
        "with 'Single Family (1-4 Units)' because legacy property_type is absent."
    )


def normalize_categorical(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.fillna("Missing").replace({"": "Missing", "nan": "Missing", "NA": "Missing"})


def build_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pd.DataFrame(
        {
            "row_id": np.arange(1, len(df) + 1, dtype=np.int64),
            "y": (df["action_taken"].astype(int) == 1).astype(int),
        }
    )

    for feature, source in NUMERIC_FEATURES.items():
        model[feature] = pd.to_numeric(df[source], errors="coerce")

    for feature, source in CATEGORICAL_FEATURES.items():
        model[feature] = normalize_categorical(df[source])

    protected = pd.DataFrame({"row_id": model["row_id"]})
    for feature, source in PROTECTED_ATTRIBUTES.items():
        protected[feature] = normalize_categorical(df[source])

    return model, protected


def split_indices(model: pd.DataFrame) -> dict[str, np.ndarray]:
    train_val_idx, test_idx = train_test_split(
        model.index.to_numpy(),
        test_size=0.20,
        stratify=model["y"],
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        stratify=model.loc[train_val_idx, "y"],
        random_state=RANDOM_STATE,
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def fit_transform_splits(
    model: pd.DataFrame,
    splits: dict[str, np.ndarray],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    train = model.loc[splits["train"]].copy()
    numeric_medians = {
        feature: float(train[feature].median()) for feature in NUMERIC_FEATURES
    }
    numeric_medians = {
        feature: (0.0 if not np.isfinite(value) else value)
        for feature, value in numeric_medians.items()
    }

    category_info: dict[str, dict[str, Any]] = {}
    for feature in CATEGORICAL_FEATURES:
        counts = train[feature].value_counts(dropna=False)
        limit = CATEGORY_LEVEL_LIMITS[feature]
        retained = [str(value) for value in counts.index[:limit]]
        uses_other = len(counts) > limit
        levels = retained + (["Other"] if uses_other else [])
        baseline = levels[0]
        encoded = [f"{feature}__{level}" for level in levels[1:]]
        category_info[feature] = {
            "source_column": CATEGORICAL_FEATURES[feature],
            "max_retained_levels": limit,
            "retained_levels": retained,
            "uses_other": uses_other,
            "baseline_level": baseline,
            "encoded_columns": encoded,
        }

    feature_columns = list(NUMERIC_FEATURES)
    for feature in CATEGORICAL_FEATURES:
        feature_columns.extend(category_info[feature]["encoded_columns"])

    processed: dict[str, pd.DataFrame] = {}
    for split_name, idx in splits.items():
        frame = model.loc[idx].copy().reset_index(drop=True)
        out = frame[["row_id", "y"]].copy()
        for feature in NUMERIC_FEATURES:
            out[feature] = frame[feature].fillna(numeric_medians[feature]).astype(float)
        for feature in CATEGORICAL_FEATURES:
            info = category_info[feature]
            retained = set(info["retained_levels"])
            values = frame[feature].where(frame[feature].isin(retained), "Other")
            values = pd.Categorical(values, categories=info["retained_levels"] + (["Other"] if info["uses_other"] else []))
            dummies = pd.get_dummies(values, prefix=feature, prefix_sep="__", dtype=float)
            for encoded_col in info["encoded_columns"]:
                out[encoded_col] = dummies.get(encoded_col, pd.Series(0.0, index=out.index)).astype(float)
        processed[split_name] = out[["row_id", "y"] + feature_columns]

    feature_dictionary = {
        "target": "y",
        "id_column": "row_id",
        "amendment": "phase17_amendment_04",
        "removed_features": REMOVED_FEATURES_AMENDMENT_04,
        "numeric_features": [
            {"feature": feature, "source_column": source, "train_median": numeric_medians[feature]}
            for feature, source in NUMERIC_FEATURES.items()
        ],
        "categorical_features": category_info,
        "feature_columns": feature_columns,
        "feature_count_after_one_hot": len(feature_columns),
        "protected_attributes_held_aside": PROTECTED_ATTRIBUTES,
        "random_state": RANDOM_STATE,
    }
    return processed, feature_dictionary


def assert_no_protected_leakage(frames: dict[str, pd.DataFrame]) -> None:
    for split_name, frame in frames.items():
        leaked = [
            col for col in frame.columns
            if any(token in col.lower() for token in PROTECTED_TOKENS)
        ]
        if leaked:
            raise ValueError(f"Protected attribute leakage in {split_name}: {leaked}")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_readme(
    *,
    raw_rows: int,
    property_rows: int,
    target_rows: int,
    property_note: str,
    base_rate: float,
    split_frames: dict[str, pd.DataFrame],
    feature_dictionary: dict[str, Any],
    checksums: dict[str, str],
) -> None:
    split_rows = [
        [name, str(len(frame)), f"{frame['y'].mean():.6f}"]
        for name, frame in split_frames.items()
    ]
    numeric_rows = [
        [item["feature"], item["source_column"], f"{item['train_median']:.6f}"]
        for item in feature_dictionary["numeric_features"]
    ]
    categorical_rows = [
        [
            feature,
            info["source_column"],
            "; ".join(info["retained_levels"]),
            "yes" if info["uses_other"] else "no",
            info["baseline_level"],
        ]
        for feature, info in feature_dictionary["categorical_features"].items()
    ]
    checksum_rows = [[name, checksum] for name, checksum in checksums.items()]
    feature_columns = "\n".join(f"- `{col}`" for col in feature_dictionary["feature_columns"])
    removed_features = "\n".join(f"- `{feature}`" for feature in feature_dictionary["removed_features"])

    text = (
        "# HMDA Boston-Cambridge-Newton 2018 Processed Splits\n\n"
        "Generated by `src/preprocessing_hmda.py` for Op4 Phase A under `phase17_amendment_04.md`.\n\n"
        "## Scope\n\n"
        f"- Raw rows after amended Boston geography union: {raw_rows}\n"
        f"- Rows after one-to-four-family property filter: {property_rows}\n"
        f"- Property filter implementation: {property_note}\n"
        f"- Rows after `action_taken in {{1,3}}`: {target_rows}\n"
        f"- Approval base rate after target collapse: {base_rate:.6f}\n"
        "- Target: `y=1` for originated loans (`action_taken=1`), `y=0` for denied loans (`action_taken=3`).\n"
        "- Protected attributes are written only to `protected_attributes.parquet` and are absent from train/val/test matrices.\n\n"
        "## Amendment 04 Feature Removal\n\n"
        "The following six post-decision or leakage-prone HMDA fields are removed before preprocessing:\n\n"
        f"{removed_features}\n\n"
        "## Split Counts\n\n"
        f"{markdown_table(['split', 'rows', 'approval_base_rate'], split_rows)}\n\n"
        "## Numeric Features\n\n"
        f"{markdown_table(['feature', 'source_column', 'train_median_imputation'], numeric_rows)}\n\n"
        "## Categorical Features\n\n"
        "Categorical features retain the most frequent training levels up to the locked per-field cap; less frequent levels map to `Other`. The most frequent retained level is the dummy-coding baseline.\n\n"
        f"{markdown_table(['feature', 'source_column', 'retained_levels', 'other_bucket', 'baseline_level'], categorical_rows)}\n\n"
        f"## Feature Columns ({feature_dictionary['feature_count_after_one_hot']})\n\n"
        f"{feature_columns}\n\n"
        "## SHA256\n\n"
        f"{markdown_table(['file', 'sha256'], checksum_rows)}\n"
    )
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    require_columns(
        pd.read_csv(RAW_PATH, nrows=0),
        list(NUMERIC_FEATURES.values())
        + list(CATEGORICAL_FEATURES.values())
        + list(PROTECTED_ATTRIBUTES.values())
        + ["action_taken"],
    )

    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw_rows = len(raw)
    property_df, property_note = filter_property_scope(raw)
    property_rows = len(property_df)
    target_df = property_df[property_df["action_taken"].isin([1, 3])].copy()
    target_rows = len(target_df)
    if target_rows == 0:
        raise ValueError("HMDA target filter produced zero rows.")

    model, protected = build_model_frame(target_df.reset_index(drop=True))
    base_rate = float(model["y"].mean())
    if not (0.4 <= base_rate <= 0.95):
        raise ValueError(f"Approval base rate {base_rate:.6f} outside [0.4, 0.95].")

    splits = split_indices(model)
    split_frames, feature_dictionary = fit_transform_splits(model, splits)
    assert_no_protected_leakage(split_frames)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_frames[split_name].to_parquet(OUT_DIR / f"{split_name}.parquet", index=False)

    protected_path = OUT_DIR / "protected_attributes.parquet"
    if protected_path.exists():
        existing_protected = pd.read_parquet(protected_path, columns=["row_id"])
        if existing_protected["row_id"].tolist() != protected["row_id"].tolist():
            raise ValueError("Existing protected_attributes.parquet row_id order does not match regenerated rows.")
    else:
        protected.to_parquet(protected_path, index=False)
    (OUT_DIR / "feature_dictionary.json").write_text(
        json.dumps(feature_dictionary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    checksums = {
        file_name: sha256_file(OUT_DIR / file_name)
        for file_name in [
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "protected_attributes.parquet",
            "feature_dictionary.json",
        ]
    }
    write_readme(
        raw_rows=raw_rows,
        property_rows=property_rows,
        target_rows=target_rows,
        property_note=property_note,
        base_rate=base_rate,
        split_frames=split_frames,
        feature_dictionary=feature_dictionary,
        checksums=checksums,
    )

    print(f"raw_rows={raw_rows}")
    print(f"property_rows={property_rows}")
    print(f"target_rows={target_rows}")
    print(f"base_rate={base_rate:.6f}")
    for split_name in ["train", "val", "test"]:
        frame = split_frames[split_name]
        print(f"{split_name}_rows={len(frame)}")
        print(f"{split_name}_base_rate={frame['y'].mean():.6f}")
    print(f"feature_count_after_one_hot={feature_dictionary['feature_count_after_one_hot']}")
    print("protected_attributes=separate")


if __name__ == "__main__":
    main()
