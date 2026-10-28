"""Preprocess Fannie Mae Single-Family Loan Performance primary files for v0.7."""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fannie_mae_primary_2018"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LAYOUT_PATH = RAW_DIR / "crt-file-layout-and-glossary.xlsx"
SPLIT_SEED = 20260504
SUBSAMPLE_N = 50_000
SUBSAMPLE_SEED = 20260801
ID_COL = "row_id"
TARGET = "ever_60dpd_24m"

RAW_FEATURES = [
    "channel",
    "original_interest_rate",
    "original_upb",
    "original_loan_term",
    "origination_date",
    "first_payment_date",
    "original_ltv",
    "original_cltv",
    "number_of_borrowers",
    "dti",
    "borrower_credit_score",
    "co_borrower_credit_score",
    "first_time_home_buyer",
    "loan_purpose",
    "property_type",
    "number_of_units",
    "occupancy_status",
    "property_state",
    "msa",
    "zip_code_short",
    "mi_pct",
    "amortization_type",
    "interest_only_indicator",
    "mi_type",
    "relocation_mortgage_indicator",
    "property_valuation_method",
    "high_balance_loan_indicator",
]

NUMERIC_FEATURES = [
    "original_interest_rate",
    "original_upb",
    "original_loan_term",
    "original_ltv",
    "original_cltv",
    "number_of_borrowers",
    "dti",
    "borrower_credit_score",
    "co_borrower_credit_score",
    "number_of_units",
    "msa",
    "mi_pct",
]

CATEGORICAL_FEATURES = [feature for feature in RAW_FEATURES if feature not in NUMERIC_FEATURES]

REMOVED_LABEL_LEAKAGE_FEATURES = [
    "servicer_name",
]

REMOVED_DESIGN_FEATURES = [
    "seller_name",
]

FIELD_RENAMES = {
    "LOAN_IDENTIFIER": "loan_id",
    "MONTHLY_REPORTING_PERIOD": "monthly_reporting_period",
    "CHANNEL": "channel",
    "ORIGINAL_INTEREST_RATE": "original_interest_rate",
    "ORIGINAL_UPB": "original_upb",
    "ORIGINAL_LOAN_TERM": "original_loan_term",
    "ORIGINATION_DATE": "origination_date",
    "FIRST_PAYMENT_DATE": "first_payment_date",
    "LOAN_AGE": "loan_age",
    "ORIGINAL_LOAN_TO_VALUE_RATIO_LTV": "original_ltv",
    "ORIGINAL_COMBINED_LOAN_TO_VALUE_RATIO_CLTV": "original_cltv",
    "NUMBER_OF_BORROWERS": "number_of_borrowers",
    "DEBT_TO_INCOME_DTI": "dti",
    "BORROWER_CREDIT_SCORE_AT_ORIGINATION": "borrower_credit_score",
    "CO_BORROWER_CREDIT_SCORE_AT_ORIGINATION": "co_borrower_credit_score",
    "FIRST_TIME_HOME_BUYER_INDICATOR": "first_time_home_buyer",
    "LOAN_PURPOSE": "loan_purpose",
    "PROPERTY_TYPE": "property_type",
    "NUMBER_OF_UNITS": "number_of_units",
    "OCCUPANCY_STATUS": "occupancy_status",
    "PROPERTY_STATE": "property_state",
    "METROPOLITAN_STATISTICAL_AREA_MSA_OR_METROPOLITAN_STATISTICAL_DIVISION_AREA_MSDA": "msa",
    "ZIP_CODE_SHORT": "zip_code_short",
    "MORTGAGE_INSURANCE_PERCENTAGE": "mi_pct",
    "AMORTIZATION_TYPE": "amortization_type",
    "INTEREST_ONLY_LOAN_INDICATOR": "interest_only_indicator",
    "CURRENT_LOAN_DELINQUENCY_STATUS": "dlq_status",
    "MORTGAGE_INSURANCE_TYPE": "mi_type",
    "RELOCATION_MORTGAGE_INDICATOR": "relocation_mortgage_indicator",
    "PROPERTY_VALUATION_METHOD": "property_valuation_method",
    "HIGH_BALANCE_LOAN_INDICATOR": "high_balance_loan_indicator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--layout", type=Path, default=LAYOUT_PATH)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "fannie_mae_2018_ever_60dpd_24m")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--target-window-months", type=int, default=24)
    parser.add_argument("--target-dpd", type=int, choices=(30, 60, 90), default=60)
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument("--subsample-n", type=int, default=SUBSAMPLE_N)
    parser.add_argument("--subsample-seed", type=int, default=SUBSAMPLE_SEED)
    parser.add_argument("--seal", action="store_true")
    return parser.parse_args()


def slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_").upper()


def load_layout_columns(path: Path) -> list[str]:
    layout = pd.read_excel(path, sheet_name="Combined Glossary")
    layout = layout.sort_values("Field Position")
    columns = [slug(value) for value in layout["Field Name"]]
    if len(columns) != 113:
        raise ValueError(f"Expected 113 Fannie columns; got {len(columns)} from {path}")
    if len(set(columns)) != len(columns):
        duplicates = sorted({col for col in columns if columns.count(col) > 1})
        raise ValueError(f"Duplicate sanitized column names: {duplicates}")
    return columns


def delinquency_code(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text in {"", "XX", "NULL"}:
        return np.nan
    if text == "RA":
        return 9.0
    try:
        return float(int(text))
    except ValueError:
        return np.nan


def threshold_code(target_dpd: int) -> int:
    return {30: 1, 60: 2, 90: 3}[target_dpd]


def first_rows_by_loan(chunk: pd.DataFrame) -> pd.DataFrame:
    ordered = chunk.copy()
    ordered["_loan_age_sort"] = ordered["loan_age"].fillna(np.inf)
    ordered["_row_order"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(
        ["loan_id", "_loan_age_sort", "_row_order"],
        kind="mergesort",
    )
    return ordered.drop_duplicates("loan_id", keep="first")[
        ["loan_id", *RAW_FEATURES, "loan_age"]
    ].copy()


def process_zip(
    zip_path: Path,
    columns: list[str],
    *,
    target_window_months: int,
    target_dpd: int,
    chunksize: int,
) -> pd.DataFrame:
    use_raw = list(FIELD_RENAMES)
    cutoff = threshold_code(target_dpd)
    acquisition_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        if len(zf.namelist()) != 1:
            raise ValueError(f"Expected one member in {zip_path}; found {zf.namelist()}")
        with zf.open(zf.namelist()[0]) as handle:
            reader = pd.read_csv(
                handle,
                sep="|",
                header=None,
                names=columns,
                usecols=use_raw,
                dtype=str,
                chunksize=chunksize,
                na_values=["", " ", "NULL"],
                keep_default_na=True,
            )
            for chunk in reader:
                chunk = chunk.rename(columns=FIELD_RENAMES)
                chunk["loan_age"] = pd.to_numeric(chunk["loan_age"], errors="coerce")
                acquisition_parts.append(first_rows_by_loan(chunk))
                within = chunk[chunk["loan_age"].between(0, target_window_months, inclusive="both")].copy()
                if not within.empty:
                    within["target_hit"] = (within["dlq_status"].map(delinquency_code) >= cutoff).astype(int)
                    target_parts.append(within.groupby("loan_id", as_index=False)["target_hit"].max())
    if not acquisition_parts or not target_parts:
        raise ValueError(f"No usable rows found in {zip_path}")
    acquisition = pd.concat(acquisition_parts, ignore_index=True)
    acquisition = first_rows_by_loan(acquisition)
    target = pd.concat(target_parts, ignore_index=True).groupby("loan_id", as_index=False)["target_hit"].max()
    return acquisition.drop(columns=["loan_age"]).merge(target, on="loan_id", how="inner")


def load_dataset(
    raw_dir: Path,
    columns: list[str],
    *,
    target_window_months: int,
    target_dpd: int,
    chunksize: int,
) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("2018Q*.zip"))
    if len(paths) != 4:
        raise FileNotFoundError(f"Expected 4 Fannie 2018 quarterly zips in {raw_dir}; found {paths}")
    frames = [
        process_zip(
            path,
            columns,
            target_window_months=target_window_months,
            target_dpd=target_dpd,
            chunksize=chunksize,
        )
        for path in paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates("loan_id", keep="first").reset_index(drop=True)


def stratified_subsample(
    frame: pd.DataFrame,
    *,
    target: str,
    n: int | None,
    seed: int,
) -> pd.DataFrame:
    if n is None or n <= 0 or n >= len(frame):
        return frame.reset_index(drop=True)
    _, sample = train_test_split(
        frame,
        test_size=int(n),
        stratify=frame[target],
        random_state=seed,
    )
    return sample.reset_index(drop=True)


def fit_imputation(train: pd.DataFrame) -> dict[str, Any]:
    medians = train[NUMERIC_FEATURES].median(numeric_only=True)
    modes: dict[str, str] = {}
    for feature in CATEGORICAL_FEATURES:
        values = train[feature].astype("string").str.strip().replace({"": pd.NA})
        mode = values.dropna().mode()
        modes[feature] = str(mode.iloc[0]) if not mode.empty else "missing"
    return {
        "numeric_medians": {
            feature: (0.0 if pd.isna(medians[feature]) else float(medians[feature]))
            for feature in NUMERIC_FEATURES
        },
        "categorical_modes": modes,
    }


def apply_imputation(frame: pd.DataFrame, imputation: dict[str, Any], target: str) -> pd.DataFrame:
    out = frame.copy()
    for feature in NUMERIC_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").fillna(
            imputation["numeric_medians"][feature]
        ).astype(float)
    for feature in CATEGORICAL_FEATURES:
        out[feature] = (
            out[feature]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA})
            .fillna(imputation["categorical_modes"][feature])
            .astype(str)
        )
    out[ID_COL] = out[ID_COL].astype(int)
    out[target] = out[target].astype(int)
    return out[[ID_COL, target, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]


def split_and_write(frame: pd.DataFrame, out_dir: Path, *, target: str, seed: int, seal: bool, metadata: dict[str, Any]) -> None:
    train_val, test = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame[target],
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val[target],
        random_state=seed,
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    imputation = fit_imputation(train)
    outputs = {
        "train": apply_imputation(train, imputation, target),
        "val": apply_imputation(val, imputation, target),
        "test": apply_imputation(test, imputation, target),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in outputs.items():
        path = out_dir / f"{name}.parquet"
        split.to_parquet(path, index=False)
        if seal and name == "test":
            os.chmod(path, 0o444)
    (out_dir / "imputation_values.json").write_text(json.dumps(imputation, indent=2), encoding="utf-8")
    summary = {
        **metadata,
        "split_seed": seed,
        "target": target,
        "id_column": ID_COL,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "split_rows": {name: int(len(split)) for name, split in outputs.items()},
        "split_positive_rates": {name: float(split[target].mean()) for name, split in outputs.items()},
        "sealed_test": bool(seal),
    }
    (out_dir / "README.md").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    target = f"ever_{args.target_dpd}dpd_{args.target_window_months}m"
    columns = load_layout_columns(args.layout)
    frame = load_dataset(
        args.raw_dir,
        columns,
        target_window_months=args.target_window_months,
        target_dpd=args.target_dpd,
        chunksize=args.chunksize,
    ).rename(columns={"target_hit": target})
    if frame[target].nunique() < 2:
        raise ValueError(f"Target has <2 classes: {frame[target].value_counts().to_dict()}")
    full_rows = int(len(frame))
    full_positive_rate = float(frame[target].mean())
    frame = stratified_subsample(
        frame,
        target=target,
        n=args.subsample_n,
        seed=args.subsample_seed,
    )
    for feature in NUMERIC_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame.insert(0, ID_COL, np.arange(1, len(frame) + 1, dtype=int))
    metadata = {
        "dataset": "fannie_mae_2018",
        "source_quarters": [path.name for path in sorted(args.raw_dir.glob("2018Q*.zip"))],
        "source_rows_labeled": full_rows,
        "source_positive_rate": full_positive_rate,
        "subsample_rows": int(len(frame)),
        "subsample_positive_rate": float(frame[target].mean()),
        "subsample_seed": int(args.subsample_seed),
        "subsample_stratified_on": target,
        "target_dpd": int(args.target_dpd),
        "target_window_months": int(args.target_window_months),
        "removed_label_leakage_features": REMOVED_LABEL_LEAKAGE_FEATURES,
        "removed_design_features": REMOVED_DESIGN_FEATURES,
    }
    out_dir = args.out_dir
    if out_dir == PROCESSED_DIR / "fannie_mae_2018_ever_60dpd_24m" and target != TARGET:
        out_dir = PROCESSED_DIR / f"fannie_mae_2018_{target}"
    split_and_write(frame, out_dir, target=target, seed=args.split_seed, seal=args.seal, metadata=metadata)
    feature_count = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    print(
        f"Wrote {out_dir.relative_to(PROJECT_ROOT)} rows={len(frame)} "
        f"features={feature_count} "
        f"positive_rate={frame[target].mean():.6f} "
        f"full_rows={full_rows} full_positive_rate={full_positive_rate:.6f}"
    )


if __name__ == "__main__":
    main()
