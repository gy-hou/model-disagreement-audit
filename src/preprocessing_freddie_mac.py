"""Preprocess Freddie Mac SFLLD sample files for v0.7.

The script creates one row per loan from origination features and labels from
monthly performance. It does not commit raw or processed loan-level data.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "freddie_mac_samples_2018_2025"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_SEED = 20260504
TARGET = "ever_60dpd_24m"
ID_COL = "row_id"

ORIG_COLUMNS = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer_flag",
    "maturity_date",
    "msa",
    "mi_pct",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "ppm_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_sequence_number",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "servicer_name",
    "super_conforming_flag",
    "pre_harp_loan_sequence_number",
    "program_indicator",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "mi_cancellation_indicator",
]

PERF_COLUMNS = [
    "loan_sequence_number",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_deferred_upb",
    "ddlpi",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance_and_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss_calculation",
    "modification_cost",
    "step_modification_flag",
    "deferred_payment_plan",
    "eltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
    "current_month_modification_cost",
    "interest_bearing_upb",
]

NUMERIC_FEATURES = [
    "credit_score",
    "msa",
    "mi_pct",
    "number_of_units",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
]

CATEGORICAL_FEATURES = [
    "first_time_homebuyer_flag",
    "occupancy_status",
    "channel",
    "ppm_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_purpose",
    "super_conforming_flag",
    "program_indicator",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
]

REMOVED_LABEL_LEAKAGE_FEATURES = [
    "servicer_name",
    "mi_cancellation_indicator",
]

REMOVED_DESIGN_FEATURES = [
    "seller_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--target-window-months", type=int, default=24)
    parser.add_argument("--target-dpd", type=int, choices=(30, 60, 90), default=60)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--seal", action="store_true")
    return parser.parse_args()


def delinquency_code(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    if text.upper() == "RA":
        return 9.0
    try:
        return float(int(text))
    except ValueError:
        return np.nan


def threshold_code(target_dpd: int) -> int:
    return {30: 1, 60: 2, 90: 3}[target_dpd]


def read_zip_member(path: Path, suffix: str) -> tuple[zipfile.ZipFile, str]:
    zf = zipfile.ZipFile(path)
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        zf.close()
        raise ValueError(f"Expected one member ending {suffix!r} in {path}; found {matches}")
    return zf, matches[0]


def load_origination(zip_path: Path, year: int) -> pd.DataFrame:
    zf, member = read_zip_member(zip_path, f"sample_orig_{year}.txt")
    with zf, zf.open(member) as handle:
        frame = pd.read_csv(
            handle,
            sep="|",
            header=None,
            names=ORIG_COLUMNS,
            dtype=str,
            na_values=["", " ", "NA"],
            keep_default_na=True,
        )
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["credit_score"] = frame["credit_score"].replace({9999: np.nan})
    frame["mi_pct"] = frame["mi_pct"].replace({999: np.nan})
    frame["number_of_borrowers"] = frame["number_of_borrowers"].replace({99: np.nan})
    frame["source_year"] = int(year)
    return frame[["loan_sequence_number", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "source_year"]]


def aggregate_performance(
    zip_path: Path,
    year: int,
    *,
    target_window_months: int,
    target_dpd: int,
    chunksize: int,
) -> pd.DataFrame:
    zf, member = read_zip_member(zip_path, f"sample_svcg_{year}.txt")
    parts: list[pd.DataFrame] = []
    cutoff = threshold_code(target_dpd)
    usecols = ["loan_sequence_number", "loan_age", "current_loan_delinquency_status"]
    with zf, zf.open(member) as handle:
        reader = pd.read_csv(
            handle,
            sep="|",
            header=None,
            names=PERF_COLUMNS,
            usecols=usecols,
            dtype=str,
            chunksize=chunksize,
            na_values=["", " ", "NA"],
            keep_default_na=True,
        )
        for chunk in reader:
            chunk["loan_age"] = pd.to_numeric(chunk["loan_age"], errors="coerce")
            within = chunk[chunk["loan_age"].between(0, target_window_months, inclusive="both")].copy()
            if within.empty:
                continue
            codes = within["current_loan_delinquency_status"].map(delinquency_code)
            within["target_hit"] = (codes >= cutoff).astype(int)
            parts.append(within.groupby("loan_sequence_number", as_index=False)["target_hit"].max())
    if not parts:
        raise ValueError(f"No performance rows found inside target window for {year}")
    out = pd.concat(parts, ignore_index=True).groupby("loan_sequence_number", as_index=False)["target_hit"].max()
    return out


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


def apply_imputation(frame: pd.DataFrame, imputation: dict[str, Any]) -> pd.DataFrame:
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
    out[TARGET] = out[TARGET].astype(int)
    return out[[ID_COL, TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]


def split_and_write(frame: pd.DataFrame, out_dir: Path, *, seed: int, seal: bool, metadata: dict[str, Any]) -> None:
    train_val, test = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame[TARGET],
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val[TARGET],
        random_state=seed,
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    imputation = fit_imputation(train)
    outputs = {
        "train": apply_imputation(train, imputation),
        "val": apply_imputation(val, imputation),
        "test": apply_imputation(test, imputation),
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
        "target": TARGET,
        "id_column": ID_COL,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "split_rows": {name: int(len(split)) for name, split in outputs.items()},
        "split_positive_rates": {name: float(split[TARGET].mean()) for name, split in outputs.items()},
        "sealed_test": bool(seal),
    }
    (out_dir / "README.md").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    zip_path = args.raw_dir / f"sample_{args.year}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    target_name = f"ever_{args.target_dpd}dpd_{args.target_window_months}m"
    global TARGET
    TARGET = target_name
    out_dir = args.out_dir or (PROCESSED_DIR / f"freddie_mac_{args.year}_{target_name}")
    orig = load_origination(zip_path, args.year)
    perf = aggregate_performance(
        zip_path,
        args.year,
        target_window_months=args.target_window_months,
        target_dpd=args.target_dpd,
        chunksize=args.chunksize,
    )
    merged = orig.merge(perf, on="loan_sequence_number", how="inner")
    if merged["target_hit"].nunique() < 2:
        raise ValueError(f"Target has <2 classes for {args.year}: {merged['target_hit'].value_counts().to_dict()}")
    merged = merged.rename(columns={"target_hit": TARGET}).reset_index(drop=True)
    merged.insert(0, ID_COL, np.arange(1, len(merged) + 1, dtype=int))
    metadata = {
        "dataset": f"freddie_mac_{args.year}",
        "source_zip": str(zip_path.relative_to(PROJECT_ROOT)),
        "source_rows_origination": int(len(orig)),
        "source_rows_labeled": int(len(merged)),
        "target_dpd": int(args.target_dpd),
        "target_window_months": int(args.target_window_months),
        "removed_label_leakage_features": REMOVED_LABEL_LEAKAGE_FEATURES,
        "removed_design_features": REMOVED_DESIGN_FEATURES,
    }
    split_and_write(merged, out_dir, seed=args.split_seed, seal=args.seal, metadata=metadata)
    feature_count = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    print(
        f"Wrote {out_dir.relative_to(PROJECT_ROOT)} rows={len(merged)} "
        f"features={feature_count} positive_rate={merged[TARGET].mean():.6f}"
    )


if __name__ == "__main__":
    main()
