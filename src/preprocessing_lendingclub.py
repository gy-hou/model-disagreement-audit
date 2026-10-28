from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "lendingclub" / "lc_2018q1_accepted.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "lendingclub"
RANDOM_STATE = 20260512
TARGET = "default"
ID_COL = "row_id"

DEFAULT_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}

NUMERIC_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tot_cur_bal",
    "total_rev_hi_lim",
]

CATEGORICAL_FEATURES = [
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "emp_length_binned",
]

RAW_COLUMNS = [
    "loan_status",
    "emp_length",
    *[col for col in NUMERIC_FEATURES if col != "term"],
    "term",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
]

FEATURE_DESCRIPTIONS = {
    "loan_amnt": "Listed loan amount requested by the borrower.",
    "term": "Loan term parsed to months.",
    "int_rate": "Interest rate at issuance.",
    "annual_inc": "Borrower annual income.",
    "dti": "Debt-to-income ratio.",
    "delinq_2yrs": "Delinquencies in the prior two years.",
    "fico_range_low": "Lower bound of FICO range at origination.",
    "fico_range_high": "Upper bound of FICO range at origination.",
    "open_acc": "Number of open credit lines.",
    "pub_rec": "Number of derogatory public records.",
    "revol_bal": "Total revolving balance.",
    "revol_util": "Revolving line utilization.",
    "total_acc": "Total number of credit lines.",
    "mort_acc": "Number of mortgage accounts.",
    "pub_rec_bankruptcies": "Number of public record bankruptcies.",
    "tot_cur_bal": "Total current balance of all accounts.",
    "total_rev_hi_lim": "Total revolving high credit limit.",
    "grade": "Lending Club assigned loan grade.",
    "home_ownership": "Borrower home-ownership category.",
    "verification_status": "Income verification status.",
    "purpose": "Borrower stated loan purpose.",
    "emp_length_binned": "Employment length binned to <1y, 1-3y, 4-7y, 8+y, or unknown.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_months(series: pd.Series) -> pd.Series:
    months = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(months, errors="coerce")


def parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace("%", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def bin_emp_length(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return "unknown"
    if text == "< 1 year":
        return "<1y"
    if text in {"1 year", "2 years", "3 years"}:
        return "1-3y"
    if text in {"4 years", "5 years", "6 years", "7 years"}:
        return "4-7y"
    if text in {"8 years", "9 years", "10+ years"}:
        return "8+y"
    raise ValueError(f"Unexpected emp_length value: {value!r}")


def load_and_transform(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, usecols=RAW_COLUMNS, low_memory=False)
    raw = raw[raw["loan_status"] != "Current"].copy()
    raw[TARGET] = raw["loan_status"].isin(DEFAULT_STATUSES).astype(int)

    frame = pd.DataFrame({ID_COL: np.arange(1, len(raw) + 1, dtype=int), TARGET: raw[TARGET].to_numpy()})
    for feature in NUMERIC_FEATURES:
        if feature == "term":
            frame[feature] = parse_months(raw[feature])
        else:
            frame[feature] = parse_numeric(raw[feature])

    for feature in CATEGORICAL_FEATURES:
        if feature == "emp_length_binned":
            frame[feature] = raw["emp_length"].map(bin_emp_length)
        else:
            values = raw[feature].astype("string").str.strip()
            frame[feature] = values.replace({"": pd.NA}).fillna("missing").astype(str)

    return frame[[ID_COL, TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_val, test = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame[TARGET],
        random_state=RANDOM_STATE,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val[TARGET],
        random_state=RANDOM_STATE,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def fit_imputation(train: pd.DataFrame) -> dict[str, Any]:
    medians = train[NUMERIC_FEATURES].median(numeric_only=True)
    missing_medians = [feature for feature, value in medians.items() if pd.isna(value)]
    if missing_medians:
        raise ValueError(f"Numeric features have undefined train medians: {missing_medians}")
    return {
        "numeric_medians": {feature: float(medians[feature]) for feature in NUMERIC_FEATURES},
        "categorical_fill": "missing",
        "emp_length_missing_fill": "unknown",
    }


def apply_imputation(frame: pd.DataFrame, imputation: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    medians = imputation["numeric_medians"]
    for feature in NUMERIC_FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").fillna(medians[feature]).astype(float)
    for feature in CATEGORICAL_FEATURES:
        fill_value = "unknown" if feature == "emp_length_binned" else "missing"
        values = out[feature].astype("string").str.strip()
        out[feature] = values.replace({"": pd.NA}).fillna(fill_value).astype(str)
    out[ID_COL] = out[ID_COL].astype(int)
    out[TARGET] = out[TARGET].astype(int)
    return out[[ID_COL, TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]


def one_hot_feature_count(train: pd.DataFrame) -> int:
    encoded = pd.get_dummies(
        train[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]],
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
    )
    return int(encoded.shape[1])


def split_summary(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "rows": int(len(frame)),
            "default_rate": float(frame[TARGET].mean()),
            "defaults": int(frame[TARGET].sum()),
        }
        for name, frame in splits.items()
    }


def write_readme(
    *,
    out_dir: Path,
    source_rows: int,
    splits: dict[str, pd.DataFrame],
    hashes: dict[str, str],
    imputation: dict[str, Any],
    one_hot_count: int,
) -> None:
    feature_rows = []
    for feature in NUMERIC_FEATURES:
        feature_rows.append(f"| `{feature}` | numeric | {FEATURE_DESCRIPTIONS[feature]} |")
    for feature in CATEGORICAL_FEATURES:
        feature_rows.append(f"| `{feature}` | categorical | {FEATURE_DESCRIPTIONS[feature]} |")

    summary = split_summary(splits)
    split_rows = [
        f"| `{name}` | {values['rows']} | {values['defaults']} | {values['default_rate']:.6f} | `{hashes[name]}` |"
        for name, values in summary.items()
    ]
    medians = "\n".join(
        f"- `{feature}`: `{value:.6g}`"
        for feature, value in imputation["numeric_medians"].items()
    )
    text = (
        "# Lending Club 2018 Q1 Processed Splits\n\n"
        "Generated by `src/preprocessing_lendingclub.py` for Op5 Phase E.\n\n"
        "## Source and Target\n\n"
        "- Raw subset: `data/raw/lendingclub/lc_2018q1_accepted.csv`\n"
        f"- Rows after locked `issue_d` and `loan_status != \"Current\"` filters: `{source_rows}`\n"
        "- Target: `default = 1` for `loan_status in {\"Charged Off\", \"Default\", \"Late (31-120 days)\"}`; otherwise `0`\n"
        f"- Split: 60/20/20 stratified by `default`, `random_state={RANDOM_STATE}`\n"
        "- Numeric imputation: median fit on train only and frozen for val/test\n"
        "- Categorical imputation: `missing`; `emp_length` NaN maps to `unknown` before split\n\n"
        "## Feature Dictionary\n\n"
        "| feature | type | description |\n"
        "|---|---|---|\n"
        + "\n".join(feature_rows)
        + "\n\n"
        "Employment-length bins:\n\n"
        "- `< 1 year` -> `<1y`\n"
        "- `1 year`, `2 years`, `3 years` -> `1-3y`\n"
        "- `4 years`, `5 years`, `6 years`, `7 years` -> `4-7y`\n"
        "- `8 years`, `9 years`, `10+ years` -> `8+y`\n"
        "- missing -> `unknown`\n\n"
        "## Split Summary\n\n"
        "| split | rows | defaults | default_rate | SHA256 |\n"
        "|---|---:|---:|---:|---|\n"
        + "\n".join(split_rows)
        + "\n\n"
        f"Base feature count: `{len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}`\n\n"
        f"One-hot feature count from train categories: `{one_hot_count}`\n\n"
        "## Train-Fit Numeric Medians\n\n"
        f"{medians}\n"
    )
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def preprocess(raw_path: Path = RAW_PATH, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_and_transform(raw_path)
    splits = split_frame(frame)
    imputation = fit_imputation(splits["train"])
    splits = {name: apply_imputation(split, imputation) for name, split in splits.items()}

    for name, split in splits.items():
        split.to_parquet(out_dir / f"{name}.parquet", index=False)

    hashes = {name: sha256_file(out_dir / f"{name}.parquet") for name in splits}
    one_hot_count = one_hot_feature_count(splits["train"])
    write_readme(
        out_dir=out_dir,
        source_rows=len(frame),
        splits=splits,
        hashes=hashes,
        imputation=imputation,
        one_hot_count=one_hot_count,
    )
    (out_dir / "imputation_values.json").write_text(
        json.dumps(imputation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "source_rows": int(len(frame)),
        "split_summary": split_summary(splits),
        "hashes": hashes,
        "one_hot_feature_count": one_hot_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    summary = preprocess(args.raw_path, args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
