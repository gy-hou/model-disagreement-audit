from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

UCI_XLS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
RAW_XLS_PATH = RAW_DIR / "default_of_credit_card_clients.xls"

V0_4_DATASET = os.environ.get("V0_4_DATASET", "uci").lower()
UCI_TARGET = "default_payment_next_month"
FICO_TARGET = "is_at_risk"
SYNTHETIC_TARGET = "y"
SYNTHETIC_V2_TARGET = "y"
SYNTHETIC_STRONG_TARGET = "y"
LENDINGCLUB_TARGET = "default"
HMDA_TARGET = "y"
GMC_TARGET = "SeriousDlqin2yrs"
HOME_CREDIT_TARGET = "TARGET"
HMEQ_TARGET = "BAD"

CANONICAL_COLUMNS = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": UCI_TARGET,
    "default payment next month": UCI_TARGET,
    "default.payment.next.month": UCI_TARGET,
}

UCI_CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
UCI_PAY_STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
LENDINGCLUB_CATEGORICAL_FEATURES = [
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "emp_length_binned",
]
LENDINGCLUB_NUMERIC_FEATURES = [
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
LENDINGCLUB_FEATURES = LENDINGCLUB_NUMERIC_FEATURES + LENDINGCLUB_CATEGORICAL_FEATURES
HMDA_CATEGORICAL_FEATURES: list[str] = []
GMC_CATEGORICAL_FEATURES: list[str] = []
HOME_CREDIT_CATEGORICAL_FEATURES = [
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
]
HMEQ_CATEGORICAL_FEATURES = ["REASON", "JOB"]

# n_g accrual step registered in preregistration_v0_7.md §4/§5; execution governed
# by docs/specs/rho_g_accrual_execution_v0_7.md. Categorical membership is taken
# from the UCI source variable metadata, not from pandas dtype — Australian Credit
# ships its eight categorical attributes pre-encoded as integers.
GERMAN_CREDIT_TARGET = "y"
GERMAN_CREDIT_CATEGORICAL_FEATURES = [
    "Attribute1", "Attribute3", "Attribute4", "Attribute6", "Attribute7",
    "Attribute9", "Attribute10", "Attribute12", "Attribute14", "Attribute15",
    "Attribute17", "Attribute19", "Attribute20",
]
AUSTRALIAN_CREDIT_TARGET = "y"
AUSTRALIAN_CREDIT_CATEGORICAL_FEATURES = [
    "A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12",
]

GMC_RAW_PATH = RAW_DIR / "gmc" / "cs-training.csv"
HOME_CREDIT_RAW_PATH = RAW_DIR / "home_credit" / "application_train.csv"
HMEQ_RAW_PATH = RAW_DIR / "hmeq" / "hmeq.csv"
NEW_SPLIT_RANDOM_STATE = 20260504
SYNTHETIC_V2_RANDOM_STATE = 20260524
SYNTHETIC_STRONG_RANDOM_STATE = 20260525

if V0_4_DATASET == "fico":
    TARGET = FICO_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES: list[str] = []
    PAY_STATUS_FEATURES: list[str] = []
elif V0_4_DATASET == "synthetic":
    TARGET = SYNTHETIC_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES = []
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET == "synthetic_v2":
    TARGET = SYNTHETIC_V2_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES = []
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET == "synthetic_strong":
    TARGET = SYNTHETIC_STRONG_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES = []
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET == "lendingclub":
    TARGET = LENDINGCLUB_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES = LENDINGCLUB_CATEGORICAL_FEATURES
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET == "hmda":
    TARGET = HMDA_TARGET
    ID_COL = "row_id"
    # HMDA Phase A split files are already median-imputed and dummy-coded.
    CATEGORICAL_FEATURES = HMDA_CATEGORICAL_FEATURES
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET == "gmc":
    TARGET = GMC_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES = GMC_CATEGORICAL_FEATURES
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET in {"home_credit", "home_credit_no_ext", "home_credit_h2"}:
    TARGET = HOME_CREDIT_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES = HOME_CREDIT_CATEGORICAL_FEATURES
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET == "hmeq":
    TARGET = HMEQ_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES = HMEQ_CATEGORICAL_FEATURES
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET == "german_credit":
    TARGET = GERMAN_CREDIT_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES = GERMAN_CREDIT_CATEGORICAL_FEATURES
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET == "australian_credit":
    TARGET = AUSTRALIAN_CREDIT_TARGET
    ID_COL = "row_id"
    CATEGORICAL_FEATURES = AUSTRALIAN_CREDIT_CATEGORICAL_FEATURES
    PAY_STATUS_FEATURES = []
elif V0_4_DATASET in {"freddie_mac_2018", "fannie_mae_2018"}:
    # Registered as the n_g = 12 accrual step. The GSE deployment pipeline already
    # produced sealed splits, committees, and per-instance RC for these two; only
    # the Layer-2 obs_score was never computed. Categorical membership is imported
    # from the dataset's own preprocessing module rather than restated here.
    TARGET = "ever_60dpd_24m"
    ID_COL = "row_id"
    if V0_4_DATASET == "freddie_mac_2018":
        from preprocessing_freddie_mac import (
            CATEGORICAL_FEATURES as _GSE_CATEGORICAL_FEATURES,
        )
    else:
        from preprocessing_fannie_mae import (
            CATEGORICAL_FEATURES as _GSE_CATEGORICAL_FEATURES,
        )
    CATEGORICAL_FEATURES = list(_GSE_CATEGORICAL_FEATURES)
    PAY_STATUS_FEATURES = []
else:
    TARGET = UCI_TARGET
    ID_COL = "ID"
    CATEGORICAL_FEATURES = UCI_CATEGORICAL_FEATURES
    PAY_STATUS_FEATURES = UCI_PAY_STATUS_FEATURES


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    categorical_features: list[str]
    numeric_features: list[str]


def ensure_raw_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_XLS_PATH.exists():
        urlretrieve(UCI_XLS_URL, RAW_XLS_PATH)
    return RAW_XLS_PATH


def load_uci_default(raw_path: Path | None = None) -> pd.DataFrame:
    path = raw_path or ensure_raw_dataset()
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={col: CANONICAL_COLUMNS.get(str(col), str(col)) for col in df.columns})

    if TARGET not in df.columns:
        raise ValueError(f"Target column not found. Columns: {list(df.columns)}")

    df = df.dropna(how="all").copy()
    df.columns = [str(col).strip() for col in df.columns]

    # UCI encodes undocumented education/marriage values. Collapse them into
    # the broad "other/unknown" buckets used in many benchmark baselines.
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    df[TARGET] = df[TARGET].astype(int)

    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(int)
    else:
        df.insert(0, ID_COL, np.arange(1, len(df) + 1))

    return df


def load_lendingclub_processed(processed_dir: Path | None = None) -> DatasetBundle:
    path = processed_dir or (PROCESSED_DIR / "lendingclub")
    train = pd.read_parquet(path / "train.parquet")
    validation = pd.read_parquet(path / "val.parquet")
    test = pd.read_parquet(path / "test.parquet")
    return DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        feature_columns=LENDINGCLUB_FEATURES.copy(),
        categorical_features=LENDINGCLUB_CATEGORICAL_FEATURES.copy(),
        numeric_features=LENDINGCLUB_NUMERIC_FEATURES.copy(),
    )


def split_dataset(df: pd.DataFrame, random_state: int = 20260504) -> DatasetBundle:
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df[TARGET],
        random_state=random_state,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val[TARGET],
        random_state=random_state,
    )

    feature_columns = [col for col in df.columns if col not in {ID_COL, TARGET}]
    numeric_features = [col for col in feature_columns if col not in CATEGORICAL_FEATURES]

    return DatasetBundle(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        feature_columns=feature_columns,
        categorical_features=CATEGORICAL_FEATURES.copy(),
        numeric_features=numeric_features,
    )


def one_hot_tree_frame(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    if not CATEGORICAL_FEATURES:
        x = frame[list(feature_columns)].copy()
        if expected_columns is not None:
            x = x.reindex(columns=expected_columns, fill_value=0)
        return x.astype(float)
    x = pd.get_dummies(frame[list(feature_columns)], columns=CATEGORICAL_FEATURES, drop_first=False)
    if expected_columns is not None:
        x = x.reindex(columns=expected_columns, fill_value=0)
    return x.astype(float)


def write_processed_data(bundle: DatasetBundle) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ]:
        split_df.to_csv(PROCESSED_DIR / f"uci_{split_name}.csv", index=False)

    split_assignments = pd.concat(
        [
            bundle.train[[ID_COL, TARGET]].assign(split="train"),
            bundle.validation[[ID_COL, TARGET]].assign(split="validation"),
            bundle.test[[ID_COL, TARGET]].assign(split="test"),
        ],
        ignore_index=True,
    )
    split_assignments.to_csv(PROCESSED_DIR / "uci_split_assignments.csv", index=False)


def dataset_quality_summary(df: pd.DataFrame, bundle: DatasetBundle) -> dict[str, object]:
    duplicate_ids = int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else 0
    missing_cells = int(df.isna().sum().sum())
    target_rate = float(df[TARGET].mean())

    split_rows = {
        "train": int(len(bundle.train)),
        "validation": int(len(bundle.validation)),
        "test": int(len(bundle.test)),
    }
    split_default_rates = {
        "train": float(bundle.train[TARGET].mean()),
        "validation": float(bundle.validation[TARGET].mean()),
        "test": float(bundle.test[TARGET].mean()),
    }

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "features": int(len(bundle.feature_columns)),
        "target": TARGET,
        "overall_default_rate": target_rate,
        "duplicate_ids": duplicate_ids,
        "missing_cells": missing_cells,
        "split_rows": split_rows,
        "split_default_rates": split_default_rates,
        "categorical_features": bundle.categorical_features,
        "numeric_features": bundle.numeric_features,
    }


def _split_frame_for_target(
    df: pd.DataFrame,
    *,
    target: str,
    id_col: str = "row_id",
    categorical_features: list[str] | None = None,
    random_state: int = NEW_SPLIT_RANDOM_STATE,
) -> DatasetBundle:
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df[target],
        random_state=random_state,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val[target],
        random_state=random_state,
    )
    categorical = categorical_features or []
    feature_columns = [col for col in df.columns if col not in {id_col, target}]
    numeric_features = [col for col in feature_columns if col not in categorical]
    return DatasetBundle(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        feature_columns=feature_columns,
        categorical_features=categorical.copy(),
        numeric_features=numeric_features,
    )


def _fit_imputation(bundle: DatasetBundle) -> dict[str, object]:
    medians = bundle.train[bundle.numeric_features].median(numeric_only=True)
    modes: dict[str, str] = {}
    for feature in bundle.categorical_features:
        values = bundle.train[feature].astype("string").str.strip().replace({"": pd.NA})
        mode = values.dropna().mode()
        modes[feature] = str(mode.iloc[0]) if not mode.empty else "missing"
    return {
        "numeric_medians": {
            feature: (0.0 if pd.isna(medians[feature]) else float(medians[feature]))
            for feature in bundle.numeric_features
        },
        "categorical_modes": modes,
    }


def _apply_imputation(
    frame: pd.DataFrame,
    *,
    id_col: str,
    target: str,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    imputation: dict[str, object],
) -> pd.DataFrame:
    out = frame.copy()
    medians = imputation["numeric_medians"]
    modes = imputation["categorical_modes"]
    for feature in numeric_features:
        out[feature] = (
            pd.to_numeric(out[feature], errors="coerce")
            .fillna(medians[feature])
            .astype(float)
        )
    for feature in categorical_features:
        out[feature] = (
            out[feature]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA})
            .fillna(modes[feature])
            .astype(str)
        )
    out[id_col] = out[id_col].astype(int)
    out[target] = out[target].astype(int)
    return out[[id_col, target, *feature_columns]]


def _impute_bundle(bundle: DatasetBundle, *, id_col: str, target: str) -> DatasetBundle:
    imputation = _fit_imputation(bundle)
    frames = {
        name: _apply_imputation(
            frame,
            id_col=id_col,
            target=target,
            feature_columns=bundle.feature_columns,
            numeric_features=bundle.numeric_features,
            categorical_features=bundle.categorical_features,
            imputation=imputation,
        )
        for name, frame in [
            ("train", bundle.train),
            ("validation", bundle.validation),
            ("test", bundle.test),
        ]
    }
    return DatasetBundle(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        feature_columns=bundle.feature_columns,
        categorical_features=bundle.categorical_features,
        numeric_features=bundle.numeric_features,
    )


def _write_new_processed_splits(bundle: DatasetBundle, *, slug: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": PROCESSED_DIR / f"{slug}_train.csv",
        "validation": PROCESSED_DIR / f"{slug}_validation.csv",
        "test": PROCESSED_DIR / f"{slug}_test.csv",
    }
    if paths["test"].exists():
        raise FileExistsError(f"Refusing to reseal existing test split: {paths['test']}")
    for split_name, split_df in [
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ]:
        split_df.to_csv(paths[split_name], index=False)
    os.chmod(paths["test"], 0o444)


def make_gmc(raw_path: Path = GMC_RAW_PATH) -> DatasetBundle:
    raw = pd.read_csv(raw_path)
    unnamed = [col for col in raw.columns if str(col).startswith("Unnamed")]
    df = raw.drop(columns=unnamed).copy()
    df["_raw_order"] = np.arange(len(df), dtype=int)
    df = df.sort_values("_raw_order", kind="mergesort").drop(columns="_raw_order").reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=int))
    df[GMC_TARGET] = df[GMC_TARGET].astype(int)
    feature_columns = [col for col in df.columns if col not in {"row_id", GMC_TARGET}]
    for feature in feature_columns:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    bundle = _split_frame_for_target(
        df,
        target=GMC_TARGET,
        categorical_features=GMC_CATEGORICAL_FEATURES,
    )
    bundle = _impute_bundle(bundle, id_col="row_id", target=GMC_TARGET)
    _write_new_processed_splits(bundle, slug="gmc")
    return bundle


def make_home_credit(raw_path: Path = HOME_CREDIT_RAW_PATH) -> DatasetBundle:
    raw = pd.read_csv(raw_path)
    missing_rate = raw.isna().mean()
    keep_columns = [col for col in raw.columns if col == HOME_CREDIT_TARGET or missing_rate[col] <= 0.50]
    df = raw[keep_columns].copy()
    df["_raw_order"] = np.arange(len(df), dtype=int)
    df = df.sort_values("_raw_order", kind="mergesort").drop(columns="_raw_order").reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=int))
    df[HOME_CREDIT_TARGET] = df[HOME_CREDIT_TARGET].astype(int)
    categorical = [
        col
        for col in HOME_CREDIT_CATEGORICAL_FEATURES
        if col in df.columns
    ]
    categorical_set = set(categorical)
    categorical.extend(
        col
        for col in df.columns
        if col not in {"row_id", HOME_CREDIT_TARGET, *categorical_set}
        and (is_object_dtype(df[col]) or is_string_dtype(df[col]))
    )
    categorical_set = set(categorical)
    for feature in df.columns:
        if feature not in {"row_id", HOME_CREDIT_TARGET, *categorical_set}:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
    bundle = _split_frame_for_target(
        df,
        target=HOME_CREDIT_TARGET,
        categorical_features=categorical,
    )
    bundle = _impute_bundle(bundle, id_col="row_id", target=HOME_CREDIT_TARGET)
    _write_new_processed_splits(bundle, slug="home_credit")
    return bundle


def make_hmeq(raw_path: Path = HMEQ_RAW_PATH) -> DatasetBundle:
    raw = pd.read_csv(raw_path)
    df = raw.copy()
    df["_raw_order"] = np.arange(len(df), dtype=int)
    df = df.sort_values("_raw_order", kind="mergesort").drop(columns="_raw_order").reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=int))
    df[HMEQ_TARGET] = df[HMEQ_TARGET].astype(int)
    for feature in df.columns:
        if feature not in {"row_id", HMEQ_TARGET, *HMEQ_CATEGORICAL_FEATURES}:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
    bundle = _split_frame_for_target(
        df,
        target=HMEQ_TARGET,
        categorical_features=HMEQ_CATEGORICAL_FEATURES,
    )
    bundle = _impute_bundle(bundle, id_col="row_id", target=HMEQ_TARGET)
    _write_new_processed_splits(bundle, slug="hmeq")
    return bundle


def make_synthetic_v2() -> DatasetBundle:
    rng = np.random.default_rng(SYNTHETIC_V2_RANDOM_STATE)
    n = 10_000
    p = 20
    x = rng.standard_normal(size=(n, p))
    informative = [f"x_lin_{i + 1}" for i in range(10)]
    noise = [f"x_noise_{i + 1}" for i in range(10)]
    df = pd.DataFrame(x, columns=informative + noise)

    linear_score = (
        1.0 * df["x_lin_1"]
        - 0.9 * df["x_lin_2"]
        + 0.75 * df["x_lin_3"]
        - 0.65 * df["x_lin_4"]
        + 0.55 * df["x_lin_5"]
    )
    interaction_score = (
        1.2 * df["x_lin_1"] * df["x_lin_6"]
        - 1.0 * df["x_lin_2"] * df["x_lin_7"]
        + 0.9 * df["x_lin_3"] * df["x_lin_8"]
        + 0.8 * ((df["x_lin_9"] > 0) & (df["x_lin_10"] < 0)).astype(float)
        - 0.7 * ((df["x_lin_6"] * df["x_lin_7"]) > 0.75).astype(float)
    )
    base_score = 1.2 * (linear_score + interaction_score)
    lo, hi = -20.0, 20.0
    for _ in range(100):
        intercept = (lo + hi) / 2.0
        prob = 1.0 / (1.0 + np.exp(-(base_score + intercept)))
        if float(prob.mean()) > 0.30:
            hi = intercept
        else:
            lo = intercept
    intercept = (lo + hi) / 2.0
    prob = 1.0 / (1.0 + np.exp(-(base_score + intercept)))
    df[SYNTHETIC_V2_TARGET] = rng.binomial(1, prob).astype(int)
    df.insert(0, "row_id", np.arange(n, dtype=int))

    bundle = _split_frame_for_target(
        df,
        target=SYNTHETIC_V2_TARGET,
        categorical_features=[],
    )
    _write_new_processed_splits(bundle, slug="synthetic_v2")
    return bundle


def make_synthetic_strong() -> DatasetBundle:
    rng = np.random.default_rng(SYNTHETIC_STRONG_RANDOM_STATE)
    n = 12_000
    x_lin = rng.standard_normal(size=(n, 20))
    x_add = rng.standard_normal(size=(n, 20))
    x_int = rng.standard_normal(size=(n, 20))
    x_sparse = rng.binomial(1, 0.08, size=(n, 20)).astype(float)
    x_noise = rng.standard_normal(size=(n, 40))

    columns = (
        [f"x_lin_{i + 1}" for i in range(20)]
        + [f"x_add_{i + 1}" for i in range(20)]
        + [f"x_int_{i + 1}" for i in range(20)]
        + [f"x_sparse_{i + 1}" for i in range(20)]
        + [f"x_noise_{i + 1}" for i in range(40)]
    )
    x = np.column_stack([x_lin, x_add, x_int, x_sparse, x_noise])
    df = pd.DataFrame(x, columns=columns)

    beta = np.array([1.1, -1.0, 0.9, -0.85, 0.75, -0.7, 0.6, -0.55, 0.5, -0.45, 0.4, -0.35])
    linear_score = x_lin[:, : len(beta)] @ beta
    additive_score = (
        1.0 * np.sin(1.4 * x_add[:, 0])
        + 0.9 * np.tanh(1.2 * x_add[:, 1])
        + 0.8 * (x_add[:, 2] > 0.6)
        - 0.8 * (x_add[:, 3] < -0.6)
        + 0.7 * np.sign(x_add[:, 4])
        - 0.7 * np.sign(x_add[:, 5])
        + 0.6 * np.cos(1.1 * x_add[:, 6])
        - 0.6 * (np.abs(x_add[:, 7]) > 1.0)
    )
    interaction_score = (
        1.1 * np.sign(x_int[:, 0] * x_int[:, 1])
        - 1.0 * np.sign(x_int[:, 2] * x_int[:, 3])
        + 0.9 * ((x_int[:, 4] > 0.4) & (x_int[:, 5] < -0.4))
        - 0.9 * ((x_int[:, 6] < -0.3) & (x_int[:, 7] > 0.3))
        + 0.8 * x_int[:, 8] * x_int[:, 9]
        - 0.8 * x_int[:, 10] * x_int[:, 11]
    )
    sparse_weights = np.array([1.4, 1.25, 1.1, 0.95, 0.8, -1.35, -1.2, -1.0, -0.85, -0.7])
    sparse_centered = (x_sparse[:, : len(sparse_weights)] - 0.08) / np.sqrt(0.08 * 0.92)
    sparse_score = sparse_centered @ sparse_weights

    def zscore(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / values.std()

    mechanism_scores = np.column_stack(
        [
            zscore(linear_score),
            zscore(additive_score),
            zscore(interaction_score),
            zscore(sparse_score),
        ]
    )
    # A high-Bayes-error mixture of independent mechanisms. The signed
    # competition term keeps model-family rankings from collapsing onto one
    # dominant risk ordering.
    combined = (
        0.75 * mechanism_scores.sum(axis=1)
        + 0.55 * mechanism_scores[:, 0] * mechanism_scores[:, 1]
        - 0.55 * mechanism_scores[:, 2] * mechanism_scores[:, 3]
        + rng.normal(scale=1.6, size=n)
    )
    combined = zscore(combined)

    lo, hi = -20.0, 20.0
    for _ in range(100):
        intercept = (lo + hi) / 2.0
        prob = 1.0 / (1.0 + np.exp(-(1.0 * combined + intercept)))
        if float(prob.mean()) > 0.35:
            hi = intercept
        else:
            lo = intercept
    intercept = (lo + hi) / 2.0
    prob = 1.0 / (1.0 + np.exp(-(1.0 * combined + intercept)))
    df[SYNTHETIC_STRONG_TARGET] = rng.binomial(1, prob).astype(int)
    df.insert(0, "row_id", np.arange(n, dtype=int))

    bundle = _split_frame_for_target(
        df,
        target=SYNTHETIC_STRONG_TARGET,
        categorical_features=[],
    )
    _write_new_processed_splits(bundle, slug="synthetic_strong")
    return bundle
