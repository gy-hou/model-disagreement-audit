"""M1 feature engineering for v0.4 Wave 1 / Step B1.

The engineered candidates remain logistic-regression models. Feature design is
separate from the model class: M1 is still linear in the model input space.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessing import CATEGORICAL_FEATURES


PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


def engineer_m1_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["LIMIT_BAL", *BILL_COLS, *PAY_AMT_COLS]:
        if col in out.columns:
            out[f"log1p_{col}"] = np.sign(out[col]) * np.log1p(np.abs(out[col]))

    for pay_col in PAY_COLS:
        if pay_col in out.columns and "AGE" in out.columns:
            out[f"{pay_col}_x_AGE"] = out[pay_col] * out["AGE"]
        if pay_col in out.columns and "LIMIT_BAL" in out.columns:
            out[f"{pay_col}_x_LIMIT"] = out[pay_col] * out["LIMIT_BAL"]

    for bill_col, pay_amt_col in zip(BILL_COLS, PAY_AMT_COLS, strict=True):
        if bill_col in out.columns and pay_amt_col in out.columns:
            out[f"{pay_amt_col}_over_{bill_col}"] = out[pay_amt_col] / (np.abs(out[bill_col]) + 1.0)

    for col in ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]:
        if col in out.columns:
            out[f"{col}_sq"] = out[col] ** 2

    return out


FICO_RATIO_PAIRS = [
    ("nr_trades_insolvent_for_over_60_days", "nr_total_trades"),
    ("nr_revolving_trades_with_balance", "nr_total_trades"),
    ("nr_banks_with_high_ratio", "nr_total_trades"),
]

FICO_BURDEN_INTERACTIONS = [
    ("net_fraction_of_revolving_burden", "nr_revolving_trades_with_balance"),
    ("net_fraction_of_installment_burden", "nr_installment_trades_with_balance"),
]

FICO_RECENCY_INTERACTIONS = [
    ("months_since_last_illegal_trade", "maximum_illegal_trades"),
    ("months_since_last_inquiry_not_recent", "nr_inquiries_in_last_6_months"),
]

FICO_LOG_COLS = [
    "nr_total_trades",
    "months_since_first_trade",
    "nr_trades_initiated_in_last_year",
]

SYNTHETIC_POLYNOMIAL_COLS = [f"x_lin_{i}" for i in range(1, 11)]


def select_fico_polynomial_columns(df: pd.DataFrame, y: np.ndarray, k: int = 4) -> list[str]:
    scores: list[tuple[float, str]] = []
    for col in df.columns:
        values = df[col].to_numpy()
        if len(np.unique(values)) <= 1:
            continue
        try:
            auc = float(roc_auc_score(y, values))
        except ValueError:
            continue
        scores.append((max(auc, 1.0 - auc), col))
    scores.sort(reverse=True)
    return [col for _, col in scores[:k]]


def engineer_m1_features_fico(
    df: pd.DataFrame,
    polynomial_columns: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()

    for numerator, denominator in FICO_RATIO_PAIRS:
        if numerator in out.columns and denominator in out.columns:
            out[f"{numerator}_over_{denominator}"] = out[numerator] / (np.abs(out[denominator]) + 1.0)

    for left, right in FICO_BURDEN_INTERACTIONS:
        if left in out.columns and right in out.columns:
            out[f"{left}_x_{right}"] = out[left] * out[right]

    for left, right in FICO_RECENCY_INTERACTIONS:
        if left in out.columns and right in out.columns:
            out[f"{left}_x_{right}"] = out[left] * out[right]

    for col in FICO_LOG_COLS:
        if col in out.columns:
            out[f"log1p_{col}"] = np.sign(out[col]) * np.log1p(np.abs(out[col]))

    for col in polynomial_columns or []:
        if col in out.columns:
            out[f"{col}_sq"] = out[col] ** 2

    return out


def engineer_m1_features_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SYNTHETIC_POLYNOMIAL_COLS:
        if col in out.columns:
            out[f"{col}_sq"] = out[col] ** 2
    return out


class M1EngineeredLogistic:
    def __init__(self, *, feature_engineering: bool, logistic_params: dict[str, object]) -> None:
        self.feature_engineering = feature_engineering
        self.logistic_params = logistic_params
        self.dataset = os.environ.get("V0_4_DATASET", "uci").lower()
        self.fico_polynomial_columns_: list[str] | None = None
        self.feature_columns_: list[str] | None = None
        self.pipeline_: Pipeline | None = None

    def __setstate__(self, state: dict) -> None:
        # Backward-compat shim for artifacts pickled before Wave 2.5 added
        # the `dataset` and `fico_polynomial_columns_` attributes. UCI was the
        # only dataset at the time, so missing attrs default to UCI semantics.
        self.__dict__.update(state)
        if "dataset" not in state:
            self.dataset = "uci"
        if "fico_polynomial_columns_" not in state:
            self.fico_polynomial_columns_ = None

    def _transform_features(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.feature_engineering and self.dataset == "fico":
            transformed = engineer_m1_features_fico(x, self.fico_polynomial_columns_)
        elif self.feature_engineering and self.dataset in {"synthetic", "synthetic_v2", "synthetic_strong"}:
            transformed = engineer_m1_features_synthetic(x)
        elif self.feature_engineering and self.dataset in {"hmda", "lendingclub", "hmeq"}:
            transformed = x.copy()
        elif self.feature_engineering:
            transformed = engineer_m1_features(x)
        else:
            transformed = x.copy()
        if self.feature_columns_ is not None:
            transformed = transformed.reindex(columns=self.feature_columns_, fill_value=0)
        return transformed

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "M1EngineeredLogistic":
        if self.feature_engineering and self.dataset == "fico":
            self.fico_polynomial_columns_ = select_fico_polynomial_columns(x, y)
        transformed = self._transform_features(x)
        self.feature_columns_ = list(transformed.columns)
        categorical = [col for col in CATEGORICAL_FEATURES if col in transformed.columns]
        numeric = [col for col in transformed.columns if col not in categorical]
        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ]
        )
        model = LogisticRegression(**self.logistic_params)
        self.pipeline_ = Pipeline([("preprocess", preprocess), ("model", model)])
        self.pipeline_.fit(transformed, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("M1EngineeredLogistic must be fit before predict_proba")
        transformed = self._transform_features(x)
        return self.pipeline_.predict_proba(transformed)
