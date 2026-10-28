"""Audit historical Freddie labels; prepare a separately named strict endpoint.

Default execution reads only vintages already used in this project. Loan IDs
never appear in public outputs. This does NOT establish an independent D_eval.
The strict endpoint is an observation-conditioned sensitivity population, not an
unbiased estimate for all originations. Unknown follow-up is never made negative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_freddie_mac import ORIG_COLUMNS, PERF_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs/revisions/kbs_v1_9"
HISTORICAL = (2018, 2021, 2022, 2023)
STRICT_TARGET = "strict_calendar_age0_24_60plus_or_reo"
MASK_0_24 = (1 << 25) - 1
MASK_1_24 = MASK_0_24 - 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def month_index(values):
    """Parse YYYYMM exactly; invalid dates stay -1, with no date inference."""
    text = values.fillna("").astype(str).str.strip()
    valid = text.str.fullmatch(r"\d{6}")
    year = pd.to_numeric(text.str[:4], errors="coerce")
    month = pd.to_numeric(text.str[4:], errors="coerce")
    valid &= year.between(1900, 2100) & month.between(1, 12)
    return np.where(valid, year.fillna(0) * 12 + month.fillna(0) - 1, -1).astype(int)


def month_string(value):
    return None if value < 0 else f"{value // 12:04d}-{value % 12 + 1:02d}"


def summarize(frame, name, scope):
    y = frame.legacy_y.to_numpy()
    defined = frame.strict_known.to_numpy()
    strict = frame.strict_y.to_numpy()
    return dict(
        cohort=name, scope=scope, N=len(frame), legacy_positive=int(y.sum()),
        calendar_event_positive=int(frame.calendar_event.sum()),
        calendar_numeric_60plus_positive=int(frame.numeric_event.sum()),
        calendar_reo_positive=int(frame.reo_event.sum()),
        legacy_vs_calendar_event_changed=int(np.sum(y != frame.calendar_event.to_numpy())),
        legacy_positive_calendar_no_event=int(np.sum((y == 1) & ~frame.calendar_event.to_numpy())),
        legacy_negative_calendar_event=int(np.sum((y == 0) & frame.calendar_event.to_numpy())),
        calendar_mature=int(frame.calendar_mature.sum()),
        all_25_calendar_ages_valid=int(frame.full_0_24.sum()),
        all_24_scheduled_ages_valid=int(frame.full_1_24.sum()),
        calendar_missing_status_loans=int((frame.invalid_bits > 0).sum()),
        unlocated_reporting_rows_loans=int(frame.unlocated.sum()),
        conflicting_month_status_loans=int(frame.conflicting.sum()),
        recorded_exit_in_calendar_window=int(frame.exit_in_window.sum()),
        loan_age_calendar_discordance_loans=int(frame.age_discordance.sum()),
        strict_known=int(defined.sum()), strict_positive=int(np.sum(strict == 1)),
        strict_negative=int(np.sum(strict == 0)), strict_unknown_excluded=int(np.sum(~defined)),
        strict_vs_legacy_changed_among_known=int(np.sum(y[defined] != strict[defined])),
    )


def audit_year(year, root=ROOT):
    path = root / f"data/raw/freddie_mac_samples_2018_2025/sample_{year}.zip"
    with zipfile.ZipFile(path) as archive:
        orig_name = next(n for n in archive.namelist() if n.endswith(f"sample_orig_{year}.txt"))
        orig = pd.read_csv(archive.open(orig_name), sep="|", header=None, names=ORIG_COLUMNS,
                           usecols=["loan_sequence_number", "first_payment_date"], dtype=str)
        ids = pd.Index(orig.loan_sequence_number)
        if ids.has_duplicates or ids.hasnans:
            raise ValueError("Origination IDs must be present and unique")
        first = month_index(orig.first_payment_date)
        n = len(orig)
        state = {key: np.zeros(n, dtype=bool) for key in
                 ("legacy_y", "legacy_window", "calendar_event", "numeric_event", "reo_event",
                  "exit_in_window", "age_discordance", "unlocated")}
        observed = np.zeros(n, dtype=np.uint32)
        valid_bits = np.zeros(n, dtype=np.uint32)
        invalid_bits = np.zeros(n, dtype=np.uint32)
        status_min = np.full((n, 25), 32767, dtype=np.int16)
        status_max = np.full((n, 25), -1, dtype=np.int16)
        cutoff = -1
        perf_name = next(n for n in archive.namelist() if n.endswith(f"sample_svcg_{year}.txt"))
        use = ["loan_sequence_number", "monthly_reporting_period", "loan_age",
               "current_loan_delinquency_status", "zero_balance_code"]
        reader = pd.read_csv(archive.open(perf_name), sep="|", header=None, names=PERF_COLUMNS,
                             usecols=use, dtype=str, chunksize=250000)
        for block in reader:
            ix = ids.get_indexer(block.loan_sequence_number)
            if (ix < 0).any():
                raise ValueError("Servicing ID absent from origination; inspect source contract")
            month = month_index(block.monthly_reporting_period)
            cutoff = max(cutoff, int(month.max()))
            age = month - first[ix] + 1
            located = (month >= 0) & (first[ix] >= 0)
            window = located & (age >= 0) & (age <= 24)
            raw_age = pd.to_numeric(block.loan_age, errors="coerce").to_numpy()
            old_window = np.isfinite(raw_age) & (raw_age >= 0) & (raw_age <= 24)
            text = block.current_loan_delinquency_status.fillna("").str.strip().str.upper()
            numeric = pd.to_numeric(text, errors="coerce").to_numpy()
            numeric_valid = np.isfinite(numeric) & (numeric >= 0) & (numeric < 30000) & (numeric == np.floor(numeric))
            reo = text.eq("RA").to_numpy()
            valid = numeric_valid | reo
            hit = (numeric_valid & (numeric >= 2)) | reo
            np.logical_or.at(state["legacy_y"], ix, old_window & hit)
            np.logical_or.at(state["legacy_window"], ix, old_window)
            np.logical_or.at(state["calendar_event"], ix, window & hit)
            np.logical_or.at(state["numeric_event"], ix, window & numeric_valid & (numeric >= 2))
            np.logical_or.at(state["reo_event"], ix, window & reo)
            np.logical_or.at(state["unlocated"], ix, ~located)
            np.logical_or.at(state["age_discordance"], ix, (window | old_window) & located & (raw_age != age))
            exits = ~block.zero_balance_code.fillna("").str.strip().isin(["", "0", "00"]).to_numpy()
            np.logical_or.at(state["exit_in_window"], ix, window & exits)
            wi, wa = ix[window], age[window]
            bit = np.left_shift(np.uint32(1), wa.astype(np.uint32))
            np.bitwise_or.at(observed, wi, bit)
            np.bitwise_or.at(valid_bits, wi, np.where(valid[window], bit, 0).astype(np.uint32))
            np.bitwise_or.at(invalid_bits, wi, np.where(~valid[window], bit, 0).astype(np.uint32))
            good = window & valid
            status = np.where(reo, 30000, np.nan_to_num(numeric, nan=-1)).astype(np.int16)
            np.minimum.at(status_min, (ix[good], age[good]), status[good])
            np.maximum.at(status_max, (ix[good], age[good]), status[good])
    frame = pd.DataFrame(state)
    frame["observed_bits"] = observed
    frame["valid_bits"] = valid_bits
    frame["invalid_bits"] = invalid_bits
    frame["conflicting"] = ((status_max >= 0) & (status_min != status_max)).any(axis=1)
    frame["full_0_24"] = (valid_bits & MASK_0_24) == MASK_0_24
    frame["full_1_24"] = (valid_bits & MASK_1_24) == MASK_1_24
    frame["calendar_mature"] = (first >= 0) & (first + 23 <= cutoff)
    # Existence of an observed event identifies a positive. For a negative,
    # require every calendar age 0..24 plus no malformed/conflicting records.
    clean_negative = frame.full_0_24 & (invalid_bits == 0) & ~frame.unlocated & ~frame.conflicting
    frame["strict_known"] = frame.calendar_mature & (frame.calendar_event | clean_negative)
    frame["strict_y"] = np.where(frame.strict_known, frame.calendar_event.astype(int), -1)
    rows = [summarize(frame, f"freddie_mac_{year}", "source_sample")]
    eligible = frame.legacy_window.to_numpy().copy()
    if year == 2023:
        eligible &= (first >= 0) & (first <= 2023 * 12 + 9)
    old = frame.loc[eligible].reset_index(drop=True)
    old["row_id"] = np.arange(1, len(old) + 1)
    # Reconstruct identity from original order and frozen inclusion logic, then
    # validate against ALL three historical split labels before trusting mapping.
    split_root = root / f"data/processed/freddie_mac_{year}_ever_60dpd_24m"
    split_ids = []
    for split in ("train", "val", "test"):
        stored = pd.read_parquet(split_root / f"{split}.parquet", columns=["row_id", "ever_60dpd_24m"])
        positions = stored.row_id.to_numpy(dtype=int) - 1
        if positions.min() < 0 or positions.max() >= len(old):
            raise ValueError("Historical row mapping outside reconstructed cohort")
        if not np.array_equal(old.legacy_y.to_numpy()[positions], stored.ever_60dpd_24m.to_numpy()):
            raise ValueError("Historical labels do not replay; identity mapping is not certified")
        split_ids.extend(stored.row_id.tolist())
        if split == "test":
            test = old.iloc[positions].copy()
            rows.append(summarize(test, f"freddie_mac_{year}", "historical_test"))
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != set(old.row_id):
        raise ValueError("Historical train/validation/test IDs are not a partition")
    rows.append(summarize(old, f"freddie_mac_{year}", "historical_full_sample"))
    meta = dict(cohort=f"freddie_mac_{year}", source_sha256=digest(path),
                source_relative_path=str(path.relative_to(root)),
                observed_max_reporting_month=month_string(cutoff),
                first_payment_min=month_string(int(first.min())),
                first_payment_max=month_string(int(first.max())),
                latest_calendar_window_end=month_string(int(first.max() + 23)),
                historical_split_ids_disjoint=True, all_historical_split_labels_replayed=True,
                independent_evaluation_eligible=False)
    # No original loan identifier leaves this function.
    return rows, meta, test.drop(columns=["legacy_window"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=list(HISTORICAL), choices=HISTORICAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write-private-test-labels", action="store_true",
                        help="Write restricted derived labels under ignored results/local_cache only")
    args = parser.parse_args()
    rows, logs = [], []
    for year in args.years:
        records, meta, frame = audit_year(year)
        rows.extend(records); logs.append(meta)
        if args.write_private_test_labels:
            private = ROOT / "results/tables/v1_9_revision/local_cache"
            private.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(private / f"freddie_mac_{year}_strict_label_audit.parquet", index=False)
        print(f"Freddie {year}: historical mapping replayed; aggregate audit complete", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "outcome_audit.csv", index=False)
    (args.out_dir / "outcome_audit_metadata.json").write_text(json.dumps(dict(
        design="retrospective endpoint-definition and follow-up audit; no independent D_eval",
        strict_target=STRICT_TARGET, script_sha256=digest(__file__), cohorts=logs), indent=2) + "\n")


if __name__ == "__main__":
    main()
