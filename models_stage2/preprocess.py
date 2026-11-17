"""Build an adult complete-label cohort and train-only sklearn transformations."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .config import (CATEGORICAL, CATEGORIES, FEATURES, LABELS, MODULE_COLUMNS, NUMERIC,
                         PROCESSED, RANGES, RAW, RESULTS, SEED, TARGETS, ensure_dirs)
    from .download_data import read_module, sha256
except ImportError:
    from config import (CATEGORICAL, CATEGORIES, FEATURES, LABELS, MODULE_COLUMNS, NUMERIC,
                        PROCESSED, RANGES, RAW, RESULTS, SEED, TARGETS, ensure_dirs)
    from download_data import read_module, sha256


def map_diagnosis(series):
    """Only an explicit 'No' is negative. Unknown, refusal, invalid and NA stay NA."""
    return series.map({1: 1.0, 2: 0.0})


def clean_features(frame, strict=False):
    """Apply fixed, label-independent validity rules. No learned statistics here."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    result = frame[FEATURES].copy()
    for name, (low, high) in RANGES.items():
        original = result[name]
        values = pd.to_numeric(original, errors="coerce")
        bad = original.notna() & (values.isna() | ~np.isfinite(values) | ~values.between(low, high))
        if strict and bad.any():
            raise ValueError(f"{name} must be a finite number between {low} and {high} in the documented units.")
        result[name] = values.mask(bad)
    # Public NHANES age has a top code of 80; apply exactly the same rule at inference.
    result["age"] = result["age"].clip(upper=80)
    inconsistent = result.systolic_bp.notna() & result.diastolic_bp.notna() & (result.systolic_bp <= result.diastolic_bp)
    if strict and inconsistent.any():
        raise ValueError("systolic_bp must be greater than diastolic_bp.")
    result.loc[inconsistent, ["systolic_bp", "diastolic_bp"]] = np.nan
    for name, categories in zip(CATEGORICAL, CATEGORIES):
        bad = result[name].notna() & ~result[name].isin(categories)
        if strict and bad.any():
            raise ValueError(f"{name} must be one of {categories} or null.")
        result[name] = result[name].where(~bad, np.nan).astype(object)
        result[name] = result[name].where(result[name].notna(), np.nan)
    return result


def build_cohort(modules):
    """Left merge onto demographics so exclusions are explicit, never silently inner join."""
    merged = modules["DEMO_L"][MODULE_COLUMNS["DEMO_L"]].copy()
    shapes = {name: list(df.shape) for name, df in modules.items()}
    for module in MODULE_COLUMNS:
        if module != "DEMO_L":
            merged = merged.merge(modules[module][MODULE_COLUMNS[module]], on="SEQN", how="left", validate="one_to_one")
    adults = merged.loc[merged.RIDAGEYR >= 20].copy()
    labels = pd.DataFrame({name: map_diagnosis(adults[raw]) for name, raw in TARGETS.items()}, index=adults.index)
    valid = labels.notna().all(axis=1)
    features = pd.DataFrame(index=adults.index)
    for name, raw in [("age", "RIDAGEYR"), ("bmi", "BMXBMI"), ("total_cholesterol", "LBXTC")]:
        features[name] = adults[raw]
    features["sex"] = adults.RIAGENDR.map({1: "Male", 2: "Female"})
    features["diabetes_history"] = adults.DIQ010.map({1: "Yes", 2: "No", 3: "Borderline"})
    features["smoking_history"] = adults.SMQ020.map({1: "Yes", 2: "No"})
    invalid_bp = 0
    for stem, name in [("BPXOSY", "systolic_bp"), ("BPXODI", "diastolic_bp")]:
        low, high = RANGES[name]
        readings = adults[[f"{stem}{i}" for i in range(1, 4)]].copy()
        bad = readings.notna() & ~(readings.ge(low) & readings.le(high))
        invalid_bp += int(bad.to_numpy().sum())
        readings = readings.mask(bad)
        # Discard an inconsistent pair; retain a systolic-only valid reading.
        for i in range(1, 4):
            sys, dia = adults[f"BPXOSY{i}"], adults[f"BPXODI{i}"]
            inconsistent = sys.notna() & dia.notna() & (sys <= dia)
            valid_systolic = sys.between(*RANGES["systolic_bp"])
            invalid_pair = inconsistent | (~valid_systolic if stem == "BPXODI" else False)
            readings.loc[invalid_pair, f"{stem}{i}"] = np.nan
        features[name] = readings.mean(axis=1, skipna=True)
    cleaned = clean_features(features)
    audit = {"raw_shapes": shapes, "merged_shape": list(merged.shape), "merged_columns": list(merged.columns),
        "adult_rows_before_label_filter": len(adults), "excluded_nonadult_or_missing_age": len(merged) - len(adults),
        "excluded_incomplete_or_invalid_labels": int((~valid).sum()), "included_complete_labels": int(valid.sum()),
        "invalid_or_missing_labels_per_condition": labels.isna().sum().to_dict(),
        "invalid_bp_readings_outside_engineering_bounds": invalid_bp,
        "feature_values_replaced_by_missing": (features.notna() & cleaned.isna()).sum().to_dict(),
        "cohort_policy": "Adults 20+, all three labels known; retain missing clinical inputs for train-only imputation.",
        "survey_policy": "Unweighted sample ML only. WTMEC2YR/WTPH2YR/SDMVSTRA/SDMVPSU retained for audit, never predictors."}
    kept = adults.loc[valid, ["SEQN", "RIDSTATR", "WTMEC2YR", "WTPH2YR", "SDMVSTRA", "SDMVPSU"]].copy()
    kept = kept.join(cleaned.loc[valid]).join(labels.loc[valid]).reset_index(drop=True)
    kept["SEQN"] = kept.SEQN.astype("int64")
    kept[LABELS] = kept[LABELS].astype("int64")
    if kept.empty:
        raise ValueError("No adults remain with complete valid labels. Check cycle, merges and source data.")
    audit["missing_features"] = kept[FEATURES].isna().sum().to_dict()
    audit["interview_only_included"] = int((kept.RIDSTATR == 1).sum())
    return kept, audit


def make_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
        ("encode", OneHotEncoder(categories=CATEGORIES, handle_unknown="error", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)],
                             remainder="drop", verbose_feature_names_out=True)


def split_participants(frame, seed=SEED, force_fallback=False):
    if frame.SEQN.duplicated().any():
        raise ValueError("Duplicate participant IDs; cannot split safely.")
    y = frame[LABELS].to_numpy(dtype=int)
    if len(frame) < 30 or any(np.bincount(y[:, j], minlength=2).min() < 6 for j in range(3)):
        raise ValueError("Too few participants or positives/negatives for reliable 70/15/15 partitions.")
    idx = np.arange(len(frame))
    # Augment labels with positive pairs to encourage preservation of co-occurrence.
    strata = np.column_stack([y, y[:, 0] * y[:, 1], y[:, 0] * y[:, 2], y[:, 1] * y[:, 2]])
    try:
        if force_fallback:
            raise ImportError
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        tr, rest = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.30, random_state=seed).split(idx, strata))
        va, te = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.50, random_state=seed).split(rest, strata[rest]))
        partitions = {"train": tr, "validation": rest[va], "test": rest[te]}
        method = "Iterative multi-label stratification augmented with positive label pairs"
    except ImportError:
        warnings.warn("iterative-stratification unavailable; using label-combination stratification where feasible.")
        combos = np.array(["".join(map(str, row)) for row in y])
        def fallback_split(indices, size):
            counts = pd.Series(combos[indices]).value_counts()
            possible = counts.min() >= 2 and min(int(len(indices) * size), int(len(indices) * (1-size))) >= len(counts)
            return train_test_split(indices, test_size=size, random_state=seed,
                                    stratify=combos[indices] if possible else None)
        tr, rest = fallback_split(idx, .30)
        va, te = fallback_split(rest, .50)
        partitions = {"train": tr, "validation": va, "test": te}
        method = "Label-combination stratification when feasible, otherwise seeded random split; inspect distributions"
    ids = [set(frame.iloc[part].SEQN) for part in partitions.values()]
    if any(ids[a] & ids[b] for a, b in [(0, 1), (0, 2), (1, 2)]) or len(set.union(*ids)) != len(frame):
        raise AssertionError("Participant partition overlap or loss.")
    for name, indices in partitions.items():
        if any(len(np.unique(y[indices, j])) != 2 for j in range(3)):
            raise ValueError(f"{name} lacks a class for at least one label. Need more real data; test metrics would be undefined.")
    return partitions, method


def prepare(raw_dir=RAW, output=PROCESSED, eda=True):
    ensure_dirs()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    modules = {name: read_module(Path(raw_dir) / f"{name}.xpt", name) for name in MODULE_COLUMNS}
    frame, audit = build_cohort(modules)
    audit["source_sha256"] = {name: sha256(Path(raw_dir) / f"{name}.xpt") for name in MODULE_COLUMNS}
    frame.to_csv(output / "cohort.csv", index=False)
    audit["cohort_sha256"] = sha256(output / "cohort.csv")
    (RESULTS / "cohort_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    if eda:
        try:
            from .evaluate import exploratory_analysis
        except ImportError:
            from evaluate import exploratory_analysis
        exploratory_analysis(frame, audit)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    args = parser.parse_args()
    prepare(args.raw_dir)


if __name__ == "__main__":
    main()
