"""Shared, side-effect-free data and PLS utilities for the v1 tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, RepeatedKFold
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler

FEATURE_META_COLUMNS = ("Sum formula", "O count", "DBE")
DEFAULT_DATA_FILE = "FKTv2-onlySamples_Reduced(viaTrendlineHigher0-6).csv"
DEFAULT_METADATA_FILE = "FKT-metadata_Histogram.csv"


@dataclass
class ProjectData:
    """Aligned feature matrix, metadata, and feature annotations."""

    X: pd.DataFrame
    metadata: pd.DataFrame
    sample_column: str
    feature_metadata: pd.DataFrame


def _find_sample_column(metadata: pd.DataFrame) -> str:
    for name in ("Sample", "sample", "ID", "id", "Name"):
        if name in metadata.columns:
            return name
    raise ValueError("Metadata must contain Sample, sample, ID, id, or Name.")


def load_project_data(data_path: str | Path, metadata_path: str | Path) -> ProjectData:
    """Load both CSV files and return samples x features aligned by sample ID."""
    raw_features = pd.read_csv(data_path, index_col=0)
    metadata = pd.read_csv(metadata_path)
    sample_column = _find_sample_column(metadata)

    metadata = metadata.copy()
    metadata[sample_column] = metadata[sample_column].astype(str).str.strip()
    raw_features.columns = raw_features.columns.astype(str).str.strip()
    raw_features.index = raw_features.index.astype(str).str.strip()

    feature_metadata = pd.DataFrame(index=raw_features.index)
    for column in FEATURE_META_COLUMNS:
        matches = [name for name in raw_features.columns if name.lower() == column.lower()]
        if matches:
            feature_metadata[column] = raw_features.pop(matches[0])
        else:
            feature_metadata[column] = np.nan

    sample_ids = [sample for sample in raw_features.columns if sample in set(metadata[sample_column])]
    if not sample_ids:
        raise ValueError("No sample IDs overlap between the feature and metadata CSV files.")

    aligned_metadata = metadata.set_index(sample_column).loc[sample_ids].reset_index()
    numeric_features = raw_features.loc[:, sample_ids].apply(pd.to_numeric, errors="coerce")
    numeric_features = numeric_features.dropna(axis=0, how="all").dropna(axis=1, how="all")
    X = numeric_features.T
    X.index.name = sample_column
    return ProjectData(X=X, metadata=aligned_metadata, sample_column=sample_column,
                       feature_metadata=feature_metadata.loc[X.columns])


def select_samples(project: ProjectData, samples: Sequence[str] | None) -> ProjectData:
    """Return a project restricted to requested sample IDs."""
    if not samples:
        return project
    requested = {str(sample).strip() for sample in samples}
    selected = [sample for sample in project.X.index if sample in requested]
    if not selected:
        raise ValueError("None of the requested samples were found.")
    metadata = project.metadata[project.metadata[project.sample_column].isin(selected)].copy()
    return ProjectData(project.X.loc[selected], metadata, project.sample_column, project.feature_metadata)


def make_scaler(method: str):
    if method in ("autoscale", "zscore"):
        return StandardScaler()
    if method in ("center", "mean_center"):
        return StandardScaler(with_std=False)
    if method == "minmax":
        return MinMaxScaler()
    if method == "maxabs":
        return MaxAbsScaler()
    if method == "robust":
        return RobustScaler()
    if method == "normalize":
        return Normalizer()
    if method == "none":
        return None
    raise ValueError(f"Unknown scaling method: {method}")


def transform_array(values: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return values
    if method == "sqrt":
        return np.sqrt(np.clip(values, 0, None))
    if method == "log10":
        return np.log10(np.clip(values, 0, None) + 1.0)
    if method == "log2":
        return np.log2(np.clip(values, 0, None) + 1.0)
    if method == "cuberoot":
        return np.cbrt(values)
    raise ValueError(f"Unknown X transform: {method}")


def prepare_xy(X: pd.DataFrame, y: pd.Series, x_transform: str, x_scaling: str,
               y_scaling: str):
    valid = pd.to_numeric(y, errors="coerce").notna()
    X_valid = X.loc[valid].apply(pd.to_numeric, errors="coerce")
    valid_rows = ~X_valid.isna().all(axis=1)
    X_valid = X_valid.loc[valid_rows]
    y_valid = pd.to_numeric(y.loc[X_valid.index], errors="coerce")
    values = transform_array(X_valid.to_numpy(dtype=float), x_transform)
    x_scaler = make_scaler(x_scaling)
    X_ready = x_scaler.fit_transform(values) if x_scaler else values
    y_values = y_valid.to_numpy(dtype=float).reshape(-1, 1)
    y_scaler = make_scaler(y_scaling)
    y_ready = y_scaler.fit_transform(y_values) if y_scaler else y_values
    return X_valid, y_valid, X_ready, y_ready, x_scaler, y_scaler


def cv_predict(X: pd.DataFrame, y: pd.Series, n_components: int, x_scaling: str,
               y_scaling: str, x_transform: str, method: str = "kfold",
               folds: int = 5, repeats: int = 3, groups=None, random_state: int = 42):
    X_valid, y_valid, _, _, _, _ = prepare_xy(X, y, x_transform, x_scaling, y_scaling)
    values = X_valid.to_numpy(dtype=float)
    targets = y_valid.to_numpy(dtype=float).reshape(-1, 1)
    if len(values) < 3:
        raise ValueError("At least 3 valid samples are required for cross-validation.")
    n_splits = min(max(int(folds), 2), len(values))
    if method == "loo":
        splitter = LeaveOneOut()
    elif method == "repeatedkfold":
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=int(repeats), random_state=random_state)
    elif method == "groupkfold":
        if groups is None:
            raise ValueError("Group labels are required for GroupKFold.")
        groups = np.asarray(groups)
        n_splits = min(n_splits, len(np.unique(groups)))
        if n_splits < 2:
            raise ValueError("GroupKFold requires at least two groups.")
        splitter = GroupKFold(n_splits=n_splits)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    predictions = np.full(len(values), np.nan)
    split_groups = groups if method == "groupkfold" else None
    for train_idx, test_idx in splitter.split(values, targets, split_groups):
        train_x = transform_array(values[train_idx], x_transform)
        test_x = transform_array(values[test_idx], x_transform)
        x_scaler = make_scaler(x_scaling)
        if x_scaler:
            train_x = x_scaler.fit_transform(train_x)
            test_x = x_scaler.transform(test_x)
        y_scaler = make_scaler(y_scaling)
        train_y = y_scaler.fit_transform(targets[train_idx]) if y_scaler else targets[train_idx]
        components = max(1, min(int(n_components), len(train_idx) - 1, train_x.shape[1]))
        model = PLSRegression(n_components=components, scale=False).fit(train_x, train_y)
        prediction = model.predict(test_x)
        if y_scaler:
            prediction = y_scaler.inverse_transform(prediction)
        predictions[test_idx] = prediction.ravel()
    return X_valid.index, y_valid, predictions


def fit_pls(X: pd.DataFrame, y: pd.Series, n_components: int, x_scaling: str,
            y_scaling: str, x_transform: str):
    X_valid, y_valid, X_ready, y_ready, _, y_scaler = prepare_xy(
        X, y, x_transform, x_scaling, y_scaling)
    components = max(1, min(int(n_components), len(X_valid) - 1, X_ready.shape[1]))
    model = PLSRegression(n_components=components, scale=False).fit(X_ready, y_ready)
    predicted_scaled = model.predict(X_ready)
    predicted = y_scaler.inverse_transform(predicted_scaled).ravel() if y_scaler else predicted_scaled.ravel()
    return model, X_valid, y_valid, predicted


def metrics(actual, predicted):
    return {"r2": float(r2_score(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))}


def parse_formula(formula) -> tuple[float, float, float]:
    import re
    counts = {element: int(count) if count else 1
              for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula))}
    carbon = counts.get("C", np.nan)
    if pd.isna(carbon) or carbon == 0:
        return np.nan, np.nan, np.nan
    return float(carbon), float(counts.get("H", np.nan)), float(counts.get("O", 0))


def compute_hc_oc(feature_metadata: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=feature_metadata.index, columns=["H/C", "O/C"], dtype=float)
    for feature, row in feature_metadata.iterrows():
        carbon, hydrogen, oxygen = parse_formula(row.get("Sum formula"))
        if not pd.isna(carbon):
            result.loc[feature] = [hydrogen / carbon, oxygen / carbon]
        elif not pd.isna(row.get("DBE")) and not pd.isna(row.get("O count")):
            result.loc[feature] = [2 * (2 - float(row["DBE"])), float(row["O count"])]
    return result
