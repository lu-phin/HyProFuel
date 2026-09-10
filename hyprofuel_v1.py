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


@dataclass
class OriginalWorkflowData:
    """Loaded state for the original dashboard and evaluator workflow."""

    filtered_data: pd.DataFrame
    filtered_metadata: pd.DataFrame
    sample_column: str
    numeric_metadata_columns: list[str]
    feature_metadata: dict[str, pd.Series]
    formula_array: np.ndarray
    dbe_array: np.ndarray
    ocount_array: np.ndarray
    vk_lookup: dict[str, tuple[float, float, str, float, float]]


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


def extract_feature_metadata_columns(raw_features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Split positional feature metadata columns from the raw feature table."""
    data = raw_features.copy()
    arrays: dict[str, np.ndarray] = {}
    for column in ("DBE", "O count", "Sum formula"):
        matches = [name for name in data.columns if name.strip().lower() == column.lower()]
        if matches:
            arrays[column] = data[matches[0]].to_numpy(copy=True)
            data = data.drop(columns=[matches[0]])
        else:
            arrays[column] = np.full(len(data), np.nan, dtype=object)
    return data, arrays


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


def to_float_array(values) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for index, value in enumerate(values):
        try:
            output[index] = float(value)
        except (TypeError, ValueError):
            pass
    return output


def sanity_check_orientation_and_alignment(filtered_data: pd.DataFrame, filtered_metadata: pd.DataFrame,
                                           sample_column: str) -> None:
    print("\n=== SANITY CHECK ===")
    print(f"Raw feature table shape (features x samples): {filtered_data.shape}")
    X_check = (filtered_data.apply(pd.to_numeric, errors="coerce")
               .dropna(axis=0, how="all")
               .dropna(axis=1, how="all")
               .T)
    print(f"Model matrix shape after transpose (samples x features): {X_check.shape}")
    print(f"Number of samples: {X_check.shape[0]}")
    print(f"Number of features: {X_check.shape[1]}")

    meta_ids = filtered_metadata[sample_column].astype(str).str.strip().tolist()
    x_ids = X_check.index.astype(str).str.strip().tolist()
    common = set(x_ids).intersection(set(meta_ids))
    print(f"Samples in X: {len(x_ids)}")
    print(f"Samples in metadata: {len(meta_ids)}")
    print(f"Overlapping samples: {len(common)}")

    missing_in_meta = sorted(set(x_ids) - set(meta_ids))
    missing_in_x = sorted(set(meta_ids) - set(x_ids))
    if missing_in_meta:
        print(f"Warning: {len(missing_in_meta)} sample(s) in X not found in metadata.")
    if missing_in_x:
        print(f"Warning: {len(missing_in_x)} sample(s) in metadata not found in X.")
    if not missing_in_meta and not missing_in_x:
        print("Sample labels match between X and metadata.")
    if X_check.shape[0] != len(common):
        print("Note: some samples will be dropped during alignment.")
    print("====================\n")


def select_samples_popup(sample_names, title="Select samples to include",
                         default_select_all=True):
    import tkinter as tk
    from tkinter import ttk

    sample_names = list(sample_names)
    selected = set(sample_names) if default_select_all else set()
    root = tk.Tk()
    root.title(title)
    root.geometry("520x600")
    search_var = tk.StringVar()

    ttk.Label(root, text="Filter:").pack(anchor="w", padx=10, pady=(10, 0))
    ttk.Entry(root, textvariable=search_var).pack(fill="x", padx=10)

    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    scrollbar = ttk.Scrollbar(frame, orient="vertical")
    listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    def refresh_list():
        listbox.delete(0, tk.END)
        filt = search_var.get().strip().lower()
        for sample in sample_names:
            if (not filt) or (filt in sample.lower()):
                listbox.insert(tk.END, sample)
        for i in range(listbox.size()):
            if listbox.get(i) in selected:
                listbox.selection_set(i)

    def select_all():
        nonlocal selected
        selected = set(sample_names)
        refresh_list()

    def select_none():
        nonlocal selected
        selected = set()
        refresh_list()

    search_var.trace_add("write", lambda *_: refresh_list())
    button_frame = ttk.Frame(root)
    button_frame.pack(fill="x", padx=10, pady=(0, 10))
    ttk.Button(button_frame, text="Select all", command=select_all).pack(side="left")
    ttk.Button(button_frame, text="Select none", command=select_none).pack(side="left", padx=(10, 0))
    result = {"samples": None}

    def on_ok():
        visible = [listbox.get(i) for i in range(listbox.size())]
        current_selected = set(listbox.get(i) for i in listbox.curselection())
        nonlocal selected
        selected = (selected - set(visible)) | current_selected
        result["samples"] = sorted(selected)
        root.destroy()

    def on_cancel():
        result["samples"] = sample_names
        root.destroy()

    ttk.Button(button_frame, text="OK (start app)", command=on_ok).pack(side="right")
    ttk.Button(button_frame, text="Cancel (include all)", command=on_cancel).pack(side="right", padx=(0, 10))
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    refresh_list()
    root.mainloop()
    return result["samples"]


def _select_requested_samples(sample_names: list[str], samples: Sequence[str] | None) -> list[str]:
    if not samples:
        return sample_names
    requested = {str(sample).strip() for sample in samples}
    selected = [sample for sample in sample_names if sample in requested]
    if not selected:
        raise ValueError("None of the requested samples were found.")
    return selected


def _aligned_feature_positions(original_index: Sequence[str], filtered_index: Sequence[str]) -> np.ndarray:
    positions_by_name: dict[str, list[int]] = {}
    for position, name in enumerate(original_index):
        positions_by_name.setdefault(str(name), []).append(position)
    seen: dict[str, int] = {}
    resolved_positions = []
    for feature in filtered_index:
        feature_name = str(feature)
        count = seen.get(feature_name, 0)
        positions = positions_by_name[feature_name]
        resolved_positions.append(positions[count] if count < len(positions) else positions[-1])
        seen[feature_name] = count + 1
    return np.asarray(resolved_positions, dtype=int)


def load_original_workflow_data(data_path: str | Path, metadata_path: str | Path,
                                samples: Sequence[str] | None = None, *,
                                use_popup: bool = False,
                                popup_title: str = "PLS Dashboard – Select samples to include (default: all)",
                                default_select_all: bool = True,
                                run_sanity_check: bool = True) -> OriginalWorkflowData:
    """Load data as expected by the original dashboard and evaluator scripts."""
    raw_features = pd.read_csv(data_path, index_col=0)
    metadata = pd.read_csv(metadata_path)
    print(f"Raw data shape: {raw_features.shape}  |  index unique: {raw_features.index.is_unique}")

    data, feature_arrays = extract_feature_metadata_columns(raw_features)
    for column in ("DBE", "O count", "Sum formula"):
        matches = [name for name in raw_features.columns if name.strip().lower() == column.lower()]
        if matches:
            print(f"  Extracted '{matches[0]}' → '{column}'")
        else:
            print(f"  Warning: '{column}' not found.")

    sample_column = _find_sample_column(metadata)
    metadata = metadata.copy()
    metadata[sample_column] = metadata[sample_column].astype(str).str.strip()
    data.index = data.index.astype(str).str.strip()
    data.columns = data.columns.astype(str).str.strip()

    valid_samples = set(metadata[sample_column])
    filtered_data = data.loc[:, data.columns.isin(valid_samples)]
    filtered_metadata = metadata[metadata[sample_column].isin(filtered_data.columns)].copy()
    if run_sanity_check:
        sanity_check_orientation_and_alignment(filtered_data, filtered_metadata, sample_column)

    all_samples = list(filtered_data.columns)
    chosen_samples = (select_samples_popup(all_samples, title=popup_title,
                                          default_select_all=default_select_all)
                      if use_popup else _select_requested_samples(all_samples, samples))
    chosen_set = set(chosen_samples)
    filtered_data = filtered_data.loc[:, filtered_data.columns.isin(chosen_set)]
    filtered_metadata = filtered_metadata[filtered_metadata[sample_column].isin(chosen_set)].copy()
    print(f"Selected {filtered_data.shape[1]} / {len(all_samples)} samples.")

    numeric_metadata_columns = []
    for column in [name for name in filtered_metadata.columns if name != sample_column]:
        try:
            pd.to_numeric(filtered_metadata[column], errors="raise")
            numeric_metadata_columns.append(column)
        except Exception:
            pass
    print(f"Found {len(numeric_metadata_columns)} numeric metadata columns for Y variables")

    dbe_array = to_float_array(feature_arrays["DBE"])
    ocount_array = to_float_array(feature_arrays["O count"])
    formula_array = feature_arrays["Sum formula"]
    positions = _aligned_feature_positions(list(data.index), list(filtered_data.index))
    aligned_dbe = dbe_array[positions]
    aligned_ocount = ocount_array[positions]
    aligned_formula = formula_array[positions]
    print("Computing H/C and O/C for all features …")
    hc_values, oc_values = compute_hc_oc(aligned_formula, aligned_dbe, aligned_ocount)
    vk_lookup = {}
    for i, feature_name in enumerate(filtered_data.index.astype(str)):
        if feature_name not in vk_lookup:
            vk_lookup[feature_name] = (
                hc_values[i],
                oc_values[i],
                str(aligned_formula[i]),
                aligned_dbe[i],
                aligned_ocount[i],
            )
    n_valid_vk = sum(1 for value in vk_lookup.values() if not np.isnan(value[0]))
    print(f"  {n_valid_vk} / {len(vk_lookup)} unique features with valid H/C & O/C")

    feature_metadata = {
        column: pd.Series(feature_arrays[column], index=data.index)
        for column in ("DBE", "O count")
    }
    return OriginalWorkflowData(
        filtered_data=filtered_data,
        filtered_metadata=filtered_metadata,
        sample_column=sample_column,
        numeric_metadata_columns=numeric_metadata_columns,
        feature_metadata=feature_metadata,
        formula_array=aligned_formula,
        dbe_array=aligned_dbe,
        ocount_array=aligned_ocount,
        vk_lookup=vk_lookup,
    )


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


def compute_hc_oc(feature_metadata, dbe_array=None, ocount_array=None):
    if isinstance(feature_metadata, pd.DataFrame):
        result = pd.DataFrame(index=feature_metadata.index, columns=["H/C", "O/C"], dtype=float)
        for feature, row in feature_metadata.iterrows():
            carbon, hydrogen, oxygen = parse_formula(row.get("Sum formula"))
            if not pd.isna(carbon):
                result.loc[feature] = [hydrogen / carbon, oxygen / carbon]
            elif not pd.isna(row.get("DBE")) and not pd.isna(row.get("O count")):
                result.loc[feature] = [2 * (2 - float(row["DBE"])), float(row["O count"])]
        return result

    formula_array = feature_metadata
    hc = np.full(len(formula_array), np.nan, dtype=float)
    oc = np.full(len(formula_array), np.nan, dtype=float)
    for index, formula in enumerate(formula_array):
        if formula is not None and not (isinstance(formula, float) and np.isnan(formula)):
            formula_string = str(formula).strip()
            if formula_string and formula_string.lower() not in ("nan", "none", ""):
                carbon, hydrogen, oxygen = parse_formula(formula_string)
                if not np.isnan(carbon) and carbon > 0:
                    hc[index] = hydrogen / carbon
                    oc[index] = oxygen / carbon
                    continue
        dbe = dbe_array[index]
        oxygen_count = ocount_array[index]
        if not (np.isnan(dbe) or np.isnan(oxygen_count)):
            hydrogen_relative = 2.0 * (2.0 - float(dbe))
            if hydrogen_relative >= 0:
                hc[index] = hydrogen_relative
                oc[index] = float(oxygen_count)
    return hc, oc
