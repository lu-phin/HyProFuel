# -*- coding: utf-8 -*-
"""
Interactive PLS Regression Dashboard (Leakage-safe CV) - Condensed layout

Includes:
- mean-center scaling option for X and Y
- automatic evaluation of key metadata targets
- leakage-safe model selection
- automatic saving of prediction and coefficient plots
- full summary CSV + slide-ready summary CSV
- full trials CSV for manual model comparison
- existing Dash dashboard preserved
"""

import re
import itertools
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (
    KFold, RepeatedKFold, LeaveOneOut, GroupKFold
)
from sklearn.metrics import r2_score, mean_squared_error

import plotly.graph_objs as go
import plotly.io as pio

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import tkinter as tk
from tkinter import ttk

warnings.filterwarnings("ignore")

# ----------------------------
# Files
# ----------------------------
DATA_FILE = "FKTv2-onlySamples_Reduced(viaTrendlineHigher0-6).csv"
METADATA_FILE = "FKT-metadata_Histogram.csv"

EVAL_TARGETS = [
    "MCR [wt %]",
    "O [wt %]",
    "Viscosity [cSt]",
    "Carbonyl number [mol/kg]",
    "CAN [mg KOH/g]",
    "SG 60/60 [°F]",
]

# Restricted search space for automated evaluation
EVAL_X_TRANSFORMS = ["log10"]
EVAL_X_SCALINGS = ["center"]
EVAL_Y_SCALINGS = ["center"]

EVAL_OUTPUT_DIR = Path("pls_evaluation_outputs")
PLOT_DIR = EVAL_OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Van Krevelen regions
# ----------------------------
VK_REGIONS = {
    "Lipids":          dict(x=[0.0, 0.3, 0.3, 0.0, 0.0], y=[1.5, 1.5, 2.2, 2.2, 1.5]),
    "Proteins":        dict(x=[0.3, 0.7, 0.7, 0.3, 0.3], y=[1.5, 1.5, 2.2, 2.2, 1.5]),
    "Carbohydrates":   dict(x=[0.6, 1.2, 1.2, 0.6, 0.6], y=[1.5, 1.5, 2.5, 2.5, 1.5]),
    "Lignin":          dict(x=[0.1, 0.67, 0.67, 0.1, 0.1], y=[0.7, 0.7, 1.5, 1.5, 0.7]),
    "Tannins":         dict(x=[0.5, 1.1, 1.1, 0.5, 0.5], y=[0.5, 0.5, 1.5, 1.5, 0.5]),
    "Condensed Arom.": dict(x=[0.0, 0.67, 0.67, 0.0, 0.0], y=[0.2, 0.2, 0.7, 0.7, 0.2]),
}

VK_REGION_COLORS = {
    "Lipids":          ("rgba(70,130,180,0.1)",  "#4682B4"),
    "Proteins":        ("rgba(60,179,113,0.1)",   "#3CB371"),
    "Carbohydrates":   ("rgba(255,165,0,0.1)",    "#FFA500"),
    "Lignin":          ("rgba(160,82,45,0.1)",    "#A0522D"),
    "Tannins":         ("rgba(148,103,189,0.1)",  "#9467BD"),
    "Condensed Arom.": ("rgba(214,39,40,0.1)",    "#D62728"),
}

# ----------------------------
# Utilities
# ----------------------------
def scale_data(df, method="autoscale"):
    if method in ["zscore", "autoscale"]:
        return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns, index=df.index)
    if method in ["center", "mean_center"]:
        return pd.DataFrame(StandardScaler(with_std=False).fit_transform(df), columns=df.columns, index=df.index)
    if method == "minmax":
        return pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)
    if method == "maxabs":
        return pd.DataFrame(MaxAbsScaler().fit_transform(df), columns=df.columns, index=df.index)
    if method == "robust":
        return pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns, index=df.index)
    if method == "log":
        return np.log1p(df)
    if method == "normalize":
        return pd.DataFrame(Normalizer().fit_transform(df), columns=df.columns, index=df.index)
    if method == "none":
        return df.copy()
    raise ValueError("Unknown scaling method.")

def transform_data(df, method="none"):
    if method == "sqrt":
        return np.sqrt(df)
    if method == "log10":
        return np.log10(df + 1)
    if method == "log2":
        return np.log2(df + 1)
    if method == "cuberoot":
        return np.cbrt(df)
    if method == "none":
        return df
    raise ValueError("Unknown transform method.")

def make_scaler(method: str):
    if method in ["zscore", "autoscale"]:
        return StandardScaler()
    if method in ["center", "mean_center"]:
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

def apply_x_transform_numpy(X: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return X
    if method == "sqrt":
        return np.sqrt(np.clip(X, 0, None))
    if method == "log10":
        return np.log10(np.clip(X, 0, None) + 1.0)
    if method == "log2":
        return np.log2(np.clip(X, 0, None) + 1.0)
    if method == "cuberoot":
        return np.cbrt(X)
    raise ValueError(f"Unknown transform method: {method}")

def calculate_vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips

def save_prediction_plot(target_name, y_true, y_pred_cal, y_pred_cv, r2_cal, r2_cv,
                         file_stem_suffix=""):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(target_name)).strip("_")
    if file_stem_suffix:
        safe = f"{safe}_{file_stem_suffix}"
    fig, ax = plt.subplots(figsize=(6.5, 6))
    y_true = np.asarray(y_true, dtype=float)
    y_pred_cal = np.asarray(y_pred_cal, dtype=float)
    y_pred_cv = np.asarray(y_pred_cv, dtype=float)

    ax.scatter(
        y_true, y_pred_cal, s=40, alpha=0.85, label="Calibration",
        color="#636EFA", marker="o", edgecolors="black", linewidths=0.5
    )
    ax.scatter(
        y_true, y_pred_cv, s=55, alpha=0.95, label="CV",
        color="#EF553B", marker="^", edgecolors="black", linewidths=0.8
    )

    all_vals = np.concatenate([y_true, y_pred_cal, y_pred_cv])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) > 0:
        mn, mx = float(np.min(all_vals)), float(np.max(all_vals))
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{target_name}\nR² cal={r2_cal:.3f} | R² CV={r2_cv:.3f}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fname = PLOT_DIR / f"{safe}_prediction.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(fname)

def save_coefficient_plot(target_name, feature_names, reg_coef, top_n=25):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(target_name)).strip("_")
    feature_names = np.asarray(feature_names)
    reg_coef = np.asarray(reg_coef, dtype=float)
    idx = np.argsort(np.abs(reg_coef))[::-1][:min(top_n, len(reg_coef))]
    idx = idx[::-1]
    labels = feature_names[idx]
    vals = reg_coef[idx]
    colors = ["royalblue" if v >= 0 else "tomato" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(labels))))
    ax.barh(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Regression coefficient")
    ax.set_title(f"{target_name} — Top coefficients")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    fname = PLOT_DIR / f"{safe}_coefficients.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(fname)

def leakage_safe_cv_predict_pls(
    X_df, y_series, n_components, x_scale_method, y_scale_method,
    x_transform_method, cv_method="kfold", cv_folds=5, cv_repeats=3,
    group_labels=None, random_state=42,
):
    X = X_df.values.astype(float)
    y = y_series.values.astype(float).reshape(-1, 1)
    n_samples, _ = X.shape

    if n_samples < 3:
        raise ValueError("Need at least 3 samples for CV.")

    k = int(min(max(int(cv_folds), 2), n_samples))
    if cv_method == "loo":
        splitter = LeaveOneOut()
    elif cv_method == "repeatedkfold":
        splitter = RepeatedKFold(n_splits=k, n_repeats=int(cv_repeats), random_state=random_state)
    elif cv_method == "groupkfold":
        if group_labels is None:
            raise ValueError("group_labels required for GroupKFold.")
        groups = np.asarray(group_labels)
        n_groups = len(np.unique(groups))
        k = min(k, n_groups)
        if k < 2:
            raise ValueError(f"Need at least 2 groups for GroupKFold (have {n_groups}).")
        splitter = GroupKFold(n_splits=k)
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)

    y_pred_oof = np.full((n_samples, 1), np.nan, dtype=float)
    split_groups = group_labels if cv_method == "groupkfold" else None

    for train_idx, test_idx in splitter.split(X, y, groups=split_groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        X_train = apply_x_transform_numpy(X_train, x_transform_method)
        X_test = apply_x_transform_numpy(X_test, x_transform_method)

        x_scaler = make_scaler(x_scale_method)
        if x_scaler is not None:
            X_train_scaled = x_scaler.fit_transform(X_train)
            X_test_scaled = x_scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test

        y_scaler = make_scaler(y_scale_method)
        y_train_scaled = y_scaler.fit_transform(y_train) if y_scaler else y_train

        n_comp_fold = max(1, int(min(n_components, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])))
        pls = PLSRegression(n_components=n_comp_fold, scale=False)
        pls.fit(X_train_scaled, y_train_scaled)

        y_pred_scaled = pls.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled) if y_scaler else y_pred_scaled
        y_pred_oof[test_idx] = y_pred

    return y_pred_oof.ravel()

def evaluate_target_models(
    X_all, meta_df, sample_col, target_name,
    x_transforms=("none", "log10", "log2"),
    x_scalings=("autoscale", "center", "minmax", "none"),
    y_scalings=("none",),
    max_lv=4,
    cv_method="kfold",
    cv_folds=5,
    cv_repeats=3,
):
    if target_name not in meta_df.columns:
        return None

    fmeta_idx = meta_df.set_index(sample_col)
    common_samples = X_all.index.intersection(fmeta_idx.index)
    X = X_all.loc[common_samples].copy()
    y = pd.to_numeric(fmeta_idx.loc[common_samples, target_name], errors="coerce")

    valid = ~y.isna()
    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 3:
        return {"target": target_name, "modelable": False, "reason": "Too few samples", "trial_rows": []}

    best = None
    best_score = (-np.inf, np.inf)
    trial_rows = []

    max_lv = int(min(max_lv, len(X) - 1, X.shape[1]))
    if max_lv < 1:
        return {"target": target_name, "modelable": False, "reason": "No feasible components", "trial_rows": []}

    for x_t, x_s, y_s, n_comp in itertools.product(
        x_transforms, x_scalings, y_scalings, range(1, max_lv + 1)
    ):
        try:
            y_cv = leakage_safe_cv_predict_pls(
                X_df=X, y_series=y, n_components=n_comp,
                x_scale_method=x_s, y_scale_method=y_s,
                x_transform_method=x_t, cv_method=cv_method,
                cv_folds=cv_folds, cv_repeats=cv_repeats, random_state=42,
            )
            r2_cv = r2_score(y.values, y_cv)
            rmse_cv = np.sqrt(mean_squared_error(y.values, y_cv))
            if np.isnan(r2_cv) or np.isinf(r2_cv):
                continue

            X_tr = pd.DataFrame(apply_x_transform_numpy(X.values, x_t), index=X.index, columns=X.columns)
            x_scaler = make_scaler(x_s)
            if x_scaler is not None:
                X_tr = pd.DataFrame(x_scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)

            y_scaler = make_scaler(y_s)
            y_fit = y.values.reshape(-1, 1)
            if y_scaler is not None:
                y_fit = y_scaler.fit_transform(y_fit)

            pls = PLSRegression(n_components=min(n_comp, len(X_tr) - 1, X_tr.shape[1]), scale=False)
            pls.fit(X_tr.values, y_fit)

            y_pred_cal_s = pls.predict(X_tr.values)
            if y_scaler is not None:
                y_pred_cal = y_scaler.inverse_transform(y_pred_cal_s).ravel()
            else:
                y_pred_cal = y_pred_cal_s.ravel()

            r2_cal = r2_score(y.values, y_pred_cal)
            rmse_cal = np.sqrt(mean_squared_error(y.values, y_pred_cal))

            plot_suffix = f"LV{n_comp}_{x_t}_{x_s}_{y_s}"
            pred_plot_path = save_prediction_plot(
                target_name,
                y.values.astype(float),
                y_pred_cal,
                y_cv,
                r2_cal,
                r2_cv,
                file_stem_suffix=plot_suffix,
            )
            
            top_idx = np.argsort(np.abs(pls.coef_.ravel()))[::-1][:20]
            feat = np.asarray(X.columns)
            coefs = np.asarray(pls.coef_.ravel(), dtype=float)

            trial_rows.append({
                "Target": target_name,
                "Best LV": int(n_comp),
                "X transform": x_t,
                "X scaling": x_s,
                "Y scaling": y_s,
                "R2 cal": float(r2_cal),
                "RMSE cal": float(rmse_cal),
                "R2 CV": float(r2_cv),
                "RMSE CV": float(rmse_cv),
                "n_samples": int(len(X)),
                "n_features": int(X.shape[1]),
                "Modelable": "Yes",
                "Prediction plot": pred_plot_path,
                "Top features": ", ".join(map(str, feat[top_idx][:10])),
                "Top feature coefficients": ", ".join([f"{v:.6g}" for v in coefs[top_idx][:10]]),
            })

            score = (r2_cv, -rmse_cv)
            if score > best_score:
                best_score = score
                best = {
                    "target": target_name,
                    "modelable": True,
                    "best_lv": n_comp,
                    "x_transform": x_t,
                    "x_scale": x_s,
                    "y_scale": y_s,
                    "r2_cal": float(r2_cal),
                    "rmse_cal": float(rmse_cal),
                    "r2_cv": float(r2_cv),
                    "rmse_cv": float(rmse_cv),
                    "n_samples": len(X),
                    "n_features": X.shape[1],
                    "feature_names": X.columns.tolist(),
                    "reg_coef": pls.coef_.ravel(),
                    "y_true": y.values.astype(float),
                    "y_pred_cal": y_pred_cal,
                    "y_pred_cv": y_cv,
                    "prediction_plot": pred_plot_path,
                }
        except Exception:
            continue

    if best is None:
        return {
            "target": target_name,
            "modelable": False,
            "reason": "No stable model found",
            "trial_rows": trial_rows,
        }

    top_idx = np.argsort(np.abs(best["reg_coef"]))[::-1][:20]
    feat = np.asarray(best["feature_names"])
    coefs = np.asarray(best["reg_coef"], dtype=float)

    best["top_features"] = pd.DataFrame({
        "feature": feat[top_idx],
        "coef": coefs[top_idx],
        "abs_coef": np.abs(coefs[top_idx]),
        "direction": np.where(coefs[top_idx] >= 0, "positive", "negative"),
    })

    best["prediction_plot"] = save_prediction_plot(
        target_name, best["y_true"], best["y_pred_cal"], best["y_pred_cv"],
        best["r2_cal"], best["r2_cv"],
    )
    best["coefficient_plot"] = save_coefficient_plot(target_name, best["feature_names"], best["reg_coef"])
    best["trial_rows"] = trial_rows
    return best

# ----------------------------
# Load data
# ----------------------------
data = pd.read_csv(DATA_FILE, index_col=0)
meta = pd.read_csv(METADATA_FILE)

print(f"Raw data shape: {data.shape}  |  index unique: {data.index.is_unique}")

FEATURE_META_COLS = ["DBE", "O count", "Sum formula"]
feat_meta_arrays = {}

for col in FEATURE_META_COLS:
    matched = [c for c in data.columns if c.strip().lower() == col.strip().lower()]
    if matched:
        actual = matched[0]
        feat_meta_arrays[col] = data[actual].values.copy()
        data = data.drop(columns=[actual])
        print(f"  Extracted '{actual}' → '{col}'")
    else:
        print(f"  Warning: '{col}' not found.")
        feat_meta_arrays[col] = np.full(len(data), np.nan)

def _to_float(arr):
    out = np.full(len(arr), np.nan)
    for i, v in enumerate(arr):
        try:
            out[i] = float(v)
        except Exception:
            pass
    return out

_dbe_arr = _to_float(feat_meta_arrays["DBE"])
_ocount_arr = _to_float(feat_meta_arrays["O count"])
_formula_arr = feat_meta_arrays["Sum formula"]

feature_metadata = {}
for col in ["DBE", "O count"]:
    feature_metadata[col] = pd.Series(feat_meta_arrays[col], index=data.index)

sample_col = None
for possible in ["Sample", "sample", "ID", "id", "Name"]:
    if possible in meta.columns:
        sample_col = possible
        break
if sample_col is None:
    raise ValueError("Could not find sample ID column in metadata!")

meta[sample_col] = meta[sample_col].astype(str).str.strip()
data.index = data.index.astype(str).str.strip()
data.columns = data.columns.astype(str).str.strip()

valid_samples = set(meta[sample_col])
filtered_data = data.loc[:, data.columns.isin(valid_samples)]
filtered_meta = meta[meta[sample_col].isin(filtered_data.columns)].copy()

def sanity_check_orientation_and_alignment(filtered_data, filtered_meta, sample_col):
    print("\n=== SANITY CHECK ===")
    print(f"Raw feature table shape (features x samples): {filtered_data.shape}")
    X_check = (
        filtered_data.apply(pd.to_numeric, errors="coerce")
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .T
    )
    print(f"Model matrix shape after transpose (samples x features): {X_check.shape}")
    print(f"Number of samples: {X_check.shape[0]}")
    print(f"Number of features: {X_check.shape[1]}")

    meta_ids = filtered_meta[sample_col].astype(str).str.strip().tolist()
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

sanity_check_orientation_and_alignment(filtered_data, filtered_meta, sample_col)

all_samples = list(filtered_data.columns)

def select_samples_popup(sample_names, title="Select samples to include", default_select_all=True):
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
        for s in sample_names:
            if (not filt) or (filt in s.lower()):
                listbox.insert(tk.END, s)
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
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=(0, 10))
    ttk.Button(btn_frame, text="Select all", command=select_all).pack(side="left")
    ttk.Button(btn_frame, text="Select none", command=select_none).pack(side="left", padx=(10, 0))
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

    ttk.Button(btn_frame, text="OK (start app)", command=on_ok).pack(side="right")
    ttk.Button(btn_frame, text="Cancel (include all)", command=on_cancel).pack(side="right", padx=(0, 10))
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    refresh_list()
    root.mainloop()
    return result["samples"]

chosen_samples = select_samples_popup(
    all_samples,
    title="PLS Dashboard – Select samples to include (default: all)",
    default_select_all=True,
)
chosen_set = set(chosen_samples)
filtered_data = filtered_data.loc[:, filtered_data.columns.isin(chosen_set)]
filtered_meta = filtered_meta[filtered_meta[sample_col].isin(chosen_set)].copy()
print(f"Selected {filtered_data.shape[1]} / {len(all_samples)} samples.")

meta_cols = [c for c in filtered_meta.columns if c != sample_col]
numeric_meta_cols = []
for col in meta_cols:
    try:
        pd.to_numeric(filtered_meta[col], errors="raise")
        numeric_meta_cols.append(col)
    except Exception:
        pass
print(f"Found {len(numeric_meta_cols)} numeric metadata columns for Y variables")

_orig_index = list(data.index)
_seen = {}
_positions = []
for feat in filtered_data.index:
    count = _seen.get(feat, 0)
    positions = [i for i, x in enumerate(_orig_index) if x == feat]
    _positions.append(positions[count] if count < len(positions) else positions[-1])
    _seen[feat] = count + 1
_positions = np.array(_positions)

all_dbe_aligned = _dbe_arr[_positions]
all_ocount_aligned = _ocount_arr[_positions]
all_formula_aligned = _formula_arr[_positions]

print("Computing H/C and O/C for all features …")

def parse_formula(formula: str):
    counts = {}
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if element:
            counts[element] = int(count) if count else 1
    C = counts.get("C", np.nan)
    H = counts.get("H", np.nan)
    O = counts.get("O", 0)
    if np.isnan(C) or C == 0:
        return np.nan, np.nan, np.nan
    return float(C), float(H), float(O)

def compute_hc_oc(formula_array, dbe_array, ocount_array):
    n = len(formula_array)
    hc = np.full(n, np.nan)
    oc = np.full(n, np.nan)
    for i in range(n):
        f = formula_array[i]
        if f is not None and not (isinstance(f, float) and np.isnan(f)):
            fs = str(f).strip()
            if fs and fs.lower() not in ("nan", "none", ""):
                C, H, O = parse_formula(fs)
                if not np.isnan(C) and C > 0:
                    hc[i] = H / C
                    oc[i] = O / C
                    continue
        dbe = dbe_array[i]
        o = ocount_array[i]
        if not (np.isnan(dbe) or np.isnan(o)):
            h_rel = 2.0 * (2.0 - float(dbe))
            if h_rel >= 0:
                hc[i] = h_rel
                oc[i] = float(o)
    return hc, oc

_hc_all, _oc_all = compute_hc_oc(all_formula_aligned, all_dbe_aligned, all_ocount_aligned)
_all_feature_names = list(filtered_data.index.astype(str))
_vk_lookup = {}
for i, fname in enumerate(_all_feature_names):
    if fname not in _vk_lookup:
        _vk_lookup[fname] = (
            _hc_all[i], _oc_all[i],
            str(all_formula_aligned[i]),
            all_dbe_aligned[i],
            all_ocount_aligned[i],
        )

n_valid_vk = sum(1 for v in _vk_lookup.values() if not np.isnan(v[0]))
print(f"  {n_valid_vk} / {len(_vk_lookup)} unique features with valid H/C & O/C")

# ----------------------------
# Styling
# ----------------------------
color_schemes = {
    "Default": ["#636EFA", "#EF553B", "#00CC96"],
    "Warm":    ["#FF6B6B", "#FFA500", "#FFD700"],
    "Cool":    ["#4169E1", "#00CED1", "#7B68EE"],
    "Earth":   ["#8B4513", "#D2691E", "#DEB887"],
    "Pastel":  ["#FFB6C1", "#98D8C8", "#F7DC6F"],
    "Vibrant": ["#E74C3C", "#9B59B6", "#3498DB"],
    "Zissou":  ["#F21A00", "#EBCC2A", "#3B9AB2"],
}
zissou_colorscale = [[0.0, "#3B9AB2"], [0.5, "#EBCC2A"], [1.0, "#F21A00"]]

try:
    import kaleido  # noqa
    KALEIDO_AVAILABLE = True
    print("Kaleido available for image export")
except ImportError:
    KALEIDO_AVAILABLE = False
    print("WARNING: Kaleido not installed.")

CTRL_STYLE = {"width": "24%", "display": "inline-block", "padding": "6px", "verticalAlign": "top"}
SECTION_BOX = {"background-color": "#f6f6f6", "padding": "10px", "margin": "10px 0", "border-radius": "6px"}

def build_vk_legend():
    items = []
    for name, (_, solid_hex) in VK_REGION_COLORS.items():
        items.append(html.Span([
            html.Span(style={
                "display": "inline-block", "width": "16px", "height": "16px",
                "background-color": solid_hex, "opacity": "0.7",
                "border": f"2px solid {solid_hex}", "border-radius": "3px",
                "margin-right": "5px", "vertical-align": "middle",
            }),
            html.Span(name, style={"font-size": "12px", "margin-right": "16px", "vertical-align": "middle"}),
        ]))
    return html.Div([
        html.Span("Regions: ", style={"font-size": "12px", "font-weight": "bold", "margin-right": "8px"}),
        *items,
    ], style={"background-color": "#fff", "border": "1px solid #ccc", "border-radius": "5px",
              "padding": "7px 14px", "margin": "6px 0", "display": "inline-block"})

VK_LEGEND_DIV = build_vk_legend()

# ----------------------------
# App layout
# ----------------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Interactive PLS Regression Dashboard (Leakage-safe CV)"),

    html.Div([
        html.Div([html.Label("Y variable:"),
            dcc.Dropdown(id="y_variable",
                options=[{"label": c, "value": c} for c in numeric_meta_cols],
                value=numeric_meta_cols[0] if numeric_meta_cols else None),
        ], style=CTRL_STYLE),
        html.Div([html.Label("Components:"),
            dcc.Dropdown(id="n_components",
                options=[{"label": str(i), "value": i} for i in range(1, 11)], value=3),
        ], style=CTRL_STYLE),
        html.Div([html.Label("X scaling:"),
            dcc.Dropdown(id="x_scale", options=[
                {"label": "Autoscale", "value": "autoscale"},
                {"label": "Mean Center Only", "value": "center"},
                {"label": "Min-Max", "value": "minmax"},
                {"label": "Max-Abs", "value": "maxabs"},
                {"label": "Robust", "value": "robust"},
                {"label": "Normalize", "value": "normalize"},
                {"label": "None", "value": "none"},
            ], value="autoscale"),
        ], style=CTRL_STYLE),
        html.Div([html.Label("Y scaling:"),
            dcc.Dropdown(id="y_scale", options=[
                {"label": "Autoscale", "value": "autoscale"},
                {"label": "Mean Center Only", "value": "center"},
                {"label": "Min-Max", "value": "minmax"},
                {"label": "Max-Abs", "value": "maxabs"},
                {"label": "Robust", "value": "robust"},
                {"label": "None", "value": "none"},
            ], value="autoscale"),
        ], style=CTRL_STYLE),
    ], style=SECTION_BOX),

    html.Div([
        html.Div([html.Label("X transform:"),
            dcc.Dropdown(id="x_transform",
                options=[{"label": m, "value": m} for m in ["none", "sqrt", "log10", "log2", "cuberoot"]],
                value="none"),
        ], style=CTRL_STYLE),
        html.Div([html.Label("Score X axis:"),
            dcc.Dropdown(id="comp_x",
                options=[{"label": f"LV{i+1}", "value": i} for i in range(5)], value=0),
        ], style=CTRL_STYLE),
        html.Div([html.Label("Score Y axis:"),
            dcc.Dropdown(id="comp_y",
                options=[{"label": f"LV{i+1}", "value": i} for i in range(5)], value=1),
        ], style=CTRL_STYLE),
        html.Div([html.Label("Color scheme:"),
            dcc.Dropdown(id="color_scheme",
                options=[{"label": k, "value": k} for k in color_schemes], value="Default"),
        ], style=CTRL_STYLE),
    ], style=SECTION_BOX),

    html.Div([
        html.Div([html.Label("Top N coefficients (plot):"),
            dcc.Slider(id="vip_threshold", min=5, max=100, step=10, value=50,
                marks={i: str(i) for i in range(5, 51, 5)}),
        ], style=CTRL_STYLE),
        html.Div([html.Label("Label top %:"),
            dcc.Slider(id="label_percent", min=0, max=20, step=1, value=5,
                marks={i: f"{i}%" for i in range(0, 21, 5)}),
        ], style=CTRL_STYLE),
        html.Div([html.Label("Loading color:"),
            dcc.Dropdown(id="loading_color", options=[
                {"label": "VIP", "value": "VIP"}, {"label": "DBE", "value": "DBE"},
                {"label": "O count", "value": "O count"},
            ], value="VIP"),
        ], style=CTRL_STYLE),
        html.Div([html.Label("Loading size:"),
            dcc.Dropdown(id="loading_size", options=[
                {"label": "VIP", "value": "VIP"}, {"label": "DBE", "value": "DBE"},
                {"label": "O count", "value": "O count"}, {"label": "Uniform", "value": "none"},
            ], value="VIP"),
        ], style=CTRL_STYLE),
    ], style=SECTION_BOX),

    html.Div([
        html.Div([html.Label("CV method:"),
            dcc.Dropdown(id="cv_method", options=[
                {"label": "K-Fold", "value": "kfold"},
                {"label": "Repeated K-Fold", "value": "repeatedkfold"},
                {"label": "Leave-One-Out", "value": "loo"},
                {"label": "Group K-Fold", "value": "groupkfold"},
            ], value="kfold"),
        ], style=CTRL_STYLE),
        html.Div([html.Label("CV folds (K):"),
            dcc.Slider(id="cv_folds", min=2, max=10, step=1, value=5,
                marks={i: str(i) for i in range(2, 11)}),
        ], style=CTRL_STYLE),
        html.Div([html.Label("CV repeats:"),
            dcc.Slider(id="cv_repeats", min=1, max=10, step=1, value=3,
                marks={i: str(i) for i in range(1, 11)}),
        ], style=CTRL_STYLE),
        html.Div([html.Label("Group column:"),
            dcc.Dropdown(id="cv_group_col",
                options=[{"label": c, "value": c} for c in meta_cols],
                value=None, placeholder="(Only for GroupKFold)"),
        ], style=CTRL_STYLE),
    ], style=SECTION_BOX),

    html.Div([
        html.Div([
            html.Label("Van Krevelen |coef| threshold:"),
            dcc.Slider(id="vk_coef_threshold", min=0.0, max=5.0, step=0.1, value=1.0,
                marks={v: str(v) for v in [0, 0.5, 1, 1.5, 2, 3, 4, 5]}),
        ], style={"width": "74%", "display": "inline-block", "padding": "6px"}),
        html.Div([
            html.Label("VK marker size:"),
            dcc.Dropdown(id="vk_size_by", options=[
                {"label": "|Coef|", "value": "coef"},
                {"label": "O count", "value": "ocount"},
                {"label": "DBE", "value": "dbe"},
                {"label": "Uniform", "value": "uniform"},
            ], value="coef"),
        ], style={"width": "24%", "display": "inline-block", "padding": "6px", "verticalAlign": "top"}),
    ], style=SECTION_BOX),

    html.Div(id="model-stats", style={
        "padding": "10px", "background-color": "#f0f0f0",
        "margin": "10px 0", "border-radius": "6px", "font-size": "14px",
    }),

    html.Div([
        html.Div([dcc.Graph(id="score-plot", style={"height": "600px"})],
                 style={"width": "50%", "display": "inline-block"}),
        html.Div([dcc.Graph(id="loading-plot", style={"height": "600px"})],
                 style={"width": "50%", "display": "inline-block"}),
    ]),

    html.Div([
        html.Div([dcc.Graph(id="prediction-plot", style={"height": "600px"})],
                 style={"width": "50%", "display": "inline-block"}),
        html.Div([dcc.Graph(id="vip-plot", style={"height": "600px"})],
                 style={"width": "50%", "display": "inline-block"}),
    ]),

    html.Div([
        html.H3("Van Krevelen Plot – Features with |coef| > threshold", style={"margin-bottom": "4px"}),
        html.Div([
            html.Span("● Positive coef", style={"color": "blue", "font-weight": "bold",
                                                 "margin-right": "24px", "font-size": "13px"}),
            html.Span("● Negative coef", style={"color": "red", "font-weight": "bold",
                                                 "font-size": "13px"}),
        ], style={"margin-bottom": "6px"}),
        VK_LEGEND_DIV,
        dcc.Graph(id="vk-plot", style={"height": "650px"}),
    ], style={**SECTION_BOX, "margin-top": "16px"}),

    html.Div([
        html.H3("Download Options"),
        html.Div([
            html.Div([html.Label("Plot:"),
                dcc.Dropdown(id="download_plot_select", options=[
                    {"label": "Score Plot", "value": "score"},
                    {"label": "Loading Plot", "value": "loading"},
                    {"label": "Prediction Plot", "value": "prediction"},
                    {"label": "Regression Coefficients Plot", "value": "vip"},
                    {"label": "Van Krevelen Plot", "value": "vk"},
                ], value="score"),
            ], style={"width": "24%", "display": "inline-block", "padding": "6px"}),
            html.Div([html.Label("DPI:"),
                dcc.Dropdown(id="download_dpi", options=[
                    {"label": "150", "value": 150}, {"label": "300", "value": 300},
                    {"label": "600", "value": 600},
                ], value=300),
            ], style={"width": "24%", "display": "inline-block", "padding": "6px"}),
            html.Div([html.Label("Format:"),
                dcc.Dropdown(id="download_format", options=[
                    {"label": "PNG", "value": "png"}, {"label": "SVG", "value": "svg"},
                    {"label": "PDF", "value": "pdf"},
                ], value="png"),
            ], style={"width": "24%", "display": "inline-block", "padding": "6px"}),
            html.Div([
                html.Button(
                    "Download Plot", id="download_button", n_clicks=0,
                    style={"margin-top": "22px", "padding": "10px 16px",
                           "background-color": "#007bff", "color": "white",
                           "border": "none", "border-radius": "5px", "cursor": "pointer"},
                    disabled=not KALEIDO_AVAILABLE
                ),
                html.Br(),
                html.Button("Download All Coefficients (CSV)", id="download_vip_button",
                    n_clicks=0,
                    style={"margin-top": "8px", "padding": "10px 16px",
                           "background-color": "#28a745", "color": "white",
                           "border": "none", "border-radius": "5px", "cursor": "pointer"}),
            ], style={"width": "24%", "display": "inline-block", "padding": "6px", "verticalAlign": "top"}),
        ]),

        html.Div([
            html.Div([html.Label("Top N per direction (positive / negative):"),
                dcc.Slider(id="top_n_posneg", min=5, max=100, step=5, value=20,
                    marks={i: str(i) for i in range(5, 101, 15)}),
            ], style={"width": "74%", "display": "inline-block", "padding": "6px", "verticalAlign": "middle"}),
            html.Div([
                html.Button("Download Top Pos / Neg (CSV)", id="download_posneg_button",
                    n_clicks=0,
                    style={"margin-top": "22px", "padding": "10px 16px",
                           "background-color": "#e67e22", "color": "white",
                           "border": "none", "border-radius": "5px", "cursor": "pointer"}),
                html.P("Exports top N positive and top N negative features sorted by |coef|.",
                       style={"font-size": "11px", "color": "#666", "margin-top": "4px"}),
            ], style={"width": "24%", "display": "inline-block", "padding": "6px", "verticalAlign": "top"}),
        ], style={"margin-top": "8px", "border-top": "1px solid #ddd", "padding-top": "8px"}),

        html.Div(id="download-status", style={"padding": "10px", "color": "red"}),
        dcc.Download(id="download_image"),
        dcc.Download(id="download_vip_data"),
        dcc.Download(id="download_posneg_data"),
    ], style={"background-color": "#f0f0f0", "padding": "12px", "margin": "12px 0", "border-radius": "6px"}),

    dcc.Store(id="pls-data"),
    dcc.Store(id="current-score-fig"),
    dcc.Store(id="current-loading-fig"),
    dcc.Store(id="current-prediction-fig"),
    dcc.Store(id="current-vip-fig"),
    dcc.Store(id="current-vk-fig"),
])

@app.callback(
    [
        Output("score-plot", "figure"),
        Output("loading-plot", "figure"),
        Output("prediction-plot", "figure"),
        Output("vip-plot", "figure"),
        Output("vk-plot", "figure"),
        Output("model-stats", "children"),
        Output("pls-data", "data"),
        Output("current-score-fig", "data"),
        Output("current-loading-fig", "data"),
        Output("current-prediction-fig", "data"),
        Output("current-vip-fig", "data"),
        Output("current-vk-fig", "data"),
    ],
    [
        Input("y_variable", "value"),
        Input("n_components", "value"),
        Input("x_scale", "value"),
        Input("y_scale", "value"),
        Input("x_transform", "value"),
        Input("comp_x", "value"),
        Input("comp_y", "value"),
        Input("color_scheme", "value"),
        Input("vip_threshold", "value"),
        Input("label_percent", "value"),
        Input("loading_color", "value"),
        Input("loading_size", "value"),
        Input("cv_method", "value"),
        Input("cv_folds", "value"),
        Input("cv_repeats", "value"),
        Input("cv_group_col", "value"),
        Input("vk_coef_threshold", "value"),
        Input("vk_size_by", "value"),
    ]
)
def update_plots(
    y_var, n_comp, x_scale_method, y_scale_method, x_transform_method,
    comp_x, comp_y, color_scheme, top_n_coef, label_percent,
    loading_color, loading_size,
    cv_method, cv_folds, cv_repeats, cv_group_col,
    vk_coef_threshold, vk_size_by,
):
    empty = go.Figure()

    if y_var is None:
        return empty, empty, empty, empty, empty, "Please select a Y variable", {}, {}, {}, {}, {}, {}

    fdata_num = (
        filtered_data.apply(pd.to_numeric, errors="coerce")
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
    )
    X = pd.DataFrame(transform_data(fdata_num.T, x_transform_method))
    X = X.dropna(axis=0)

    fmeta_idx = filtered_meta.set_index(sample_col)
    common_samples = X.index.intersection(fmeta_idx.index)
    X = X.loc[common_samples]
    y = fmeta_idx.loc[common_samples, y_var]
    valid_idx = ~y.isna()
    X, y = X.loc[valid_idx], y.loc[valid_idx]

    if len(X) < 3:
        msg = f"Not enough samples (need ≥3, have {len(X)})"
        return empty, empty, empty, empty, empty, msg, {}, {}, {}, {}, {}, {}

    X_scaled = scale_data(X, x_scale_method)
    y_scaled = scale_data(pd.DataFrame(y), y_scale_method).values.ravel()
    n_comp = max(1, int(min(n_comp, len(X) - 1, X_scaled.shape[1])))

    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X_scaled, y_scaled)
    y_pred_scaled = pls.predict(X_scaled).ravel()

    y_true_orig = y.values.astype(float)
    y_sf = make_scaler(y_scale_method)
    if y_sf:
        y_sf.fit(y_true_orig.reshape(-1, 1))
        y_pred_orig = y_sf.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    else:
        y_pred_orig = y_pred_scaled

    groups = None
    if cv_method == "groupkfold":
        if not cv_group_col:
            msg = "GroupKFold selected but no group column chosen."
            return empty, empty, empty, empty, empty, msg, {}, {}, {}, {}, {}, {}
        groups = fmeta_idx.loc[X.index, cv_group_col].astype(str).values

    try:
        y_cv_orig = leakage_safe_cv_predict_pls(
            X_df=X, y_series=y, n_components=n_comp,
            x_scale_method=x_scale_method, y_scale_method=y_scale_method,
            x_transform_method=x_transform_method,
            cv_method=cv_method, cv_folds=cv_folds, cv_repeats=cv_repeats,
            group_labels=groups, random_state=42,
        )
    except Exception as e:
        return empty, empty, empty, empty, empty, f"CV error: {e}", {}, {}, {}, {}, {}, {}

    r2_cal = r2_score(y_true_orig, y_pred_orig)
    rmse_cal = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2_cv = r2_score(y_true_orig, y_cv_orig)
    rmse_cv = np.sqrt(mean_squared_error(y_true_orig, y_cv_orig))

    vip_scores = calculate_vip(pls)
    T = pls.x_scores_
    P = pls.x_loadings_
    reg_coef = pls.coef_.ravel()
    feature_names = X.columns
    n_features = len(feature_names)

    pls_export_data = {
        "T": T.tolist(), "P": P.tolist(),
        "vip_scores": vip_scores.tolist(), "reg_coef": reg_coef.tolist(),
        "feature_names": feature_names.tolist(), "sample_names": X.index.tolist(),
        "y_true": y_true_orig.tolist(), "y_pred": y_pred_orig.tolist(),
        "y_cv": y_cv_orig.tolist(),
        "r2_cal": float(r2_cal), "rmse_cal": float(rmse_cal),
        "r2_cv": float(r2_cv), "rmse_cv": float(rmse_cv),
        "n_components": int(n_comp), "cv_method": cv_method,
        "cv_folds": int(cv_folds) if cv_folds is not None else None,
        "cv_repeats": int(cv_repeats) if cv_repeats is not None else None,
        "cv_group_col": cv_group_col,
    }

    colors = color_schemes.get(color_scheme, color_schemes["Default"])
    max_lv = T.shape[1] - 1
    cx = int(min(comp_x, max_lv))
    cy = int(min(comp_y, max_lv))

    score_fig = go.Figure()
    score_fig.add_trace(go.Scatter(
        x=T[:, cx], y=T[:, cy], mode="markers+text",
        marker=dict(size=10, color=colors[0], line=dict(width=1, color="black")),
        text=X.index, textposition="top center", textfont=dict(size=8),
        hovertemplate=(f"Sample: %{{text}}<br>LV{cx+1}: %{{x:.3f}}<br>"
                       f"LV{cy+1}: %{{y:.3f}}<extra></extra>"),
    ))
    score_fig.update_layout(
        title=f"Score Plot: {y_var}",
        xaxis_title=f"LV{cx+1}", yaxis_title=f"LV{cy+1}", height=600,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
    )

    loading_fig = go.Figure()
    top_lbl = set()
    if label_percent and label_percent > 0:
        n_lbl = max(1, int(np.ceil(n_features * label_percent / 100)))
        top_lbl = (set(np.argsort(np.abs(P[:, cx]))[-n_lbl:]) |
                   set(np.argsort(np.abs(P[:, cy]))[-n_lbl:]))

    c_vals = (vip_scores if loading_color == "VIP"
              else [feature_metadata[loading_color].get(f, 0) for f in feature_names]
              if loading_color in feature_metadata else [0] * n_features)
    s_vals = (vip_scores if loading_size == "VIP"
              else [feature_metadata[loading_size].get(f, 1) for f in feature_names]
              if loading_size != "none" and loading_size in feature_metadata
              else [1] * n_features)
    s_vals = np.asarray(s_vals, dtype=float)
    mn, mx = np.nanmin(s_vals), np.nanmax(s_vals)
    s_vals = 5 + 15 * (s_vals - mn) / (mx - mn) if mx != mn else np.full(n_features, 10.0)

    loading_fig.add_trace(go.Scatter(
        x=P[:, cx], y=P[:, cy], mode="markers+text",
        marker=dict(size=s_vals, color=c_vals, colorscale=zissou_colorscale,
                    showscale=True, colorbar=dict(title=loading_color),
                    line=dict(width=1, color="black"), opacity=0.85),
        text=[feature_names[i] if i in top_lbl else "" for i in range(n_features)],
        textposition="top center", textfont=dict(size=8),
        hovertext=[f"Feature: {feature_names[i]}<br>LV{cx+1}: {P[i, cx]:.3f}<br>"
                   f"LV{cy+1}: {P[i, cy]:.3f}<br>VIP: {vip_scores[i]:.2f}"
                   for i in range(n_features)],
        hoverinfo="text",
    ))
    loading_fig.update_layout(
        title=f"Loading Plot: {y_var}",
        xaxis_title=f"LV{cx+1} Loadings", yaxis_title=f"LV{cy+1} Loadings", height=600,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
    )

    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(
        x=y_true_orig, y=y_pred_orig, mode="markers",
        name="Calibration", marker=dict(size=8, color=colors[0]),
        hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"
    ))
    pred_fig.add_trace(go.Scatter(
        x=y_true_orig, y=y_cv_orig, mode="markers",
        name="Cross-validation (leakage-safe)",
        marker=dict(size=10, color=colors[1], symbol="triangle-up"),
        hovertemplate="Actual: %{x:.3f}<br>CV Predicted: %{y:.3f}<extra></extra>"
    ))

    mnv = float(np.nanmin([y_true_orig.min(), y_pred_orig.min(), y_cv_orig.min()]))
    mxv = float(np.nanmax([y_true_orig.max(), y_pred_orig.max(), y_cv_orig.max()]))
    pred_fig.add_trace(go.Scatter(
        x=[mnv, mxv], y=[mnv, mxv], mode="lines",
        name="1:1 line", line=dict(color="black", dash="dash")
    ))
    pred_fig.update_layout(
        title=f"Predicted vs Actual: {y_var}",
        xaxis_title=f"Actual {y_var}", yaxis_title=f"Predicted {y_var}",
        height=600, showlegend=True
    )

    coef_fig = go.Figure()
    top_n = int(min(top_n_coef, n_features))
    top_idx = np.argsort(np.abs(reg_coef))[::-1][:top_n]
    tc = reg_coef[top_idx]
    tn = [feature_names[i] for i in top_idx]
    coef_fig.add_trace(go.Bar(
        x=tn, y=tc,
        marker=dict(color=["blue" if c >= 0 else "red" for c in tc]),
        hovertemplate="Feature: %{x}<br>Regression Coefficient: %{y:.4f}<extra></extra>",
    ))
    coef_fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    coef_fig.update_layout(
        title=f"Regression Coefficients (Top {top_n} by |coef|): {y_var}",
        xaxis_title="Features", yaxis_title="Regression Coefficient",
        height=600, xaxis=dict(tickangle=-45)
    )

    thr = float(vk_coef_threshold or 1.0)
    above_mask = np.abs(reg_coef) > thr
    sel_names = feature_names[above_mask]
    sel_coefs = reg_coef[above_mask]

    vk_hc, vk_oc, vk_formula = [], [], []
    vk_dbe, vk_ocount, vk_coef_vals, vk_feat_labels = [], [], [], []

    for fname, coef_val in zip(sel_names, sel_coefs):
        entry = _vk_lookup.get(fname)
        if entry is None:
            continue
        hc_v, oc_v, formula_s, dbe_v, oc_cnt = entry
        if np.isnan(hc_v) or np.isnan(oc_v):
            continue
        vk_hc.append(hc_v)
        vk_oc.append(oc_v)
        vk_formula.append(formula_s)
        vk_dbe.append(dbe_v)
        vk_ocount.append(oc_cnt)
        vk_coef_vals.append(coef_val)
        vk_feat_labels.append(fname)

    vk_fig = go.Figure()
    for rname, coords in VK_REGIONS.items():
        fill_rgba, solid_hex = VK_REGION_COLORS[rname]
        vk_fig.add_trace(go.Scatter(
            x=coords["x"], y=coords["y"],
            fill="toself", fillcolor=fill_rgba,
            line=dict(color=solid_hex, width=1.5),
            mode="lines", hoverinfo="skip", showlegend=False,
        ))

    if vk_hc:
        vk_hc_arr = np.array(vk_hc)
        vk_oc_arr = np.array(vk_oc)
        vk_coef_arr = np.array(vk_coef_vals)
        vk_ocount_arr = np.array(vk_ocount, dtype=float)
        vk_dbe_arr = np.array(vk_dbe, dtype=float)

        if vk_size_by == "coef":
            sv = np.abs(vk_coef_arr)
        elif vk_size_by == "ocount":
            sv = np.where(np.isfinite(vk_ocount_arr), vk_ocount_arr, 0)
        elif vk_size_by == "dbe":
            sv = np.where(np.isfinite(vk_dbe_arr), np.abs(vk_dbe_arr), 0)
        else:
            sv = np.ones(len(vk_hc_arr))
        mn_s, mx_s = sv.min(), sv.max()
        sz = 6 + 14 * (sv - mn_s) / (mx_s - mn_s) if mx_s > mn_s else np.full(len(sv), 10.0)

        for sign, colour, label in [(1, "blue", "Positive coef"), (-1, "red", "Negative coef")]:
            mask = (vk_coef_arr >= 0) if sign == 1 else (vk_coef_arr < 0)
            if not np.any(mask):
                continue
            hover = [
                f"<b>{vk_feat_labels[i]}</b><br>"
                f"Formula: {vk_formula[i]}<br>"
                f"O/C: {vk_oc_arr[i]:.3f}  H/C: {vk_hc_arr[i]:.3f}<br>"
                f"DBE: {vk_dbe_arr[i]:.1f}  O count: {vk_ocount_arr[i]:.0f}<br>"
                f"Coef: {vk_coef_arr[i]:.4f}"
                for i in np.where(mask)[0]
            ]
            vk_fig.add_trace(go.Scatter(
                x=vk_oc_arr[mask], y=vk_hc_arr[mask],
                mode="markers",
                name=label,
                marker=dict(
                    size=sz[mask], color=colour,
                    line=dict(width=0.5, color="black"), opacity=0.85,
                ),
                text=hover, hoverinfo="text",
            ))
    else:
        vk_fig.add_annotation(
            text=f"No features with |coef| > {thr} that have a parseable Sum formula",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray"),
        )

    n_pos = int(np.sum(np.array(vk_coef_vals) >= 0)) if vk_coef_vals else 0
    n_neg = int(np.sum(np.array(vk_coef_vals) < 0)) if vk_coef_vals else 0

    vk_fig.update_layout(
        title=(f"Van Krevelen: |coef| > {thr}  —  "
               f"{n_pos} positive (blue)  |  {n_neg} negative (red)"),
        xaxis_title="O/C",
        yaxis_title="H/C",
        xaxis=dict(range=[-0.05, 1.35], zeroline=True, zerolinewidth=1, zerolinecolor="black",
                   showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        yaxis=dict(range=[-0.1, 2.8], zeroline=True, zerolinewidth=1, zerolinecolor="black",
                   showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=650, legend=dict(x=1.01, y=1, xanchor="left"),
        hoverlabel=dict(font_size=11),
    )

    cv_desc = {
        "kfold": f"KFold (K={int(cv_folds)})",
        "repeatedkfold": f"RepeatedKFold (K={int(cv_folds)} × repeats={int(cv_repeats)})",
        "loo": "Leave-One-Out",
        "groupkfold": f"GroupKFold (K={int(cv_folds)}), group={cv_group_col}",
    }.get(cv_method, str(cv_method))

    stats_html = html.Div([
        html.H3("Model Statistics"),
        html.P(f"Y Variable: {y_var} | Components: {n_comp} | Samples: {len(X)}"),
        html.P(f"Cross-validation: {cv_desc}"),
        html.Div([
            html.Div([
                html.H4("Calibration"),
                html.P(f"R² = {r2_cal:.4f}"),
                html.P(f"RMSE = {rmse_cal:.4f}")
            ], style={"width": "48%", "display": "inline-block", "padding": "10px"}),
            html.Div([
                html.H4("Cross-Validation (leakage-safe)"),
                html.P(f"R² = {r2_cv:.4f}"),
                html.P(f"RMSE = {rmse_cv:.4f}")
            ], style={"width": "48%", "display": "inline-block", "padding": "10px"}),
        ]),
        html.P(f"Features shown in coef plot: {top_n} | VK threshold: |coef| > {thr} → {n_pos + n_neg} features plotted"),
    ])

    return (
        score_fig, loading_fig, pred_fig, coef_fig, vk_fig,
        stats_html, pls_export_data,
        score_fig, loading_fig, pred_fig, coef_fig, vk_fig
    )

@app.callback(
    [Output("download_image", "data"), Output("download-status", "children")],
    Input("download_button", "n_clicks"),
    [
        State("current-score-fig", "data"), State("current-loading-fig", "data"),
        State("current-prediction-fig", "data"), State("current-vip-fig", "data"),
        State("current-vk-fig", "data"),
        State("download_plot_select", "value"), State("download_dpi", "value"),
        State("download_format", "value")
    ],
    prevent_initial_call=True,
)
def download_plot(n_clicks, s_fig, l_fig, p_fig, v_fig, vk_fig_data,
                  plot_select, dpi, file_format):
    if n_clicks == 0 or not KALEIDO_AVAILABLE:
        return None, ""
    try:
        fig_dict = {
            "score": (s_fig, "score_plot"),
            "loading": (l_fig, "loading_plot"),
            "prediction": (p_fig, "prediction_plot"),
            "vip": (v_fig, "regression_coefficients_plot"),
            "vk": (vk_fig_data, "van_krevelen_plot"),
        }
        fig_data, base_name = fig_dict[plot_select]
        if fig_data is None:
            return None, "No figure data available"
        fig = go.Figure(fig_data)
        img = pio.to_image(fig, format=file_format,
                           width=int(1200 * (dpi / 100)), height=int(650 * (dpi / 100)), scale=1)
        fname = f"{base_name}.{file_format}"
        return dcc.send_bytes(img, fname), f"Downloaded: {fname}"
    except Exception as e:
        return None, f"Error: {str(e)}"

@app.callback(
    Output("download_vip_data", "data"),
    Input("download_vip_button", "n_clicks"),
    State("pls-data", "data"),
    prevent_initial_call=True,
)
def download_coef_data(n_clicks, pls_data):
    if n_clicks == 0 or pls_data is None:
        return None
    try:
        df = pd.DataFrame({
            "Feature": pls_data["feature_names"],
            "Regression_Coefficient": pls_data["reg_coef"],
            "VIP": pls_data["vip_scores"],
        })
        P = np.array(pls_data["P"])
        for i in range(P.shape[1]):
            df[f"LV{i+1}_loading"] = P[:, i]
        for col in ["DBE", "O count"]:
            if col in feature_metadata:
                df[col] = [feature_metadata[col].get(f, np.nan) for f in pls_data["feature_names"]]
        df = df.sort_values("Regression_Coefficient", key=abs, ascending=False)
        return dict(content=df.to_csv(index=False), filename="pls_regression_coefficients.csv")
    except Exception as e:
        print(f"Error: {e}")
        return None

@app.callback(
    Output("download_posneg_data", "data"),
    Input("download_posneg_button", "n_clicks"),
    [State("pls-data", "data"), State("top_n_posneg", "value")],
    prevent_initial_call=True,
)
def download_posneg(n_clicks, pls_data, top_n):
    if n_clicks == 0 or pls_data is None:
        return None
    try:
        top_n = int(top_n or 20)
        features = np.array(pls_data["feature_names"])
        coefs = np.array(pls_data["reg_coef"])
        vips = np.array(pls_data["vip_scores"])
        P = np.array(pls_data["P"])

        def make_block(mask, label):
            idx = np.where(mask)[0]
            idx = idx[np.argsort(np.abs(coefs[idx]))[::-1]][:top_n]
            df = pd.DataFrame({
                "Rank": range(1, len(idx) + 1),
                "Direction": label,
                "Feature": features[idx],
                "Regression_Coefficient": coefs[idx],
                "VIP": vips[idx],
            })
            for lv in range(P.shape[1]):
                df[f"LV{lv+1}_loading"] = P[idx, lv]
            for col in ["DBE", "O count"]:
                if col in feature_metadata:
                    df[col] = [feature_metadata[col].get(f, np.nan) for f in features[idx]]
            return df

        combined = pd.concat(
            [make_block(coefs >= 0, "positive"), make_block(coefs < 0, "negative")],
            ignore_index=True,
        )
        return dict(content=combined.to_csv(index=False),
                    filename=f"pls_top{top_n}_positive_negative_coefficients.csv")
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("\nRunning automated PLS evaluation for key metadata targets...")
    eval_results = []
    all_trials = []

    X_eval = (
        filtered_data.apply(pd.to_numeric, errors="coerce")
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .T
    )

    for target in EVAL_TARGETS:
        res = evaluate_target_models(
            X_all=X_eval,
            meta_df=filtered_meta,
            sample_col=sample_col,
            target_name=target,
            max_lv=4,
            cv_method="kfold",
            cv_folds=5,
            cv_repeats=3,
            x_transforms=EVAL_X_TRANSFORMS,
            x_scalings=EVAL_X_SCALINGS,
            y_scalings=EVAL_Y_SCALINGS,
        )

        if res and res.get("trial_rows"):
            all_trials.extend(res["trial_rows"])

        if not res or not res.get("modelable", False):
            eval_results.append({
                "Target": target,
                "Modelable": "No",
                "Reason": (res or {}).get("reason", "No stable model found"),
                "Prediction plot": "",
                "Coefficient plot": "",
            })
        else:
            top_feats = res["top_features"].copy()
            top_feats["coef_str"] = top_feats["coef"].map(lambda v: f"{v:.6g}")
            eval_results.append({
                "Target": target,
                "Modelable": "Yes",
                "Best LV": res["best_lv"],
                "X transform": res["x_transform"],
                "X scaling": res["x_scale"],
                "Y scaling": res["y_scale"],
                "R2 cal": round(res["r2_cal"], 4),
                "RMSE cal": round(res["rmse_cal"], 4),
                "R2 CV": round(res["r2_cv"], 4),
                "RMSE CV": round(res["rmse_cv"], 4),
                "Prediction plot": res["prediction_plot"],
                "Coefficient plot": res["coefficient_plot"],
                "Top features": ", ".join(top_feats["feature"].head(8).tolist()),
                "Top feature coefficients": ", ".join(top_feats["coef_str"].head(8).tolist()),
            })

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(eval_results)
    summary_path = EVAL_OUTPUT_DIR / "pls_target_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    trials_df = pd.DataFrame(all_trials)
    if not trials_df.empty:
        trials_df = trials_df.sort_values(
            ["Target", "R2 CV", "RMSE CV"],
            ascending=[True, False, True],
        )
    trials_path = EVAL_OUTPUT_DIR / "pls_target_all_trials.csv"
    # --------------------------------------------------
# Select a single LV for the final summary CSV
# --------------------------------------------------
if trials_df.empty:
    print("No trial results available; skipping LV selection summary.")
else:
    print("\nAvailable LVs in trial results:")
    available_lvs = sorted(trials_df["Best LV"].dropna().astype(int).unique().tolist())
    print("  " + ", ".join(map(str, available_lvs)))

    chosen_lv_raw = input("Enter the single LV number to use for the final summary CSV: ").strip()
    try:
        chosen_lv = int(chosen_lv_raw)
    except ValueError:
        raise ValueError(f"Invalid LV selection: '{chosen_lv_raw}' is not an integer.")

    if chosen_lv not in available_lvs:
        raise ValueError(
            f"Chosen LV {chosen_lv} is not present in the trial results. "
            f"Available LVs are: {available_lvs}"
        )

    selected_trials = trials_df[trials_df["Best LV"] == chosen_lv].copy()

    final_rows = []
    for target in EVAL_TARGETS:
        trows = selected_trials[selected_trials["Target"] == target].copy()

        if trows.empty:
            final_rows.append({
                "Target": target,
                "Modelable": "No",
                "Reason": f"No stable model found at selected LV={chosen_lv}",
                "Best LV": chosen_lv,
                "Prediction plot": "",
                "Coefficient plot": "",
            })
            continue

        # Choose best preprocessing combo at the selected LV:
        # highest R2 CV, then lowest RMSE CV
        trows = trows.sort_values(
            ["R2 CV", "RMSE CV"],
            ascending=[False, True],
        )
        best_row = trows.iloc[0].to_dict()

        final_rows.append({
            "Target": target,
            "Modelable": "Yes",
            "Best LV": int(best_row["Best LV"]),
            "X transform": best_row["X transform"],
            "X scaling": best_row["X scaling"],
            "Y scaling": best_row["Y scaling"],
            "R2 cal": round(float(best_row["R2 cal"]), 4),
            "RMSE cal": round(float(best_row["RMSE cal"]), 4),
            "R2 CV": round(float(best_row["R2 CV"]), 4),
            "RMSE CV": round(float(best_row["RMSE CV"]), 4),
            "Prediction plot": best_row.get("Prediction plot", ""),
            "Coefficient plot": "",
            "Top features": best_row.get("Top features", ""),
            "Top feature coefficients": best_row.get("Top feature coefficients", ""),
        })

    summary_lv_df = pd.DataFrame(final_rows)
    summary_lv_path = EVAL_OUTPUT_DIR / f"pls_target_summary_lv{chosen_lv}.csv"
    summary_lv_df.to_csv(summary_lv_path, index=False)
    print(f"Saved: {summary_lv_path}")

    slide_cols = [
        "Target", "Modelable", "Best LV", "R2 CV", "RMSE CV",
        "Prediction plot", "Coefficient plot",
    ]
    slide_lv_df = summary_lv_df[[c for c in slide_cols if c in summary_lv_df.columns]].copy()
    slide_lv_path = EVAL_OUTPUT_DIR / f"pls_target_summary_lv{chosen_lv}_slide_ready.csv"
    slide_lv_df.to_csv(slide_lv_path, index=False)
    print(f"Saved: {slide_lv_path}")
    
    trials_df.to_csv(trials_path, index=False)
    print(f"Saved: {trials_path}")

    slide_cols = [
        "Target", "Modelable", "Best LV", "R2 CV", "RMSE CV",
        "Prediction plot", "Coefficient plot",
    ]
    slide_df = summary_df[[c for c in slide_cols if c in summary_df.columns]].copy()
    slide_path = EVAL_OUTPUT_DIR / "pls_target_summary_slide_ready.csv"
    slide_df.to_csv(slide_path, index=False)
    print(f"Saved: {slide_path}")

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))

    app.run(debug=True, port=8051)
    
# %%
    
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SUMMARY_FILE = "pls_evaluation_outputs/plots_withoutFeed/pls_target_summary_lv3_withoutFeed.csv"
DATA_FILE = "FKTv2-onlySamples_Reduced(viaTrendlineHigher0-6).csv"

summary = pd.read_csv(SUMMARY_FILE)
data = pd.read_csv(DATA_FILE)

# Use the actual feature identifier column or the row index
if "Feature" in data.columns:
    data["Feature"] = data["Feature"].astype(str).str.strip()
elif "Sample" in data.columns:
    # fallback only if your file truly stores features in Sample
    data["Feature"] = data["Sample"].astype(str).str.strip()
else:
    data["Feature"] = data.index.astype(str).str.strip()

data["DBE"] = pd.to_numeric(data.get("DBE"), errors="coerce")
data["O count"] = pd.to_numeric(data.get("O count"), errors="coerce")

meta_lookup = (
    data[["Feature", "Sum formula", "DBE", "O count"]]
    .dropna(subset=["Feature"])
    .drop_duplicates(subset=["Feature"], keep="first")
    .set_index("Feature")
)

def parse_top_features(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]

def parse_top_coeffs(s):
    if pd.isna(s) or not str(s).strip():
        return []
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            try:
                vals.append(float(x))
            except ValueError:
                pass
    return vals

rows = []

for _, r in summary.iterrows():
    target = r.get("Target")
    feats = parse_top_features(r.get("Top features"))
    coefs = parse_top_coeffs(r.get("Top feature coefficients"))

    n = min(len(feats), len(coefs))
    feats = feats[:n]
    coefs = coefs[:n]

    for rank, (feat, coef) in enumerate(zip(feats, coefs), start=1):
        if feat in meta_lookup.index:
            meta = meta_lookup.loc[feat]
            rows.append({
                "Target": target,
                "Rank": rank,
                "Feature": feat,
                "Formula": meta.get("Sum formula", np.nan),
                "DBE": meta.get("DBE", np.nan),
                "O count": meta.get("O count", np.nan),
                "Coefficient": coef,
                "AbsCoefficient": abs(coef),
                "Direction": "positive" if coef >= 0 else "negative",
            })

df = pd.DataFrame(rows)

if df.empty:
    raise ValueError(
        "No annotated rows were created. Check that the feature names in "
        "'Top features' match the feature names in your data table."
    )

df.to_csv("pls_top_features_annotated.csv", index=False)

corr_rows = []
for target, g in df.groupby("Target"):
    g = g.dropna(subset=["Coefficient", "DBE", "O count"])
    if len(g) >= 3:
        r_db, p_db = pearsonr(g["Coefficient"], g["DBE"])
        rs_db, ps_db = spearmanr(g["Coefficient"], g["DBE"])
        r_o, p_o = pearsonr(g["Coefficient"], g["O count"])
        rs_o, ps_o = spearmanr(g["Coefficient"], g["O count"])

        corr_rows.append({
            "Target": target,
            "n": len(g),
            "Pearson_coef_DBE": r_db,
            "Pearson_p_DBE": p_db,
            "Spearman_coef_DBE": rs_db,
            "Spearman_p_DBE": ps_db,
            "Pearson_coef_Ocount": r_o,
            "Pearson_p_Ocount": p_o,
            "Spearman_coef_Ocount": rs_o,
            "Spearman_p_Ocount": ps_o,
        })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv("pls_feature_metadata_correlations.csv", index=False)

print("Saved:")
print(" - pls_top_features_annotated.csv")
print(" - pls_feature_metadata_correlations.csv")
print("\nCorrelation summary:")
print(corr_df.to_string(index=False))