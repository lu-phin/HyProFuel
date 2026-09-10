"""Cleaned target evaluation workflow based on the original HyProFuel scripts.

This keeps the original evaluation logic and outputs, but skips the embedded
Dash dashboard so the file only handles target evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, RepeatedKFold

from hyprofuel_v1 import (
    DEFAULT_DATA_FILE,
    DEFAULT_METADATA_FILE,
    load_original_workflow_data,
    make_scaler,
    transform_array,
)

TARGETS = (
    "MCR [wt %]",
    "O [wt %]",
    "Viscosity [cSt]",
    "Carbonyl number [mol/kg]",
    "CAN [mg KOH/g]",
    "SG 60/60 [°F]",
)

EVAL_X_TRANSFORMS = ("log10",)
EVAL_X_SCALINGS = ("center",)
EVAL_Y_SCALINGS = ("center",)


def apply_x_transform_numpy(values: np.ndarray, method: str) -> np.ndarray:
    return transform_array(values, method)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def save_prediction_plot(target_name, y_true, y_pred_cal, y_pred_cv, r2_cal, r2_cv,
                         plot_dir: Path, file_stem_suffix=""):
    safe = safe_name(target_name)
    if file_stem_suffix:
        safe = f"{safe}_{file_stem_suffix}"
    fig, ax = plt.subplots(figsize=(6.5, 6))
    y_true = np.asarray(y_true, dtype=float)
    y_pred_cal = np.asarray(y_pred_cal, dtype=float)
    y_pred_cv = np.asarray(y_pred_cv, dtype=float)

    ax.scatter(
        y_true, y_pred_cal, s=40, alpha=0.85, label="Calibration",
        color="#636EFA", marker="o", edgecolors="black", linewidths=0.5,
    )
    ax.scatter(
        y_true, y_pred_cv, s=55, alpha=0.95, label="CV",
        color="#EF553B", marker="^", edgecolors="black", linewidths=0.8,
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

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = plot_dir / f"{safe}_prediction.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(filename)


def save_coefficient_plot(target_name, feature_names, reg_coef, plot_dir: Path, top_n=25):
    safe = safe_name(target_name)
    feature_names = np.asarray(feature_names)
    reg_coef = np.asarray(reg_coef, dtype=float)
    idx = np.argsort(np.abs(reg_coef))[::-1][:min(top_n, len(reg_coef))]
    idx = idx[::-1]
    labels = feature_names[idx]
    vals = reg_coef[idx]
    colors = ["royalblue" if value >= 0 else "tomato" for value in vals]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(labels))))
    ax.barh(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Regression coefficient")
    ax.set_title(f"{target_name} — Top coefficients")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = plot_dir / f"{safe}_coefficients.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(filename)


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
    X_all, meta_df, sample_col, target_name, plot_dir: Path,
    x_transforms=EVAL_X_TRANSFORMS,
    x_scalings=EVAL_X_SCALINGS,
    y_scalings=EVAL_Y_SCALINGS,
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
                X_df=X,
                y_series=y,
                n_components=n_comp,
                x_scale_method=x_s,
                y_scale_method=y_s,
                x_transform_method=x_t,
                cv_method=cv_method,
                cv_folds=cv_folds,
                cv_repeats=cv_repeats,
                random_state=42,
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
            y_pred_cal = y_scaler.inverse_transform(y_pred_cal_s).ravel() if y_scaler is not None else y_pred_cal_s.ravel()

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
                plot_dir,
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
                "Top feature coefficients": ", ".join(f"{value:.6g}" for value in coefs[top_idx][:10]),
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
        target_name,
        best["y_true"],
        best["y_pred_cal"],
        best["y_pred_cv"],
        best["r2_cal"],
        best["r2_cv"],
        plot_dir,
    )
    best["coefficient_plot"] = save_coefficient_plot(target_name, best["feature_names"], best["reg_coef"], plot_dir)
    best["trial_rows"] = trial_rows
    return best


def choose_lv(trials_df: pd.DataFrame, requested_lv: int | None) -> int:
    available_lvs = sorted(trials_df["Best LV"].dropna().astype(int).unique().tolist())
    print("\nAvailable LVs in trial results:")
    print("  " + ", ".join(map(str, available_lvs)))

    if requested_lv is None:
        chosen_lv_raw = input("Enter the single LV number to use for the final summary CSV: ").strip()
        try:
            requested_lv = int(chosen_lv_raw)
        except ValueError as error:
            raise ValueError(f"Invalid LV selection: '{chosen_lv_raw}' is not an integer.") from error

    if requested_lv not in available_lvs:
        raise ValueError(
            f"Chosen LV {requested_lv} is not present in the trial results. Available LVs are: {available_lvs}"
        )
    return requested_lv


def write_selected_lv_outputs(trials_df: pd.DataFrame, output_dir: Path, chosen_lv: int):
    selected_trials = trials_df[trials_df["Best LV"] == chosen_lv].copy()
    final_rows = []

    for target in TARGETS:
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

        trows = trows.sort_values(["R2 CV", "RMSE CV"], ascending=[False, True])
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
    summary_lv_path = output_dir / f"pls_target_summary_lv{chosen_lv}.csv"
    summary_lv_df.to_csv(summary_lv_path, index=False)
    print(f"Saved: {summary_lv_path}")

    slide_cols = ["Target", "Modelable", "Best LV", "R2 CV", "RMSE CV", "Prediction plot", "Coefficient plot"]
    slide_lv_df = summary_lv_df[[column for column in slide_cols if column in summary_lv_df.columns]].copy()
    slide_lv_path = output_dir / f"pls_target_summary_lv{chosen_lv}_slide_ready.csv"
    slide_lv_df.to_csv(slide_lv_path, index=False)
    print(f"Saved: {slide_lv_path}")

    return summary_lv_df


def run_evaluation(data_path: Path, metadata_path: Path, output_dir: Path,
                   samples=None, max_lv: int = 4, selected_lv: int | None = None,
                   run_sanity_check: bool = True):
    workflow = load_original_workflow_data(
        data_path,
        metadata_path,
        samples=samples,
        use_popup=samples is None,
        run_sanity_check=run_sanity_check,
    )
    X_eval = (
        workflow.filtered_data.apply(pd.to_numeric, errors="coerce")
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .T
    )

    eval_results = []
    all_trials = []
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        result = evaluate_target_models(
            X_all=X_eval,
            meta_df=workflow.filtered_metadata,
            sample_col=workflow.sample_column,
            target_name=target,
            plot_dir=plot_dir,
            max_lv=max_lv,
            cv_method="kfold",
            cv_folds=5,
            cv_repeats=3,
        )
        if result and result.get("trial_rows"):
            all_trials.extend(result["trial_rows"])

        if not result or not result.get("modelable", False):
            eval_results.append({
                "Target": target,
                "Modelable": "No",
                "Reason": (result or {}).get("reason", "No stable model found"),
                "Prediction plot": "",
                "Coefficient plot": "",
            })
            continue

        top_features = result["top_features"].copy()
        top_features["coef_str"] = top_features["coef"].map(lambda value: f"{value:.6g}")
        eval_results.append({
            "Target": target,
            "Modelable": "Yes",
            "Best LV": result["best_lv"],
            "X transform": result["x_transform"],
            "X scaling": result["x_scale"],
            "Y scaling": result["y_scale"],
            "R2 cal": round(result["r2_cal"], 4),
            "RMSE cal": round(result["rmse_cal"], 4),
            "R2 CV": round(result["r2_cv"], 4),
            "RMSE CV": round(result["rmse_cv"], 4),
            "Prediction plot": result["prediction_plot"],
            "Coefficient plot": result["coefficient_plot"],
            "Top features": ", ".join(top_features["feature"].head(8).tolist()),
            "Top feature coefficients": ", ".join(top_features["coef_str"].head(8).tolist()),
        })

    summary_df = pd.DataFrame(eval_results)
    summary_path = output_dir / "pls_target_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    trials_df = pd.DataFrame(all_trials)
    if not trials_df.empty:
        trials_df = trials_df.sort_values(["Target", "R2 CV", "RMSE CV"], ascending=[True, False, True])
    trials_path = output_dir / "pls_target_all_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"Saved: {trials_path}")

    slide_cols = ["Target", "Modelable", "Best LV", "R2 CV", "RMSE CV", "Prediction plot", "Coefficient plot"]
    slide_df = summary_df[[column for column in slide_cols if column in summary_df.columns]].copy()
    slide_path = output_dir / "pls_target_summary_slide_ready.csv"
    slide_df.to_csv(slide_path, index=False)
    print(f"Saved: {slide_path}")

    if trials_df.empty:
        print("No trial results available; skipping LV selection summary.")
    else:
        chosen_lv = choose_lv(trials_df, selected_lv)
        write_selected_lv_outputs(trials_df, output_dir, chosen_lv)

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="Feature CSV path")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_FILE, help="Metadata CSV path")
    parser.add_argument("--output-dir", default="pls_evaluation_outputs_v1", help="Directory for CSV and plot outputs")
    parser.add_argument("--samples", nargs="*", help="Optional sample IDs to include. If omitted, a selection dialog opens.")
    parser.add_argument("--max-lv", type=int, default=4, help="Maximum latent variables to test")
    parser.add_argument("--lv", type=int, help="Optional single LV to use for the final summary CSVs")
    parser.add_argument("--skip-sanity-check", action="store_true", help="Skip the startup sample/alignment sanity-check printout")
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(
        data_path=Path(args.data),
        metadata_path=Path(args.metadata),
        output_dir=Path(args.output_dir),
        samples=args.samples,
        max_lv=args.max_lv,
        selected_lv=args.lv,
        run_sanity_check=not args.skip_sanity_check,
    )


if __name__ == "__main__":
    main()
