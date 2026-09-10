"""Evaluate configured metadata targets with leakage-safe PLS cross-validation.

Example:
    python pls_evaluate_targets_v1.py --output-dir pls_evaluation_outputs_v1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hyprofuel_v1 import (
    DEFAULT_DATA_FILE,
    DEFAULT_METADATA_FILE,
    cv_predict,
    fit_pls,
    load_project_data,
    metrics,
    select_samples,
)

TARGETS = (
    "MCR [wt %]", "O [wt %]", "Viscosity [cSt]",
    "Carbonyl number [mol/kg]", "CAN [mg KOH/g]", "SG 60/60 [°F]",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def prediction_title(target: str, r2_cal: float, r2_cv: float) -> str:
    return f"{target}\nR2Cal={r2_cal:.3f} | R2CV={r2_cv:.3f}"


def evaluate_target(project, target: str, output_dir: Path, max_components: int = 4):
    if target not in project.metadata.columns:
        return {"Target": target, "Modelable": "No", "Reason": "Target column missing"}
    y = project.metadata.set_index(project.sample_column)[target]
    best = None
    trials = []
    for components in range(1, max_components + 1):
        try:
            model, X_valid, y_valid, fitted = fit_pls(
                project.X, y, components, "center", "center", "log10")
            _, y_cv, predictions = cv_predict(
                project.X, y, components, "center", "center", "log10", "kfold")
            mask = np.isfinite(predictions)
            calibration = metrics(y_valid, fitted)
            cross_validation = metrics(y_cv[mask], predictions[mask])
            trial = {
                "Target": target, "LV": components,
                "R2 cal": calibration["r2"], "RMSE cal": calibration["rmse"],
                "R2 CV": cross_validation["r2"], "RMSE CV": cross_validation["rmse"],
                "n_samples": len(y_valid), "n_features": X_valid.shape[1],
            }
            trials.append(trial)
            score = (cross_validation["r2"], -cross_validation["rmse"])
            if best is None or score > best["score"]:
                best = {"score": score, "trial": trial, "model": model,
                        "actual": y_valid, "predicted": fitted, "cv": predictions}
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            trials.append({"Target": target, "LV": components, "Modelable": "No",
                           "Reason": str(error)})

    if best is None:
        return {"Target": target, "Modelable": "No", "Reason": "No valid model configuration",
                "trials": trials}

    trial = best["trial"]
    stem = safe_name(target)
    figure_path = output_dir / "plots" / f"{stem}_prediction.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(best["actual"], best["predicted"], label="Calibration")
    mask = np.isfinite(best["cv"])
    axis.scatter(best["actual"][mask], best["cv"][mask], label="Cross-validation", marker="^")
    bounds = np.concatenate([best["actual"], best["predicted"], best["cv"][mask]])
    axis.plot([bounds.min(), bounds.max()], [bounds.min(), bounds.max()], "k--")
    axis.set(
        xlabel="Actual",
        ylabel="Predicted",
        title=prediction_title(target, trial["R2 cal"], trial["R2 CV"]),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)

    result = {**trial, "Modelable": "Yes", "Prediction plot": str(figure_path)}
    result["trials"] = trials
    return result


def run_evaluation(project, targets, output_dir: Path, max_components: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    trials = []
    for target in targets:
        result = evaluate_target(project, target, output_dir, max_components)
        results.append({key: value for key, value in result.items() if key != "trials"})
        trials.extend(result.get("trials", []))
        status = result.get("Reason", "ok")
        print(f"{target}: {status}")
    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "pls_target_summary.csv", index=False)
    pd.DataFrame(trials).to_csv(output_dir / "pls_target_all_trials.csv", index=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_FILE)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--output-dir", default="pls_evaluation_outputs_v1")
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--targets", nargs="*", default=list(TARGETS))
    parser.add_argument("--samples", nargs="*", help="Optional sample IDs to include")
    return parser.parse_args()


def main():
    args = parse_args()
    project = load_project_data(Path(args.data), Path(args.metadata))
    project = select_samples(project, args.samples)
    summary = run_evaluation(project, args.targets, Path(args.output_dir), args.max_components)
    print("\nEvaluation summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
