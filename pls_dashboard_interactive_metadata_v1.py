"""Clean v1 interactive PLS dashboard.

Run with: python pls_dashboard_interactive_metadata_v1.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from hyprofuel_v1 import (
    DEFAULT_DATA_FILE,
    DEFAULT_METADATA_FILE,
    compute_hc_oc,
    cv_predict,
    fit_pls,
    load_project_data,
    make_scaler,
    metrics,
    select_samples,
)


def numeric_targets(metadata, sample_column):
    return [column for column in metadata.columns if column != sample_column
            and metadata[column].apply(lambda value: _is_number(value)).all()]


def _is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def create_app(project):
    targets = numeric_targets(project.metadata, project.sample_column)
    if not targets:
        raise ValueError("No numeric metadata targets were found.")
    feature_annotations = compute_hc_oc(project.feature_metadata)

    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("HyProFuel PLS Dashboard v1"),
        html.Div([
            html.Label("Target"), dcc.Dropdown(targets, targets[0], id="target"),
            html.Label("Components"), dcc.Input(id="components", type="number", value=3, min=1, step=1),
            html.Label("X scaling"), dcc.Dropdown(
                ["autoscale", "center", "minmax", "robust", "none"], "autoscale", id="x-scaling"),
            html.Label("Y scaling"), dcc.Dropdown(
                ["autoscale", "center", "none"], "autoscale", id="y-scaling"),
            html.Label("X transform"), dcc.Dropdown(
                ["none", "sqrt", "log10", "log2", "cuberoot"], "none", id="x-transform"),
            html.Label("CV method"), dcc.Dropdown(
                ["kfold", "repeatedkfold", "loo"], "kfold", id="cv-method"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px"}),
        html.Div(id="stats", style={"padding": "12px"}),
        dcc.Graph(id="prediction"),
        dcc.Graph(id="coefficients"),
        dcc.Graph(id="van-krevelen"),
    ], style={"maxWidth": "1400px", "margin": "auto", "padding": "24px"})

    @app.callback(
        Output("stats", "children"), Output("prediction", "figure"),
        Output("coefficients", "figure"), Output("van-krevelen", "figure"),
        Input("target", "value"), Input("components", "value"),
        Input("x-scaling", "value"), Input("y-scaling", "value"),
        Input("x-transform", "value"), Input("cv-method", "value"),
    )
    def update(target, components, x_scaling, y_scaling, x_transform, cv_method):
        empty = go.Figure()
        try:
            target_values = project.metadata.set_index(project.sample_column)[target]
            model, X_valid, y_valid, fitted = fit_pls(
                project.X, target_values, components, x_scaling, y_scaling, x_transform)
            indices, y_cv, cv_values = cv_predict(
                project.X, target_values, components, x_scaling, y_scaling,
                x_transform, cv_method)
            fit_stats = metrics(y_valid, fitted)
            cv_mask = np.isfinite(cv_values)
            cv_stats = metrics(y_cv[cv_mask], cv_values[cv_mask])
            stats = html.Div([
                html.H3(f"{target}: {len(y_valid)} samples, {model.n_components} components"),
                html.P(f"Calibration R2={fit_stats['r2']:.4f}, RMSE={fit_stats['rmse']:.4f} | "
                       f"CV R2={cv_stats['r2']:.4f}, RMSE={cv_stats['rmse']:.4f}"),
            ])
            prediction = go.Figure([
                go.Scatter(x=y_valid, y=fitted, mode="markers", name="Calibration"),
                go.Scatter(x=y_cv[cv_mask], y=cv_values[cv_mask], mode="markers", name="CV"),
            ])
            prediction.update_layout(title="Predicted vs actual", xaxis_title="Actual", yaxis_title="Predicted")

            coefficients = np.asarray(model.coef_).ravel()
            order = np.argsort(np.abs(coefficients))[::-1][:50][::-1]
            coefficient_plot = go.Figure(go.Bar(
                x=[str(X_valid.columns[i]) for i in order], y=coefficients[order],
                marker_color=["royalblue" if value >= 0 else "tomato" for value in coefficients[order]],
            ))
            coefficient_plot.update_layout(title="Top regression coefficients", xaxis_tickangle=-45)

            hc_oc = feature_annotations.loc[X_valid.columns]
            valid_features = hc_oc[hc_oc.notna().all(axis=1)]
            van_krevelen = go.Figure(go.Scatter(
                x=valid_features["O/C"], y=valid_features["H/C"], mode="markers",
                text=valid_features.index.astype(str), marker={"size": 7, "color": "#2a6f97"},
            ))
            van_krevelen.update_layout(title="Van Krevelen feature map", xaxis_title="O/C", yaxis_title="H/C")
            return stats, prediction, coefficient_plot, van_krevelen
        except (ValueError, KeyError) as error:
            return html.P(f"Model error: {error}"), empty, empty, empty

    return app


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="Feature CSV path")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_FILE, help="Metadata CSV path")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--samples", nargs="*", help="Optional sample IDs to include")
    return parser.parse_args()


def main():
    args = parse_args()
    project = load_project_data(Path(args.data), Path(args.metadata))
    project = select_samples(project, args.samples)
    create_app(project).run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
