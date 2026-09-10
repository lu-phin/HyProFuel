# HyProFuel

PLS regression tools for the FKT stability test set (LignoStab). The project
uses mass-spectrometry feature data to model process and fuel-property metadata
with leakage-safe cross-validation.

## Project contents

- `FKTv2-onlySamples_Reduced(viaTrendlineHigher0-6).csv`: feature matrix. Rows
	are molecular features and columns are samples.
- `FKT-metadata_Histogram.csv`: sample metadata and numeric target variables.
- `pls_dashboard_interactive_metadata.py`: interactive Dash application for
	fitting and exploring one PLS model at a time.
- `pls_evaluate_targets.py`: automated evaluation of several metadata targets,
	including model comparison, plots, and CSV summaries.

## Requirements

Python 3.9 or newer is recommended. Install the Python dependencies with:

```bash
python -m pip install numpy pandas scikit-learn matplotlib plotly dash kaleido
```

The applications use Tkinter to show the sample-selection dialog. On Debian or
Ubuntu, install it separately if needed:

```bash
sudo apt-get install python3-tk
```

`kaleido` is optional, but is required to download PNG, SVG, or PDF plots from
the Dash dashboard.

## Interactive dashboard

Run from the project directory:

```bash
python pls_dashboard_interactive_metadata.py
```

Choose the samples in the startup dialog, then open
`http://127.0.0.1:8051` in a browser. The dashboard supports:

- selection of numeric metadata targets;
- configurable PLS components, X/Y scaling, and X transformations;
- K-fold, repeated K-fold, leave-one-out, and group K-fold validation;
- score, loading, prediction, regression-coefficient, and Van Krevelen plots;
- CSV downloads for all coefficients or the strongest positive and negative
	coefficients.

Stop the server with `Ctrl+C`.

## Automated target evaluation

Run:

```bash
python pls_evaluate_targets.py
```

The evaluator tests the configured targets:

- MCR [wt %]
- O [wt %]
- Viscosity [cSt]
- Carbonyl number [mol/kg]
- CAN [mg KOH/g]
- SG 60/60 [°F]

It performs leakage-safe cross-validation, selects the best tested latent
variable count by cross-validated R² and RMSE, and writes results to
`pls_evaluation_outputs/`. During the run it asks which single latent-variable
count should be used for the final summary, then starts the Dash dashboard on
port 8051.

Typical outputs include:

- `pls_target_summary.csv`: best result for each target;
- `pls_target_summary_slide_ready.csv`: compact summary;
- `pls_target_summary_lvN.csv`: final summary using the selected `N`;
- `pls_target_summary_lvN_slide_ready.csv`: compact selected-LV summary;
- `pls_target_all_trials.csv`: every tested model configuration;
- `plots/`: prediction and coefficient PNG files for evaluated models.

## Data expectations

The scripts expect both CSV files to remain in the project directory. The
feature CSV must contain `Sum formula`, `O count`, and `DBE` columns, plus
sample columns. The metadata CSV must contain a sample identifier column named
`Sample`, `sample`, `ID`, `id`, or `Name`. Sample identifiers are trimmed and
matched between the two files before modeling.

## Notes

Model quality should be judged primarily with cross-validated metrics, not
calibration metrics. The scripts use a fixed random state of 42 for shuffled
K-fold validation to make runs reproducible.
