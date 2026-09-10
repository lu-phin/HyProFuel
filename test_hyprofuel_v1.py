import numpy as np
import pandas as pd

from hyprofuel_v1 import compute_hc_oc, load_project_data, parse_formula, transform_array
from pls_evaluate_targets_v1 import prediction_title


def test_parse_formula_and_van_krevelen_coordinates():
    assert parse_formula("C9H14O2") == (9.0, 14.0, 2.0)
    metadata = pd.DataFrame({"Sum formula": ["C9H14O2"], "DBE": [3], "O count": [2]}, index=["feature"])
    result = compute_hc_oc(metadata)
    assert np.isclose(result.loc["feature", "H/C"], 14 / 9)
    assert np.isclose(result.loc["feature", "O/C"], 2 / 9)


def test_load_project_data_aligns_samples(tmp_path):
    feature_path = tmp_path / "features.csv"
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({
        "Sum formula": ["C2H4O", "C3H6O2"], "O count": [1, 2], "DBE": [1, 1],
        "sample_b": [2, 4], "sample_a": [1, 3], "unused": [9, 9],
    }, index=["f1", "f2"]).to_csv(feature_path)
    pd.DataFrame({"Sample": ["sample_a", "sample_b"], "target": [10, 20]}).to_csv(metadata_path, index=False)
    project = load_project_data(feature_path, metadata_path)
    assert list(project.X.index) == ["sample_b", "sample_a"]
    assert project.X.shape == (2, 2)
    assert list(project.metadata["Sample"]) == ["sample_b", "sample_a"]


def test_nonnegative_transforms_are_finite():
    values = np.array([[0.0, 3.0]])
    assert np.isfinite(transform_array(values, "log10")).all()


def test_prediction_title_includes_r2_values():
    assert prediction_title("Target A", 0.91234, 0.85678) == "Target A\nR2Cal=0.912 | R2CV=0.857"
