import numpy as np
import pandas as pd

from hyprofuel_v1 import (
    compute_hc_oc,
    load_original_workflow_data,
    load_project_data,
    parse_formula,
    transform_array,
)


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


def test_load_original_workflow_data_filters_requested_samples(tmp_path):
    feature_path = tmp_path / "features.csv"
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({
        "Sum formula": ["C2H4O", "C3H6O2"],
        "O count": [1, 2],
        "DBE": [1, 1],
        "sample_b": [2, 4],
        "sample_a": [1, 3],
        "unused": [9, 9],
    }, index=["f1", "f2"]).to_csv(feature_path)
    pd.DataFrame({
        "Sample": ["sample_a", "sample_b"],
        "target": [10, 20],
        "group": ["x", "y"],
    }).to_csv(metadata_path, index=False)

    workflow = load_original_workflow_data(
        feature_path,
        metadata_path,
        samples=["sample_a"],
        use_popup=False,
        run_sanity_check=False,
    )

    assert list(workflow.filtered_data.columns) == ["sample_a"]
    assert list(workflow.filtered_metadata["Sample"]) == ["sample_a"]
    assert "target" in workflow.numeric_metadata_columns
    assert "f1" in workflow.vk_lookup
