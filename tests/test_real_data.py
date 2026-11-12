"""Integration checks against the actual UCI workbook. No generated borrowers.

Set CREDIT_RISK_REAL_DATA=1 to download/use the real dataset and run these checks.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from credit_risk.data import download_dataset, validate_and_clean
from credit_risk.modeling import make_models, stratified_split, calibrate_probabilities

pytestmark = pytest.mark.skipif(os.getenv("CREDIT_RISK_REAL_DATA") != "1",
                               reason="Opt-in integration checks use the real UCI data")


@pytest.fixture(scope="module")
def actual_data():
    frame, checksum = download_dataset(Path("data/raw/default_of_credit_card_clients.xls"))
    X, y = validate_and_clean(frame)
    assert len(frame) == 30000 and X.shape[1] == 23
    assert len(checksum) == 64
    return stratified_split(X, y, random_state=42)


def test_boosting_and_shap_reconstruct_actual_heldout_predictions(actual_data, tmp_path):
    pytest.importorskip("shap")
    from credit_risk.explanations import explain_model
    split = actual_data
    model = make_models(split.X_train)["Gradient boosting"]
    model.fit(split.X_train, split.y_train)
    calibrated = calibrate_probabilities(model, split.X_cal, split.y_cal)
    metadata = explain_model(model, calibrated, split.X_test.head(40), split.y_test.head(40), tmp_path, 42)
    assert metadata["explained_rows"] == 40
    assert metadata["max_log_odds_reconstruction_error"] < 1e-6
    assert metadata["max_probability_reconstruction_error"] < 1e-7
    assert (tmp_path / "individual_predictions.json").is_file()


def test_real_split_has_no_overlap_or_missing_rows(actual_data):
    split = actual_data
    parts = [set(split.X_train.index), set(split.X_cal.index), set(split.X_test.index)]
    assert len(set.union(*parts)) == 30000
    assert sum(map(len, parts)) == 30000
    for X in [split.X_train, split.X_cal, split.X_test]:
        assert np.isfinite(X.to_numpy()).all()
