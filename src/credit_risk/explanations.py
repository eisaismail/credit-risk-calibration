"""Explain the boosted model on actual held-out UCI records using exact Tree SHAP.

The explanation is of the uncalibrated model's log-odds, not the sigmoid-calibrated
probability. No fabricated borrowers or generated invoice-style demo data are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "LIMIT_BAL": "Credit limit (NT dollars)", "AGE": "Age",
    "SEX": "Recorded sex code", "EDUCATION": "Education code", "MARRIAGE": "Marital-status code",
    "PAY_0": "Most recent repayment-status code",
    **{f"PAY_{i}": f"Repayment-status code, month {i}" for i in range(2, 7)},
    **{f"BILL_AMT{i}": f"Bill amount, month {i} (NT dollars)" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"Payment amount, month {i} (NT dollars)" for i in range(1, 7)},
}


def explain_model(model, calibrated, X_test, y_test, output: Path, seed: int):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError('Install explanation dependencies: pip install -e ".[explain]"') from exc
    output.mkdir(parents=True, exist_ok=True)
    transformed = np.asarray(model.named_steps["preprocess"].transform(X_test))
    names = [str(n).split("__", 1)[-1]
             for n in model.named_steps["preprocess"].get_feature_names_out()]
    if names != list(X_test.columns):
        raise ValueError("This explanation export expects the original numeric UCI feature columns.")
    estimator = model.named_steps["model"]
    explainer = shap.TreeExplainer(estimator, model_output="raw",
                                   feature_perturbation="tree_path_dependent")
    explained = explainer(transformed, check_additivity=True)
    values = np.asarray(explained.values, dtype=float)
    base = np.broadcast_to(np.asarray(explained.base_values).reshape(-1), (len(X_test),))
    raw_scores = estimator.decision_function(transformed)
    reconstructed = base + values.sum(axis=1)
    max_error = float(np.max(np.abs(reconstructed - raw_scores)))
    if not np.isfinite(values).all() or max_error > 1e-6:
        raise ValueError("SHAP contributions failed the prediction-reconstruction check.")
    raw_probability = model.predict_proba(X_test)[:, 1]
    reconstructed_probability = 1 / (1 + np.exp(-reconstructed))
    probability_error = float(np.max(np.abs(reconstructed_probability - raw_probability)))
    if probability_error > 1e-7:
        raise ValueError("Reconstructed probabilities do not match the uncalibrated model.")
    calibrated_probability = calibrated.predict_proba(X_test)[:, 1]
    global_importance = pd.DataFrame({"feature": names,
        "description": [LABELS.get(n, n) for n in names],
        "mean_absolute_shap_log_odds": np.abs(values).mean(axis=0),
        "mean_signed_shap_log_odds": values.mean(axis=0)}).sort_values(
            "mean_absolute_shap_log_odds", ascending=False)
    global_importance.to_csv(output / "global_importance.csv", index=False)
    shap_frame = pd.DataFrame(values, columns=names, index=X_test.index)
    shap_frame.index.name = "source_row_id"
    shap_frame.to_csv(output / "heldout_shap_values.csv")
    X_test.to_csv(output / "heldout_features.csv", index_label="source_row_id")

    # Random selection is fixed before inspecting outcomes; not a hand-picked success set.
    selected = np.random.default_rng(seed).choice(len(X_test), size=min(5, len(X_test)), replace=False)
    examples = []
    for position in selected:
        row = X_test.iloc[position]
        order = np.argsort(-np.abs(values[position]))
        contributions = [{"feature": names[i], "description": LABELS.get(names[i], names[i]),
                          "observed_value": float(row.iloc[i]), "shap_log_odds": float(values[position, i]),
                          "direction": "raises model risk score" if values[position, i] > 0 else "lowers model risk score"}
                         for i in order]
        examples.append({"source_row_id": int(row.name),
                         "source_excel_row": int(row.name) + 3,
                         "observed_default_next_month": int(y_test.iloc[position]),
                         "uncalibrated_probability": float(raw_probability[position]),
                         "calibrated_probability": float(calibrated_probability[position]),
                         "base_log_odds": float(base[position]),
                         "reconstructed_log_odds": float(reconstructed[position]),
                         "contributions": contributions})
    (output / "individual_predictions.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    metadata = {"method": "Tree SHAP", "feature_perturbation": "tree_path_dependent",
                "target": "uncalibrated gradient-boosting log-odds of default",
                "explained_rows": len(X_test), "local_examples": len(examples),
                "selection": f"Five held-out rows sampled without replacement; seed {seed}",
                "max_log_odds_reconstruction_error": max_error,
                "max_probability_reconstruction_error": probability_error,
                "shap_version": shap.__version__,
                "notes": ["All explained inputs are actual held-out UCI records.",
                          "Training tree-path counts supply the reference distribution; no invented borrowers.",
                          "SHAP values are model attributions, not causes or lending recommendations.",
                          "Correlated payment and bill features can share attribution in assumption-dependent ways.",
                          "Values do not add up to the calibrated probability.",
                          "source_row_id is the zero-based data-row index, not the original client ID."]}
    (output / "explanation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    top = global_importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["description"], top["mean_absolute_shap_log_odds"], color="#327264")
    ax.set(xlabel="Mean absolute contribution to uncalibrated log-odds",
           title=f"Gradient boosting: global attribution on {len(X_test):,} real held-out records")
    fig.tight_layout()
    fig.savefig(output / "global_importance.png", dpi=160)
    plt.close(fig)
    first = examples[0]
    displayed = first["contributions"][:8]
    fig, ax = plt.subplots(figsize=(9, 5))
    items = list(reversed(displayed))
    ax.barh([x["feature"] for x in items], [x["shap_log_odds"] for x in items],
            color=["#b34b3e" if x["shap_log_odds"] > 0 else "#327264" for x in items])
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set(xlabel="SHAP contribution to uncalibrated log-odds (top 8 features)",
           title=f"Actual held-out row {first['source_row_id']}: factors influencing the model score")
    fig.tight_layout()
    fig.savefig(output / "individual_example.png", dpi=160)
    plt.close(fig)
    lines = ["# Prediction explanations", "", f"Explained {len(X_test):,} real held-out records using Tree SHAP.",
             "Positive values raise the model's default score; negative values lower it. Values use log-odds, not percentage points.",
             "These describe model behaviour and do not show why a person actually defaulted.", "",
             "| Real data row | Observed default | Raw probability | Calibrated probability | Largest contribution |",
             "|---|---:|---:|---:|---|"]
    for example in examples:
        strongest = example["contributions"][0]
        lines.append(f"| {example['source_row_id']} | {example['observed_default_next_month']} | "
                     f"{example['uncalibrated_probability']:.3f} | {example['calibrated_probability']:.3f} | "
                     f"{strongest['feature']} = {strongest['observed_value']:g}: {strongest['direction']} |")
    lines += ["", "All 23 contributions plus the base score reconstruct each uncalibrated score; the chart displays only the top eight.",
              f"Maximum score reconstruction error across all rows: {max_error:.3g}.",
              "The calibrated probability is shown separately; these SHAP values do not explain the calibration layer.",
              "Rows were sampled with a fixed seed, not selected for correct predictions. See JSON for all contributions and observed values."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metadata
