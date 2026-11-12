"""Run a real-data credit-default modelling experiment."""
from __future__ import annotations
import argparse
import json
import copy
from importlib.metadata import version
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import download_dataset, validate_and_clean
from .modeling import (
    stratified_split, fit_models, calibrate_probabilities,
    evaluate_probabilities, calibration_points, confusion_matrix_values,
)


def run(args: argparse.Namespace) -> pd.DataFrame:
    output = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw, checksum = download_dataset(args.raw_path, refresh=args.refresh)
    X, y = validate_and_clean(raw)
    split = stratified_split(X, y, random_state=args.seed)
    models = fit_models(split.X_train, split.y_train, random_state=args.seed)

    metrics = {}
    curves = {}
    calibrated_models = {}
    for name, model in models.items():
        raw_prob = model.predict_proba(split.X_test)[:, 1]
        calibrated = calibrate_probabilities(model, split.X_cal, split.y_cal, method=args.calibration)
        calibrated_models[name] = calibrated
        calibrated_prob = calibrated.predict_proba(split.X_test)[:, 1]
        metrics[f"{name} | uncalibrated"] = evaluate_probabilities(split.y_test, raw_prob, args.threshold)
        metrics[f"{name} | calibrated ({args.calibration})"] = evaluate_probabilities(
            split.y_test, calibrated_prob, args.threshold
        )
        curves[f"{name} | uncalibrated"] = calibration_points(split.y_test, raw_prob)
        curves[f"{name} | calibrated"] = calibration_points(split.y_test, calibrated_prob)
        joblib.dump(model, model_dir / (name.lower().replace(" ", "_") + ".joblib"))
        joblib.dump(calibrated, model_dir / (name.lower().replace(" ", "_") + "_calibrated.joblib"))
        pd.DataFrame({
            "y_true": split.y_test,
            "probability_uncalibrated": raw_prob,
            "probability_calibrated": calibrated_prob,
        }, index=split.y_test.index).to_csv(
            output / (name.lower().replace(" ", "_") + "_test_probabilities.csv"), index_label="row_id"
        )
        cm = confusion_matrix_values(split.y_test, calibrated_prob, args.threshold)
        pd.DataFrame(cm, index=["actual_nondefault", "actual_default"],
                     columns=["predicted_nondefault", "predicted_default"]).to_csv(
                         output / (name.lower().replace(" ", "_") + "_confusion_matrix.csv")
                     )

    metric_df = pd.DataFrame(metrics).T
    metric_df.index.name = "model"
    metric_df.to_csv(output / "metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, (predicted, observed) in curves.items():
        axes[0].plot(predicted, observed, marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axes[0].set(xlabel="Mean predicted default probability", ylabel="Observed default rate",
                title="Reliability diagram")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)

    metric_df[["ROC_AUC", "PR_AUC_average_precision"]].plot(kind="bar", ax=axes[1])
    axes[1].set(title="Ranking performance", ylabel="Score", ylim=(0, 1))
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)

    explanation_metadata = None
    if getattr(args, "explain", False):
        from .explanations import explain_model
        explanation_metadata = explain_model(
            models["Gradient boosting"], calibrated_models["Gradient boosting"],
            split.X_test, split.y_test, output / "explanations", args.seed,
        )
    partition = pd.concat([
        pd.DataFrame({"source_row_id": split.X_train.index, "partition": "train"}),
        pd.DataFrame({"source_row_id": split.X_cal.index, "partition": "calibration"}),
        pd.DataFrame({"source_row_id": split.X_test.index, "partition": "test"}),
    ]).sort_values("source_row_id")
    partition.to_csv(output / "split_membership.csv", index=False)

    report = {
        "source": "UCI Machine Learning Repository — Default of Credit Card Clients",
        "source_url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "sha256_raw_dataset": checksum,
        "rows": int(len(raw)),
        "features": int(X.shape[1]),
        "default_rate": float(y.mean()),
        "train_rows": int(len(split.y_train)),
        "calibration_rows": int(len(split.y_cal)),
        "test_rows": int(len(split.y_test)),
        "random_seed": args.seed,
        "calibration_method": args.calibration,
        "threshold": args.threshold,
        "explanations": explanation_metadata,
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "matplotlib"]},
        "model_parameters": {name: model.named_steps["model"].get_params() for name, model in models.items()},
        "notes": [
            "Random stratified split, not a temporal split: use only for a baseline project.",
            "Dataset is historical and from Taiwan; it is not a current lending population.",
            "Threshold 0.5 is a reporting convention, not an optimal lending decision threshold.",
        ],
    }
    (output / "run_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Dataset: UCI Default of Credit Card Clients (real historical records)")
    print(f"Rows={len(raw):,}; features={X.shape[1]}; default rate={y.mean():.3%}")
    print(f"Split sizes: train={len(split.y_train):,}, calibration={len(split.y_cal):,}, test={len(split.y_test):,}")
    print("\nMetrics:")
    print(metric_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved outputs to {output.resolve()} and models to {model_dir.resolve()}")
    return metric_df


def run_comparison(args: argparse.Namespace) -> None:
    """Repeat the fixed experiment; never select settings using test scores."""
    seeds = args.seeds or [args.seed]
    tables = []
    root = Path(args.output_dir)
    for number, seed in enumerate(seeds):
        current = copy.copy(args)
        current.seed = seed
        current.output_dir = str(root / f"seed-{seed}")
        current.model_dir = str(Path(args.model_dir) / f"seed-{seed}")
        current.explain = args.explain and number == 0
        print(f"\nRunning fixed model comparison, seed {seed}...", flush=True)
        frame = run(current).reset_index()
        frame.insert(0, "seed", seed)
        tables.append(frame)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(root / "comparison_by_seed.csv", index=False)
    numeric = combined.drop(columns=["seed", "model"]).columns.tolist()
    summary = combined.groupby("model")[numeric].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(pair) for pair in summary.columns]
    summary.to_csv(root / "comparison_summary.csv")
    lines = ["# Real-data model comparison", "", f"UCI Default of Credit Card Clients: 30,000 actual historical records. Seeds: {seeds}.",
             "Each run uses 18,000 training, 6,000 calibration and 6,000 test rows. Models share the same split within each seed.",
             "All settings were fixed before this comparison; the test set is not used for training, calibration or tuning.", "",
             "| Model | ROC-AUC mean (range) | Brier mean | Log loss mean | ECE mean |",
             "|---|---:|---:|---:|---:|"]
    for name, row in summary.iterrows():
        display_name = str(name).replace("|", "/")
        lines.append(f"| {display_name} | {row['ROC_AUC_mean']:.4f} ({row['ROC_AUC_min']:.4f}–{row['ROC_AUC_max']:.4f}) | "
                     f"{row['Brier_score_mean']:.4f} | {row['Log_loss_mean']:.4f} | {row['ECE_10_bins_mean']:.4f} |")
    lines += ["", "ROC-AUC measures ranking; higher is better. Brier, log loss and ECE measure probability error; lower is better.",
              "These ranges describe sensitivity to random splits. They are not confidence intervals: the test sets overlap across seeds.",
              f"Classification precision, recall and F1 use the configured {args.threshold:g} threshold, not an optimised lending threshold.",
              "This is historical Taiwanese credit-card data, not live borrower data or evidence of current lending performance."]
    if args.explain:
        lines += ["", f"[Explanations for real held-out rows, seed {seeds[0]}](seed-{seeds[0]}/explanations/README.md)",
                  "Tree SHAP explains the uncalibrated gradient-boosting score in log-odds. Calibrated probabilities are shown separately."]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default="data/raw/default_of_credit_card_clients.xls")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--refresh", action="store_true", help="Download original data again")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", help="Run a fixed comparison over these random split seeds")
    parser.add_argument("--explain", action="store_true", help="Explain boosted predictions on real held-out rows; requires .[explain]")
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must lie strictly between 0 and 1")
    if args.seeds:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must not contain duplicates")
        run_comparison(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
