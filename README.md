# Credit Default Prediction and Calibration

This project predicts whether a credit card client will default the following month. It compares three models and checks how accurate their predicted probabilities are. It also explains which recorded features influenced the gradient boosting model.

The results come from 30,000 real historical customer records. No customer records were generated for the experiment. Everything runs locally without a paid AI service or an API key.

## The data

The dataset is [Default of Credit Card Clients from UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). It contains records from Taiwan, including credit limits, bill amounts, payments and demographic information. Payment histories cover April to September 2005. There are 23 input features, and 22.12% of clients defaulted the following month.

These are real records, but they are old. They are not a live feed and do not show how the models would perform on borrowers today.

The workbook downloads on the first run and is then kept locally. Each run records its file checksum, package versions and model settings so the experiment can be reproduced. The raw workbook and trained model files are not uploaded to this repository.

## How the comparison works

The three models are logistic regression, random forest and histogram gradient boosting.

Each run uses 18,000 records for training, 6,000 for calibration and 6,000 for testing. The split keeps roughly the same default rate in each group. All models use the same groups within a run.

Training fits the models. Calibration adjusts their probability estimates using a separate group of clients. Testing checks the predictions on records that were not used for either step.

The experiment repeats with random seeds 42, 43 and 44. Model settings were fixed before the comparison. Test scores were not used to choose settings or thresholds.

## What the results showed

After calibration, the average results across the three splits were:

1. Logistic regression: ROC AUC 0.7230, Brier score 0.1465, log loss 0.4691 and calibration error 0.0521.
2. Random forest: ROC AUC 0.7797, Brier score 0.1358, log loss 0.4316 and calibration error 0.0158.
3. Gradient boosting: ROC AUC 0.7782, Brier score 0.1351, log loss 0.4314 and calibration error 0.0123.

ROC AUC measures how well a model ranks clients by risk. Higher is better. Brier score, log loss and calibration error measure errors in the probability estimates. Lower is better.

There was no clear winner between the two tree models. Random forest ranked clients slightly better on average. Gradient boosting had slightly smaller probability errors after calibration. The differences were small, and this experiment does not establish that they are statistically significant.

Calibration helped all three models. For gradient boosting, the average Brier score improved from 0.1745 to 0.1351. The default classification threshold of 0.5 is only used for reporting. It has not been chosen as the best threshold for a lending decision.

[Full comparison and results for each split](results/comparison/README.md)

![Model comparison for seed 42](results/comparison/seed-42/evaluation.png)

The original results for two models and seed 42 remain directly inside `results/`. The newer comparison is in `results/comparison/`.

## Explaining the predictions

SHAP is used to explain the gradient boosting model on all 6,000 test records from seed 42. It assigns a contribution to each feature, showing whether that feature pushed the model score up or down.

The largest average contribution came from the most recent repayment status, `PAY_0`, followed by credit limit, `LIMIT_BAL`.

![Features influencing the boosted model](results/comparison/seed-42/explanations/global_importance.png)

Five actual test records were also selected using a fixed random seed. Their exports show the recorded values, the observed default outcome, the predicted probabilities and every feature contribution. They were not selected because the predictions were correct.

[Individual prediction explanations](results/comparison/seed-42/explanations/README.md)

The contributions explain the raw model score in log odds. They are not percentage point changes and do not explain the separate calibration step. The base score plus all 23 contributions was checked against the original prediction for every explained record. The largest reconstruction error was about `1.2e-14`.

Tree SHAP uses counts from the training records that passed through the trees as its reference. It does not generate example borrowers. The exported `source_row_id` is the data row's position in the workbook, not the original client ID. The corresponding Excel row number is included too.

These explanations describe what influenced the model. They do not prove why someone defaulted. Related payment and bill features can share contributions, depending on the explanation method's assumptions.

## Running the project

Python 3.10 or newer is required. Run these commands from the project folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,explain]"

credit-risk --seeds 42 43 44 --explain --output-dir outputs/comparison --model-dir models/comparison
```

For one split without explanations:

```powershell
credit-risk --seed 42
```

On macOS or Linux, create the environment with `python3 -m venv .venv` and activate it with `source .venv/bin/activate`.

Internet access is needed for the first dataset download. Later runs can use the cached workbook. If the download fails, place UCI's workbook at `data/raw/default_of_credit_card_clients.xls`. The Excel reader, `xlrd`, is installed with the project.

Full runs save predictions, split memberships, SHAP values and the actual test inputs under `outputs/`. The repository contains the smaller result tables, charts, metadata and five individual explanations.

## Testing

Run the basic checks with:

```powershell
pytest -q
```

Enable the tests that use the real UCI workbook with:

```powershell
$env:CREDIT_RISK_REAL_DATA = "1"
pytest -q
```

The four original unit tests use small constructed arrays to check calculations and splitting functions. Those arrays are not used to produce the model results.

The two additional integration tests use the real workbook. They check that records do not overlap between groups, train and calibrate the boosted model, and check explanation reconstruction on actual test records. GitHub Actions runs these tests with real data enabled. The full experiment also checks the explanations across all 6,000 test records.

## Code and results

`data.py` downloads and checks the workbook. `modeling.py` contains the models, splits, calibration and evaluation functions. `explanations.py` creates the SHAP outputs. `cli.py` runs either one experiment or the repeated comparison.

Tests are in `tests/`. The published comparison is in `results/comparison/`.

## Limitations

The dataset is historical and comes from Taiwan. It has not been validated against current Irish or European lending data.

Random splits do not test performance on future customers. Records can appear in the test sets of more than one seed, so the reported ranges are not confidence intervals or independent repetitions.

The project does not include an extensive search for model settings, validation on another dataset or a statistically established model winner.

Age and demographic codes are included. No fairness audit or validation for real credit decisions has been carried out. Categories stored as integers are still treated numerically, as in the original project.

SHAP explanations depend on the model and assumptions about relationships between features. They do not establish causes or tell someone how to become creditworthy.

The explanations were not used to tune the models. Further changes need a fresh evaluation plan to avoid repeatedly adjusting models to the same test results.

This is a learning project, not a tool for making lending decisions.

## Data credit

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C55S3H](https://doi.org/10.24432/C55S3H).

The dataset is licensed under CC BY 4.0. The recorded results are derived from it.

[Gradient boosting documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

[SHAP documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
