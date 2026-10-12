# Pulse coronary disease research classifier

This is an experimental classifier for the binary coronary disease label in the UCI Cleveland dataset. It is not clinically validated, does not detect every form of heart disease, and does not predict future cardiovascular events.

## Reproduce

Use Python 3.12 in an isolated environment. From the project root:

    python -m pip install -r model/requirements.txt
    python model/train.py
    node tests/predictor.test.cjs

Training writes dist/model.json, tests/parity-fixtures.json, and model/split.json. The shipped model is the same fitted model evaluated on the held-out test set; no later refit uses the test data. There is no parameter or threshold search on the test set.

## Data and pipeline

- Source: processed.cleveland.data, 303 records; 6 incomplete records excluded; 297 complete records.
- Label: num > 0 versus num = 0, reflecting coronary angiographic disease status.
- Stratified 80/20 split with seed 42: 237 training and 60 test records.
- Numeric standardization is fitted on training data only. The 8 categorical fields are one-hot encoded with the documented UCI categories.
- L2 logistic regression, C=1.0, lbfgs, max_iter=2000. Decision threshold 0.5.
- Numeric input bounds reflect the observed complete dataset envelope. They are not normal clinical ranges. No imputation is performed at inference; missing or unsupported inputs are rejected.
- Feature explanations are grouped coefficient contributions relative to the encoded training mean, in log-odds units. They describe the model computation, not causal medical effects.

## Evaluation and limitations

See dist/model.json for the full metrics, source checksum, versions, and configuration. The single random internal holdout is small. No external, prospective, subgroup, or clinical validation has been performed. Positive-class scores are not externally calibrated personal disease probabilities. The source population is historical, selected, and limited in size. A negative prediction cannot rule out disease; no result should change treatment or delay care.

The UI's demo values are a fictional profile, not a real user's health record. All input processing and session history remain in page memory. The hosted site only serves the trained parameters and app assets.

## Attribution

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
Archive: https://archive.ics.uci.edu/static/public/45/heart%2Bdisease.zip
License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/

Pulse cleans, transforms, and trains on this dataset. The original authors do not endorse this app. The training-data checksum is stored in model.json.
