# Pulse

A responsive heart-health interface with a trained, browser-based coronary disease classifier. Plain-language choice cards show the matching medical-report terms, with contextual help and explicit handling of missing results.

## Run locally

Requires Python 3 for the static server. No application dependencies or build step are needed.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Open http://127.0.0.1:8765. Use **Try a fictional sample** to explore a complete prediction without personal health data.

## Verify the predictor

Requires Node.js 18 or later.

```sh
node --test tests/predictor.test.cjs
```

Checks 60 held-out Python/browser prediction comparisons, contribution reconstruction, rejected missing/invalid inputs, and a disjoint train/test split.

## Model and data

The app uses an L2-regularized logistic regression trained on the public UCI Cleveland Heart Disease dataset. Of 303 source records, 297 are complete: 237 are used for training and 60 for a held-out evaluation. The deployed model achieved 81.7% accuracy on that small internal test split. See `model/README.md` for training instructions, provenance and limitations, and `dist/model.json` for the exact fitted coefficients and metrics.

- `dist/`: deployable static app, model, inference engine and downloadable training source
- `model/`: reproducible training script, public source data and split
- `tests/`: predictor checks and Python reference outputs

Dataset: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), Janosi, Steinbrunn, Pfisterer and Detrano, [DOI 10.24432/C52P4X](https://doi.org/10.24432/C52P4X), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Pulse transforms the dataset into an independently trained model; the dataset authors do not endorse the app. Source data contains the published research records, not app users’ inputs.

## Intended use and privacy

This is an educational, experimental classifier, not a clinical diagnostic service. Its target is the dataset’s coronary angiographic disease label; it does not detect every heart condition or estimate future risk. Its score is not a clinically validated personal disease probability. False negatives and false positives occur. Do not change treatment or delay care based on its output.

All 13 model inputs require values from existing clinical records. Missing results are never silently replaced with normal values. The interface offers plain-language descriptions alongside report terminology; users should match the report rather than interpret a medical test themselves.

Inputs and prediction history remain in memory in the current browser tab. Predictions are computed locally. Downloaded reports contain the values entered and should be stored privately. The app requests Google Fonts for typography; it does not send assessment values with those requests.

The static app can be hosted by serving `dist/`. The original Sites hosting identity is omitted from this public repository so it can be deployed independently.
