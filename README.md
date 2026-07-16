# 🍷 Wine Quality Regressor

Streamlit app that predicts the quality score of a red wine from its chemical composition, explains *why* it got that score using SHAP values, and then asks Gemini to turn the explanation into a sommelier review.

The prediction is a **regression**, not a classification — the model outputs a continuous score (e.g. `5.99`) rather than a "good/bad" label.

## Features

- **Quality prediction** from the 11 physicochemical features of the UCI Wine Quality dataset.
- **SHAP explainability** — a Plotly bar chart showing the top 3 features that pushed the score up or down, computed via XGBoost's native `pred_contribs`.
- **AI Sommelier** — Gemini receives the chemical profile plus the SHAP drivers and writes a short review in Portuguese explaining the score.

## Model

`XGBRegressor` (500 estimators, `max_depth=6`, `learning_rate=0.05`), trained on the 1,599 red wines of the UCI Wine Quality dataset with an 80/20 split.

| Metric | Value |
|---|---|
| R² | 0.519 |
| RMSE | 0.561 |

The model explains about 51% of the variance and is off by roughly 0.56 points on average — so for a wine whose true quality is 6, it will usually guess somewhere between 5.44 and 6.56. This is a demo-grade model, not a production one.

Training also included a `RandomForestClassifier` baseline (67% accuracy) and an `XGBClassifier` on SMOTE-balanced quality groups (88% accuracy, but only 0.09 F1 on the rare "bad" class). The regressor is what ships in the app. The full process is in [WineQuality.ipynb](WineQuality.ipynb).

### Feature engineering

Three ratios are derived from the raw inputs and must be present at inference time, since the model was trained on 14 features:

```
acidity_ratio   = volatile acidity / fixed acidity
sulfur_ratio    = free sulfur dioxide / total sulfur dioxide
alcohol_density = alcohol / density
```

The app computes these automatically from the sliders — the column order in `input_data` matches the booster's `feature_names` exactly, which XGBoost requires.

## Setup

```bash
git clone <repo-url>
cd WineQuality_Project
pip install -r requirements.txt
```

### Gemini API key

The SHAP chart and the prediction work without a key; only the AI Sommelier section needs one. Get a key at [Google AI Studio](https://aistudio.google.com/apikey), then either create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

or export it:

```bash
export GEMINI_API_KEY="your-key-here"
```

> **Do not commit `.streamlit/secrets.toml`.** Make sure it's in your `.gitignore` before pushing.

## Running

```bash
python -m streamlit run wine_app.py
```

Use `python -m streamlit` rather than bare `streamlit` — it guarantees the app runs under the interpreter of the currently active environment. Running it with the wrong interpreter is what produces `ModuleNotFoundError: No module named 'xgboost'` when loading `wine_model.pkl`: the pickle imports XGBoost at unpickle time, so the error surfaces from `joblib.load` rather than from an import line.

## Notes on the Gemini call

The API returns `503 UNAVAILABLE` ("high demand") intermittently on **every** model, including the `lite` tier — it is not specific to any one model, and there is no model you can switch to in order to avoid it. The app handles this with a bounded retry: up to 3 attempts per model with a 1s/2s backoff, falling back through `GEMINI_MODELS` in order. Worst case is 6 attempts over ~6 seconds before it gives up and shows the error.

To use a different model, edit `GEMINI_MODELS` at the top of [wine_app.py](wine_app.py).

## Project structure

```
├── wine_app.py           # Streamlit app
├── wine_model.pkl        # Trained XGBRegressor (14 features)
├── WineQuality.ipynb     # EDA, feature engineering, model comparison
├── requirements.txt
└── .streamlit/
    ├── config.toml       # Theme
    └── secrets.toml      # GEMINI_API_KEY (do not commit)
```

## Dataset

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality*. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T

The notebook reads the CSV from a local path (`~/Desktop/Datasets/winequality-red.csv`) that is not part of this repo — download it from UCI and adjust the path if you want to retrain.
