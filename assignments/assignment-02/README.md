# Assignment 2 — Automated ML Pipelines & Model Serving

**Course:** OIM3641 — AI Driven App Development (Spring 2026)
**Author:** Joshua Bell

This project takes the role of a Data Engineer comparing **PyCaret** (low-code) against
**scikit-learn** (manual) on the same classification problem, then ships the winning
model behind a **FastAPI** service.

## Dataset

[Bank Marketing](https://archive.ics.uci.edu/dataset/222) (UCI ID **222**) —
45,211 client records from a Portuguese bank's direct-marketing campaigns.
The target column `y` is binary: did the client subscribe to a term deposit?

- 16 features (7 numeric, 9 categorical)
- Sub-sampled to 8,000 rows in `discovery.py` to keep `compare_models` tractable on a laptop
  (still well above the 1,000-row requirement)

## Codebase

| File | What it does |
| --- | --- |
| `discovery.py` | Loads the dataset via `ucimlrepo`, runs PyCaret's `setup` → `compare_models` → `plot_model`, then re-implements the winning model manually with sklearn (`ColumnTransformer` + `train_test_split` + `classification_report`). Saves `best_pipeline.pkl`. |
| `main.py` | FastAPI app with `POST /predict`. Loads the saved PyCaret pipeline at startup and returns `{prediction, score}` for a JSON client record. |
| `requirements.txt` | Pinned versions for PyCaret 3.3, FastAPI, sklearn, ucimlrepo, etc. |
| `pycaret_leaderboard.csv` | Top-10 model leaderboard from `compare_models`. |
| `sklearn_classification_report.txt` | Per-class metrics for the manual sklearn pipeline. |
| `Confusion Matrix.png` | PyCaret-generated confusion matrix for the best model. |
| `best_pipeline.pkl` | Persisted PyCaret pipeline (preprocessing + estimator). |

## Outcomes

PyCaret's leaderboard ranks (top 3, sorted by accuracy):

| Rank | Model | Accuracy | AUC | F1 |
| --- | --- | --- | --- | --- |
| 1 | **Random Forest Classifier** | **0.8998** | 0.9087 | 0.8819 |
| 2 | Light Gradient Boosting Machine | 0.8988 | 0.9100 | 0.8913 |
| 3 | Gradient Boosting Classifier | 0.8966 | 0.9083 | 0.8852 |

The manual scikit-learn replication of Random Forest on an 80/20 split scored
**0.8975 accuracy** — within ~0.2 pts of PyCaret's 10-fold CV mean, which
matches expectations given the methodological differences (single split vs.
10-fold, different categorical encoders). The 200-word PyCaret-vs-sklearn
synthesis lives in the `# SYNTHESIS` comment block at the bottom of
`discovery.py`.

The dataset is heavily imbalanced (~88% "no"), so accuracy is a misleading
headline metric — the classification report shows the "yes" class recall is
the real challenge (0.35), which is exactly the kind of insight a Data
Engineer surfaces before handing off for tuning.

## Running it

```bash
# from this directory
uv venv --python 3.11 .venv               # PyCaret needs Python 3.9–3.11
uv pip install -r requirements.txt --python .venv/Scripts/python.exe

# 1. Train + compare + save the model (~2 min)
.venv/Scripts/python.exe discovery.py

# 2. Serve predictions
.venv/Scripts/python.exe -m uvicorn main:app --reload
# Swagger UI at http://127.0.0.1:8000/docs
```

## Sample API call

**Request**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 29, "job": "student", "marital": "single",
    "education": "tertiary", "default": "no", "balance": 500,
    "housing": "no", "loan": "no", "contact": "cellular",
    "day_of_week": 4, "month": "oct", "duration": 900,
    "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"
  }'
```

**Response**

```json
{"prediction": "yes", "score": 0.68}
```

A more conservative client profile flips the prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 41, "job": "management", "marital": "married",
    "education": "tertiary", "default": "no", "balance": 1500,
    "housing": "yes", "loan": "no", "contact": "cellular",
    "day_of_week": 15, "month": "may", "duration": 250,
    "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown"
  }'
```

```json
{"prediction": "no", "score": 0.86}
```

The `score` is the model's probability for the predicted class. Both responses
were captured live from the FastAPI service running on the saved
`best_pipeline.pkl`.

## AI disclosure

Used Claude (Anthropic) to scaffold the pipeline structure and FastAPI types.
All design choices, dataset selection, and interpretation of results are mine.
