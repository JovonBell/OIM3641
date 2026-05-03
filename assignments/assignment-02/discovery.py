"""
discovery.py - Major Assignment 2: Automated ML Pipelines & Model Serving
Author: Joshua Bell - OIM3641 (Spring 2026)

Dataset: Bank Marketing (UCI ID 222) - https://archive.ics.uci.edu/dataset/222
Business problem: Predict whether a client will subscribe to a term deposit
(target column 'y', yes/no) based on demographic + campaign-contact features.

This script runs two parallel ML workflows on the same dataset:
  1. PyCaret  (low-code) -- setup -> compare_models -> plot_model -> save_model
  2. Sklearn  (manual)   -- ColumnTransformer + train_test_split + classification_report

The PyCaret best model is saved as 'best_pipeline.pkl' for FastAPI serving in main.py.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from ucimlrepo import fetch_ucirepo

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(OUTPUT_DIR)  # so PyCaret's save=True writes plots/models here

# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
print("Loading Bank Marketing dataset from UCI...")
bank = fetch_ucirepo(id=222)
X = bank.data.features
y = bank.data.targets
df = pd.concat([X, y], axis=1)
target_col = y.columns[0]                     # 'y'
df = df.dropna(subset=[target_col])
print(f"Loaded {len(df):,} rows x {df.shape[1]} columns. Target: '{target_col}'")
print(df[target_col].value_counts().to_string())

# Sub-sample to keep compare_models tractable on a laptop
# (still well above the 1,000-row requirement)
SAMPLE_SIZE = 8000
if len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Sampled to {len(df):,} rows for tractable training")

# -----------------------------------------------------------------------------
# PART 1: PyCaret (low-code) workflow
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PYCARET WORKFLOW")
print("=" * 70)

from pycaret.classification import (
    setup,
    compare_models,
    plot_model,
    save_model,
    pull,
)

setup(
    data=df,
    target=target_col,
    session_id=42,
    verbose=False,
    html=False,
)

# Top 3 performers from the leaderboard (sorted by accuracy by default)
top3 = compare_models(n_select=3, verbose=False)
leaderboard = pull()
print("\nPyCaret leaderboard (top of compare_models):")
print(leaderboard.head(10).to_string())
leaderboard.head(10).to_csv("pycaret_leaderboard.csv", index=False)

best_model = top3[0] if isinstance(top3, list) else top3
best_name = type(best_model).__name__
print(f"\nBest PyCaret model: {best_name}")

# Confusion matrix on the held-out PyCaret split
try:
    plot_model(best_model, plot="confusion_matrix", save=True)
    print("Saved confusion matrix plot to OUTPUT_DIR")
except Exception as e:
    print(f"Confusion matrix plot failed: {e}")

# Persist the full PyCaret pipeline for FastAPI to load
save_model(best_model, "best_pipeline")
print("Saved best_pipeline.pkl")

# -----------------------------------------------------------------------------
# PART 2: Scikit-learn manual workflow (replicates PyCaret's best estimator)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SCIKIT-LEARN MANUAL WORKFLOW")
print("=" * 70)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Map PyCaret's chosen class name -> a sklearn equivalent
estimator_map = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42, n_jobs=-1),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "GaussianNB": GaussianNB(),
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
}
# Boosted-tree fall-back (LightGBM / XGBoost / CatBoost all map to GBM here)
boosted_tokens = ("LGBM", "XGB", "CatBoost", "Booster")
if best_name in estimator_map:
    sklearn_est = estimator_map[best_name]
elif any(tok in best_name for tok in boosted_tokens):
    sklearn_est = GradientBoostingClassifier(random_state=42)
else:
    sklearn_est = LogisticRegression(max_iter=1000, random_state=42)

print(f"Replicating PyCaret best ({best_name}) with: {type(sklearn_est).__name__}")

X_df = df.drop(columns=[target_col])
y_series = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
)

numeric_cols = X_df.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X_df.columns if c not in numeric_cols]
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

manual_pipe = Pipeline([("prep", preprocessor), ("clf", sklearn_est)])
manual_pipe.fit(X_train, y_train)
preds = manual_pipe.predict(X_test)

report = classification_report(y_test, preds, digits=4)
print("\nClassification report (sklearn manual pipeline):")
print(report)

with open("sklearn_classification_report.txt", "w") as f:
    f.write(f"Sklearn manual pipeline -- {type(sklearn_est).__name__}\n")
    f.write("=" * 70 + "\n")
    f.write(report)

# -----------------------------------------------------------------------------
# SYNTHESIS - 200-word summary
# -----------------------------------------------------------------------------
# For this Bank Marketing classification task, the PyCaret (low-code) workflow
# was meaningfully more efficient than the manual scikit-learn workflow. A single
# setup() call inferred numeric vs. categorical columns, applied imputation,
# scaling, encoding, and a stratified train/test split automatically, then
# compare_models() trained and 10-fold cross-validated more than a dozen
# classifiers in roughly the time it took to write the manual ColumnTransformer.
# For a data engineer triaging which family of models even merits tuning, that
# breadth-first sweep is enormous leverage: minutes of code replace what would
# otherwise be a day of boilerplate.
#
# The manual scikit-learn pipeline trained the single best estimator that
# PyCaret identified, but using slightly different defaults: a single 80/20
# split instead of 10-fold CV, vanilla hyperparameters, no feature selection,
# and one-hot encoding for every categorical (PyCaret often selects target or
# ordinal encoders). That is why the headline metrics rarely match exactly:
# PyCaret reports averaged CV accuracy across 10 folds, while the sklearn
# report reflects a single held-out split, so stochastic estimators plus the
# preprocessing differences compound the gap.
#
# When PyCaret and sklearn disagree, sklearn's transparency wins for production
# since every transform is auditable. PyCaret wins for discovery.

print("\nDone. Artifacts written to:", OUTPUT_DIR)
