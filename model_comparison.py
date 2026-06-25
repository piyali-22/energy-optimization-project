"""
model_comparison.py
---------------------
Trains several regression models on the SAME train/test split and compares
them on the SAME metrics, for both targets (production_time_min and
energy_kwh). This justifies why RandomForest was chosen instead of just
picking one model and assuming it's best.

Models compared:
  - Linear Regression       (simple baseline — is the relationship even linear?)
  - Decision Tree           (single tree — how much does ensembling help?)
  - Random Forest           (the one actually used in the deployed app)
  - Gradient Boosting       (sequential ensemble — does it beat bagging here?)
  - XGBoost                 (industry-standard gradient boosting implementation)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

df = pd.read_csv(DATA_DIR / "production_data.csv")

CATEGORICAL = ["machine_type", "product_type", "material_type", "shift", "day_of_week"]
NUMERIC_TIME = ["quantity", "operator_experience_years", "setup_complexity",
                 "machine_age_years", "ambient_temperature_c"]
NUMERIC_ENERGY = NUMERIC_TIME + ["production_time_min"]

MODELS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=4, random_state=42, verbosity=0),
}


def evaluate(target, numeric_features):
    X = df[CATEGORICAL + numeric_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )

    rows = []
    for name, model in MODELS.items():
        pipe = Pipeline([("preprocess", preprocessor), ("regressor", model)])

        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = pipe.predict(X_test)
        predict_time = time.time() - t0

        rows.append({
            "model": name,
            "R2": round(r2_score(y_test, y_pred), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
            "train_time_sec": round(train_time, 3),
            "predict_time_sec": round(predict_time, 4),
        })
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


time_results = evaluate("production_time_min", NUMERIC_TIME)
energy_results = evaluate("energy_kwh", NUMERIC_ENERGY)

print("=== Production Time Model Comparison ===")
print(time_results.to_string(index=False))
print("\n=== Energy Model Comparison ===")
print(energy_results.to_string(index=False))

time_results.to_csv(DATA_DIR / "model_comparison_time.csv", index=False)
energy_results.to_csv(DATA_DIR / "model_comparison_energy.csv", index=False)

# ---- Plot: R² comparison, both targets side by side ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, results, title in [(axes[0], time_results, "Production Time — R² by Model"),
                            (axes[1], energy_results, "Energy Consumption — R² by Model")]:
    colors = ["#1F7A6C" if m == "Random Forest" else "#C9711F" for m in results["model"]]
    ax.barh(results["model"][::-1], results["R2"][::-1], color=colors[::-1])
    ax.set_xlabel("R² score")
    ax.set_xlim(0, 1.05)
    ax.set_title(title)
    for i, v in enumerate(results["R2"][::-1]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
plt.close()

print("\nSaved: model_comparison_time.csv, model_comparison_energy.csv, plots/model_comparison.png")
