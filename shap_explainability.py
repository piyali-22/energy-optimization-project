"""
shap_explainability.py
------------------------
Generates SHAP (SHapley Additive exPlanations) plots for the production
time model, so predictions aren't a black box.

Produces two things:
  1. A global summary plot: which features matter most ACROSS ALL jobs,
     and whether high/low values of each feature push the prediction up
     or down.
  2. A single-job waterfall: for ONE specific job, exactly how each
     feature contributed to that prediction (e.g. "batch size added 180
     minutes, but high operator experience subtracted 40 minutes").
"""

from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

pipeline = joblib.load(MODELS_DIR / "production_time_model.pkl")
preprocessor = pipeline.named_steps["preprocess"]
regressor = pipeline.named_steps["regressor"]

CATEGORICAL = ["machine_type", "product_type", "material_type", "shift", "day_of_week"]
NUMERIC = ["quantity", "operator_experience_years", "setup_complexity",
           "machine_age_years", "ambient_temperature_c"]

df = pd.read_csv(DATA_DIR / "production_data.csv")
sample = df.sample(200, random_state=21)
X_sample = sample[CATEGORICAL + NUMERIC]

# Transform to the same encoded feature space the model actually sees
X_encoded = preprocessor.transform(X_sample)
feature_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)) + NUMERIC

# SHAP needs a dense array and a DataFrame with matching column names
import numpy as np
if hasattr(X_encoded, "toarray"):
    X_encoded = X_encoded.toarray()
X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

explainer = shap.TreeExplainer(regressor)
shap_values = explainer(X_encoded_df)

# ---- Plot 1: Global feature importance (mean |SHAP value|) ----------------
plt.figure()
shap.plots.bar(shap_values, max_display=12, show=False)
plt.title("What drives the production time prediction? (mean impact across 200 jobs)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Plot 2: Beeswarm (shows direction, not just magnitude) ----------------
plt.figure()
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title("Feature impact and direction (red = increases time, blue = decreases)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Plot 3: Single-job waterfall — explain ONE concrete prediction -------
example_idx = 0
example_row = sample.iloc[example_idx]
print("Explaining this specific job:")
print(example_row[CATEGORICAL + NUMERIC].to_string())
print(f"\nActual production_time_min in data: {example_row['production_time_min']:.1f}")
print(f"Model's predicted value (base + contributions): {shap_values[example_idx].base_values + shap_values[example_idx].values.sum():.1f}")

plt.figure()
shap.plots.waterfall(shap_values[example_idx], max_display=10, show=False)
plt.title(f"Why this job (#{example_row['job_id']}) was predicted at this time")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved: plots/shap_summary_bar.png, plots/shap_beeswarm.png, plots/shap_waterfall_example.png")
