"""
train_model.py
---------------
Trains a RandomForestRegressor to predict production_time_min from job
features. Saves the trained model + encoders, plus evaluation plots.
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("/home/claude/project/data/production_data.csv")

target = "production_time_min"
categorical_features = ["machine_type", "product_type", "material_type", "shift", "day_of_week"]
numeric_features = ["quantity", "operator_experience_years", "setup_complexity",
                     "machine_age_years", "ambient_temperature_c"]

X = df[categorical_features + numeric_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

metrics = {"MAE_minutes": round(mae, 2), "RMSE_minutes": round(rmse, 2),
           "R2_score": round(r2, 4), "MAPE_percent": round(mape, 2)}
print("Model performance on test set:")
print(json.dumps(metrics, indent=2))

with open("/home/claude/project/models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(model, "/home/claude/project/models/production_time_model.pkl")

# ---- Plot 1: Actual vs Predicted ----
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=12, color="#2563eb")
lims = [0, max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, color="#dc2626", linestyle="--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Production Time (min)")
plt.ylabel("Predicted Production Time (min)")
plt.title(f"Actual vs Predicted Production Time (R² = {r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/project/plots/actual_vs_predicted.png", dpi=150)
plt.close()

# ---- Plot 2: Feature importance ----
ohe = model.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
all_feature_names = cat_feature_names + numeric_features
importances = model.named_steps["regressor"].feature_importances_

imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#0d9488")
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances - Production Time Model")
plt.tight_layout()
plt.savefig("/home/claude/project/plots/feature_importance.png", dpi=150)
plt.close()

print("\nSaved: model, metrics.json, actual_vs_predicted.png, feature_importance.png")
