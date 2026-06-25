"""
train_energy_model.py
----------------------
Trains a second RandomForestRegressor to predict energy_kwh.
Uses the same job features PLUS production_time_min (which in real
deployment would come from the production-time model's prediction,
chaining the two models together).
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

target = "energy_kwh"
categorical_features = ["machine_type", "product_type", "material_type", "shift", "day_of_week"]
numeric_features = ["quantity", "operator_experience_years", "setup_complexity",
                     "machine_age_years", "ambient_temperature_c", "production_time_min"]

X = df[categorical_features + numeric_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
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

metrics = {"MAE_kwh": round(mae, 2), "RMSE_kwh": round(rmse, 2), "R2_score": round(r2, 4)}
print("Energy model performance on test set:")
print(json.dumps(metrics, indent=2))

with open("/home/claude/project/models/energy_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(model, "/home/claude/project/models/energy_model.pkl")
print("Saved energy_model.pkl")
