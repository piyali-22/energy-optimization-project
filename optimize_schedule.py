"""
optimize_schedule.py
---------------------
The "optimization" layer of the project.

For every job, this simulates running it on each available shift
(Morning / Evening / Night), uses the two trained ML models to predict
production_time and energy_kwh under each scenario, applies a
time-of-day electricity tariff, and recommends the shift with the
lowest total energy COST -- then compares that against the shift the
job was originally (naively) scheduled on.

This is a deliberately simple, explainable optimization strategy
(per-job greedy search over 3 discrete options) rather than a complex
solver -- appropriate for an internship-scale project, and easy to
explain in an interview.
"""

import pandas as pd
import numpy as np
import joblib
import json

production_model = joblib.load("/home/claude/project/models/production_time_model.pkl")
energy_model = joblib.load("/home/claude/project/models/energy_model.pkl")

df = pd.read_csv("/home/claude/project/data/production_data.csv")

# Time-of-day industrial electricity tariff (INR per kWh) -- realistic ToD slab pattern
TARIFF = {"Morning": 7.5, "Evening": 9.5, "Night": 5.5}

categorical_features = ["machine_type", "product_type", "material_type", "shift", "day_of_week"]
numeric_features = ["quantity", "operator_experience_years", "setup_complexity",
                     "machine_age_years", "ambient_temperature_c"]

sample = df.sample(500, random_state=7).reset_index(drop=True)

results = []
for _, row in sample.iterrows():
    base_features = row[categorical_features + numeric_features].to_dict()

    scenario_costs = {}
    for shift in TARIFF:
        feats = base_features.copy()
        feats["shift"] = shift
        X_time = pd.DataFrame([feats])[categorical_features + numeric_features]
        pred_time = production_model.predict(X_time)[0]

        feats_energy = feats.copy()
        feats_energy["production_time_min"] = pred_time
        X_energy = pd.DataFrame([feats_energy])[categorical_features + numeric_features + ["production_time_min"]]
        pred_energy = energy_model.predict(X_energy)[0]

        cost = pred_energy * TARIFF[shift]
        scenario_costs[shift] = {"pred_time_min": round(pred_time, 1),
                                  "pred_energy_kwh": round(pred_energy, 2),
                                  "cost_inr": round(cost, 2)}

    best_shift = min(scenario_costs, key=lambda s: scenario_costs[s]["cost_inr"])
    original_shift = row["shift"]

    results.append({
        "job_id": row["job_id"],
        "machine_type": row["machine_type"],
        "original_shift": original_shift,
        "original_cost_inr": scenario_costs[original_shift]["cost_inr"],
        "recommended_shift": best_shift,
        "recommended_cost_inr": scenario_costs[best_shift]["cost_inr"],
        "savings_inr": round(scenario_costs[original_shift]["cost_inr"] - scenario_costs[best_shift]["cost_inr"], 2),
    })

res_df = pd.DataFrame(results)
res_df.to_csv("/home/claude/project/data/optimization_results.csv", index=False)

total_original = res_df["original_cost_inr"].sum()
total_optimized = res_df["recommended_cost_inr"].sum()
total_savings = total_original - total_optimized
pct_savings = (total_savings / total_original) * 100
jobs_with_savings = (res_df["savings_inr"] > 0).sum()

summary = {
    "jobs_simulated": len(res_df),
    "jobs_with_cost_reduction": int(jobs_with_savings),
    "total_original_cost_inr": round(total_original, 2),
    "total_optimized_cost_inr": round(total_optimized, 2),
    "total_savings_inr": round(total_savings, 2),
    "percent_cost_reduction": round(pct_savings, 2),
}
print(json.dumps(summary, indent=2))

with open("/home/claude/project/data/optimization_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
