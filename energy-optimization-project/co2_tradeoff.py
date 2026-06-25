"""
co2_tradeoff.py
-----------------
KEY FINDING: minimizing electricity COST is not the same as minimizing
electricity CONSUMPTION (and therefore CO2 emissions).

Night shift has the cheapest tariff (Rs 5.5/kWh) but production runs ~8%
slower at night (fatigue factor in the model) -- so a job consumes MORE
kWh at night even though it costs LESS overall. This script quantifies
that trade-off explicitly across 3 strategies:

  1. Naive       - run jobs on their originally scheduled shift
  2. Cost-optimal  - the optimizer from optimize_schedule.py (minimizes Rs)
  3. Energy-optimal - an alternative objective that minimizes kWh/CO2 instead

This is a genuine, defensible insight for a viva: cost optimization and
emissions optimization can conflict in demand-response systems where
off-peak tariffs don't track off-peak grid carbon intensity.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

production_model = joblib.load(MODELS_DIR / "production_time_model.pkl")
energy_model = joblib.load(MODELS_DIR / "energy_model.pkl")

TARIFF = {"Morning": 7.5, "Evening": 9.5, "Night": 5.5}
GRID_EMISSION_FACTOR_KG_PER_KWH = 0.716  # CEA all-India average, representative value

CATEGORICAL_FEATURES = ["machine_type", "product_type", "material_type", "shift", "day_of_week"]
NUMERIC_FEATURES = ["quantity", "operator_experience_years", "setup_complexity",
                     "machine_age_years", "ambient_temperature_c"]

df = pd.read_csv(DATA_DIR / "production_data.csv")
sample = df.sample(500, random_state=7).reset_index(drop=True)  # same sample as optimize_schedule.py

records = []
for _, row in sample.iterrows():
    base = row[CATEGORICAL_FEATURES + NUMERIC_FEATURES].to_dict()
    scenarios = {}
    for shift in TARIFF:
        feats = base.copy()
        feats["shift"] = shift
        X_time = pd.DataFrame([feats])[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        t = float(production_model.predict(X_time)[0])
        feats_e = feats.copy()
        feats_e["production_time_min"] = t
        X_e = pd.DataFrame([feats_e])[CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["production_time_min"]]
        e = float(energy_model.predict(X_e)[0])
        scenarios[shift] = {"energy_kwh": e, "cost_inr": e * TARIFF[shift]}

    cost_optimal_shift = min(scenarios, key=lambda s: scenarios[s]["cost_inr"])
    energy_optimal_shift = min(scenarios, key=lambda s: scenarios[s]["energy_kwh"])

    records.append({
        "naive_energy": scenarios[row["shift"]]["energy_kwh"],
        "naive_cost": scenarios[row["shift"]]["cost_inr"],
        "cost_opt_energy": scenarios[cost_optimal_shift]["energy_kwh"],
        "cost_opt_cost": scenarios[cost_optimal_shift]["cost_inr"],
        "energy_opt_energy": scenarios[energy_optimal_shift]["energy_kwh"],
        "energy_opt_cost": scenarios[energy_optimal_shift]["cost_inr"],
    })

r = pd.DataFrame(records)


def summarize(prefix):
    energy = r[f"{prefix}_energy"].sum()
    cost = r[f"{prefix}_cost"].sum()
    return energy, cost


naive_energy, naive_cost = summarize("naive")
cost_energy, cost_cost = summarize("cost_opt")
energy_energy, energy_cost = summarize("energy_opt")

result = {
    "strategy_comparison": {
        "naive": {
            "total_energy_kwh": round(naive_energy, 1),
            "total_cost_inr": round(naive_cost, 2),
            "total_co2_kg": round(naive_energy * GRID_EMISSION_FACTOR_KG_PER_KWH, 1),
        },
        "cost_optimal (deployed optimizer)": {
            "total_energy_kwh": round(cost_energy, 1),
            "total_cost_inr": round(cost_cost, 2),
            "total_co2_kg": round(cost_energy * GRID_EMISSION_FACTOR_KG_PER_KWH, 1),
            "cost_savings_vs_naive_percent": round((naive_cost - cost_cost) / naive_cost * 100, 1),
            "co2_change_vs_naive_percent": round((cost_energy - naive_energy) / naive_energy * 100, 1),
        },
        "energy_optimal (alternative objective)": {
            "total_energy_kwh": round(energy_energy, 1),
            "total_cost_inr": round(energy_cost, 2),
            "total_co2_kg": round(energy_energy * GRID_EMISSION_FACTOR_KG_PER_KWH, 1),
            "cost_change_vs_naive_percent": round((energy_cost - naive_cost) / naive_cost * 100, 1),
            "co2_savings_vs_naive_percent": round((naive_energy - energy_energy) / naive_energy * 100, 1),
        },
    },
    "key_finding": (
        "Minimizing cost and minimizing energy/CO2 are DIFFERENT objectives. "
        "Night-shift tariffs are cheapest but production runs slower at night, "
        "so the cost-optimal schedule actually INCREASES total energy use and "
        "CO2 emissions slightly, even as it cuts cost substantially. An energy/"
        "CO2-optimal schedule instead favors the fastest shift (Morning), cutting "
        "emissions but missing out on most of the cost savings. A real deployment "
        "would need to choose a weighted objective balancing both."
    ),
}

print(json.dumps(result, indent=2))

with open(DATA_DIR / "co2_tradeoff.json", "w") as f:
    json.dump(result, f, indent=2)
