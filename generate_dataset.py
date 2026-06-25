"""
generate_dataset.py
--------------------
Generates a synthetic but realistic manufacturing production dataset.

Why synthetic: real factory-floor data is rarely available to an intern.
This dataset is built using domain-reasonable formulas (not random noise)
so the relationships between features and production time / energy are
genuinely learnable -- exactly like real-world data would behave.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 4000  # number of production job records

# ---- Categorical domains -------------------------------------------------
machine_types = ["CNC", "Lathe", "Press", "Welding", "Assembly"]
product_types = ["A", "B", "C", "D", "E"]
material_types = ["Steel", "Aluminum", "Plastic", "Composite"]
shifts = ["Morning", "Evening", "Night"]
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Base processing rate (minutes per unit) for each machine type
machine_base_rate = {"CNC": 2.2, "Lathe": 1.8, "Press": 1.1, "Welding": 2.6, "Assembly": 1.4}

# Power rating (kW) when machine is actively running, per machine type
machine_power_kw = {"CNC": 7.5, "Lathe": 5.0, "Press": 9.0, "Welding": 11.0, "Assembly": 3.5}

# Difficulty multiplier per material (harder material = slower + more energy)
material_difficulty = {"Steel": 1.25, "Aluminum": 1.05, "Plastic": 0.85, "Composite": 1.15}

# Setup energy (kWh) — fixed cost paid once per job regardless of quantity
machine_setup_energy_kwh = {"CNC": 1.2, "Lathe": 0.8, "Press": 1.5, "Welding": 2.0, "Assembly": 0.5}

rows = []
for i in range(N):
    machine_type = np.random.choice(machine_types)
    product_type = np.random.choice(product_types)
    material_type = np.random.choice(material_types)
    shift = np.random.choice(shifts, p=[0.45, 0.35, 0.20])
    day = np.random.choice(days)

    quantity = np.random.randint(10, 501)
    operator_experience_years = np.random.randint(0, 16)
    setup_complexity = np.random.randint(1, 6)        # 1 (simple) - 5 (very complex)
    machine_age_years = np.random.randint(1, 11)
    ambient_temperature_c = np.round(np.random.normal(28, 4), 1)

    # ---- Production time formula (minutes) ----
    base = machine_base_rate[machine_type] * quantity
    base *= material_difficulty[material_type]

    # Setup overhead scales with complexity (fixed time, independent of quantity)
    setup_overhead = setup_complexity * 8

    # Experienced operators are faster (up to ~20% faster at 15 yrs experience)
    experience_factor = 1 - (operator_experience_years / 15) * 0.20

    # Older machines run slower (up to ~15% slower at 10 yrs)
    age_factor = 1 + (machine_age_years / 10) * 0.15

    # Night shift slightly slower (fatigue), evening neutral, morning fastest
    shift_factor = {"Morning": 1.0, "Evening": 1.03, "Night": 1.08}[shift]

    # Mild heat penalty above 30C
    temp_factor = 1 + max(0, ambient_temperature_c - 30) * 0.01

    noise = np.random.normal(1.0, 0.05)  # +/-5% real-world noise

    production_time_min = (base * experience_factor * age_factor * shift_factor * temp_factor + setup_overhead) * noise
    production_time_min = max(5, round(production_time_min, 1))

    # ---- Energy consumption (kWh) — derived from time + machine power ----
    running_hours = production_time_min / 60
    energy_kwh = running_hours * machine_power_kw[machine_type] * material_difficulty[material_type] ** 0.5
    energy_kwh += machine_setup_energy_kwh[machine_type]
    energy_kwh *= np.random.normal(1.0, 0.04)
    energy_kwh = round(max(0.1, energy_kwh), 2)

    rows.append({
        "job_id": f"J{i+1:05d}",
        "machine_type": machine_type,
        "product_type": product_type,
        "material_type": material_type,
        "quantity": quantity,
        "operator_experience_years": operator_experience_years,
        "setup_complexity": setup_complexity,
        "machine_age_years": machine_age_years,
        "ambient_temperature_c": ambient_temperature_c,
        "shift": shift,
        "day_of_week": day,
        "production_time_min": production_time_min,
        "energy_kwh": energy_kwh,
    })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/project/data/production_data.csv", index=False)
print(df.head(10).to_string())
print("\nShape:", df.shape)
print("\nDescribe (target columns):")
print(df[["production_time_min", "energy_kwh"]].describe())
