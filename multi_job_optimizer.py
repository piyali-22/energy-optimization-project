"""
multi_job_optimizer.py
------------------------
Upgrade over optimize_schedule.py: instead of picking the cheapest shift
for each job INDEPENDENTLY (which can overload a station beyond its real
shift capacity), this solves a proper constrained optimization problem:

    Minimize total energy cost across ALL jobs in a day's batch,
    subject to: each station can only handle a limited number of
    machine-minutes per shift (it has finite capacity — you can't run
    50 jobs on one CNC machine in an 8-hour shift).

This is a Mixed-Integer Linear Program (MILP), solved with PuLP (CBC solver).
This is the difference between "per-job heuristic" and "real scheduling
optimization" — the kind of problem an Operations Research / industrial
engineering team would actually solve.
"""

import json
import joblib
import pandas as pd
import pulp

from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

production_model = joblib.load(MODELS_DIR / "production_time_model.pkl")
energy_model = joblib.load(MODELS_DIR / "energy_model.pkl")

TARIFF = {"Morning": 7.5, "Evening": 9.5, "Night": 5.5}
SHIFTS = list(TARIFF.keys())

# Shift length: 8 hours = 480 minutes. "Lines" = how many parallel machines
# of that station type the plant has — more lines = more capacity per shift.
STATION_LINES = {"CNC": 2, "Lathe": 2, "Press": 2, "Welding": 2, "Assembly": 3}
SHIFT_MINUTES = 480
STATION_CAPACITY = {s: lines * SHIFT_MINUTES for s, lines in STATION_LINES.items()}

CATEGORICAL_FEATURES = ["machine_type", "product_type", "material_type", "shift", "day_of_week"]
NUMERIC_FEATURES = ["quantity", "operator_experience_years", "setup_complexity",
                     "machine_age_years", "ambient_temperature_c"]


def predict_time_energy(row, shift):
    feats = row.copy()
    feats["shift"] = shift
    X_time = pd.DataFrame([feats])[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    time_min = float(production_model.predict(X_time)[0])

    feats_e = feats.copy()
    feats_e["production_time_min"] = time_min
    X_energy = pd.DataFrame([feats_e])[CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["production_time_min"]]
    energy_kwh = float(energy_model.predict(X_energy)[0])
    return time_min, energy_kwh


# ---- Build one day's batch of jobs ----------------------------------------
df = pd.read_csv(DATA_DIR / "production_data.csv")
# Keep only jobs that could plausibly fit in a single shift (batch sizes
# representative of one shift's worth of work, not multi-day mega-batches)
df = df[df["quantity"] <= 130]
batch = df.sample(60, random_state=11).reset_index(drop=True)

jobs = []
for _, row in batch.iterrows():
    base = row[CATEGORICAL_FEATURES + NUMERIC_FEATURES].to_dict()
    scenarios = {}
    for shift in SHIFTS:
        t, e = predict_time_energy(base, shift)
        scenarios[shift] = {"time_min": t, "energy_kwh": e, "cost_inr": e * TARIFF[shift]}
    jobs.append({"job_id": row["job_id"], "station": row["machine_type"], "scenarios": scenarios})

# ---- Baseline 1: naive (original shift in data, no optimization) ----------
naive_cost = 0.0
for j, row in zip(jobs, batch.itertuples()):
    naive_cost += j["scenarios"][row.shift]["cost_inr"]

# ---- Baseline 2: per-job greedy optimum (IGNORES capacity) -----------------
greedy_cost = 0.0
greedy_station_load = {}  # (station, shift) -> minutes used
for j in jobs:
    best_shift = min(j["scenarios"], key=lambda s: j["scenarios"][s]["cost_inr"])
    greedy_cost += j["scenarios"][best_shift]["cost_inr"]
    key = (j["station"], best_shift)
    greedy_station_load[key] = greedy_station_load.get(key, 0) + j["scenarios"][best_shift]["time_min"]

greedy_overloads = {k: v for k, v in greedy_station_load.items() if v > STATION_CAPACITY[k[0]]}

# ---- Real solve: constrained MILP ------------------------------------------
prob = pulp.LpProblem("ShiftScheduling", pulp.LpMinimize)
x = {(j["job_id"], s): pulp.LpVariable(f"x_{j['job_id']}_{s}", cat="Binary") for j in jobs for s in SHIFTS}

# Objective: minimize total cost
prob += pulp.lpSum(x[j["job_id"], s] * j["scenarios"][s]["cost_inr"] for j in jobs for s in SHIFTS)

# Constraint 1: each job assigned to exactly one shift
for j in jobs:
    prob += pulp.lpSum(x[j["job_id"], s] for s in SHIFTS) == 1

# Constraint 2: station capacity per shift can't be exceeded
stations = set(j["station"] for j in jobs)
for station in stations:
    for s in SHIFTS:
        station_jobs = [j for j in jobs if j["station"] == station]
        prob += pulp.lpSum(
            x[j["job_id"], s] * j["scenarios"][s]["time_min"] for j in station_jobs
        ) <= STATION_CAPACITY[station]

prob.solve(pulp.PULP_CBC_CMD(msg=0))

constrained_cost = pulp.value(prob.objective)
assignment = {}
station_load = {}
for j in jobs:
    for s in SHIFTS:
        if x[j["job_id"], s].value() == 1:
            assignment[j["job_id"]] = s
            key = (j["station"], s)
            station_load[key] = station_load.get(key, 0) + j["scenarios"][s]["time_min"]

results = {
    "status": pulp.LpStatus[prob.status],
    "jobs_in_batch": len(jobs),
    "naive_cost_inr": round(naive_cost, 2),
    "greedy_unconstrained_cost_inr": round(greedy_cost, 2),
    "greedy_capacity_violations": len(greedy_overloads),
    "greedy_overload_detail": {f"{k[0]}/{k[1]}": round(v - STATION_CAPACITY[k[0]], 1) for k, v in greedy_overloads.items()},
    "constrained_optimal_cost_inr": round(constrained_cost, 2),
    "savings_vs_naive_inr": round(naive_cost - constrained_cost, 2),
    "savings_vs_naive_percent": round((naive_cost - constrained_cost) / naive_cost * 100, 2),
    "gap_vs_unconstrained_ideal_inr": round(constrained_cost - greedy_cost, 2),
    "station_utilization": {
        f"{station}/{s}": {
            "minutes_used": round(station_load.get((station, s), 0), 1),
            "capacity_minutes": STATION_CAPACITY[station],
            "utilization_percent": round(station_load.get((station, s), 0) / STATION_CAPACITY[station] * 100, 1),
        }
        for station in stations for s in SHIFTS
    },
}

print(json.dumps(results, indent=2))

with open(DATA_DIR / "multi_job_optimization_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save the actual assignment too
assignment_rows = [{"job_id": jid, "assigned_shift": s} for jid, s in assignment.items()]
pd.DataFrame(assignment_rows).to_csv(DATA_DIR / "multi_job_assignment.csv", index=False)
