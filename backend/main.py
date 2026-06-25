"""
main.py — Energy Consumption Optimization API
------------------------------------------------
Loads the ACTUAL trained models (production_time_model.pkl, energy_model.pkl)
and serves real predictions + the shift-cost optimizer over HTTP.

Run locally:
    pip install fastapi uvicorn[standard] pydantic joblib pandas scikit-learn --break-system-packages
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs.
"""

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="Energy Consumption Optimization API",
    description="Predicts production time & energy for Tata Motors Sanand plant jobs, "
                 "and recommends the cheapest shift to run them on.",
    version="1.0.0",
)

# Allow the frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load real trained models on startup --------------------------------
production_model = joblib.load(MODELS_DIR / "production_time_model.pkl")
energy_model = joblib.load(MODELS_DIR / "energy_model.pkl")

with open(MODELS_DIR / "metrics.json") as f:
    TIME_METRICS = json.load(f)
with open(MODELS_DIR / "energy_metrics.json") as f:
    ENERGY_METRICS = json.load(f)

OPT_LOG_DF = pd.read_csv(DATA_DIR / "optimization_results.csv")
with open(DATA_DIR / "optimization_summary.json") as f:
    OPT_SUMMARY = json.load(f)

TARIFF = {"Morning": 7.5, "Evening": 9.5, "Night": 5.5}
CATEGORICAL_FEATURES = ["machine_type", "product_type", "material_type", "shift", "day_of_week"]
NUMERIC_FEATURES = ["quantity", "operator_experience_years", "setup_complexity",
                     "machine_age_years", "ambient_temperature_c"]


# ---- Request/response schemas --------------------------------------------
class JobInput(BaseModel):
    machine_type: Literal["CNC", "Lathe", "Press", "Welding", "Assembly"]
    product_type: Literal["A", "B", "C", "D", "E"]
    material_type: Literal["Steel", "Aluminum", "Plastic", "Composite"]
    quantity: int = Field(ge=1, le=2000)
    operator_experience_years: int = Field(ge=0, le=40)
    setup_complexity: int = Field(ge=1, le=5)
    machine_age_years: int = Field(ge=0, le=30)
    ambient_temperature_c: float
    shift: Literal["Morning", "Evening", "Night"]
    day_of_week: Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _predict_time(job: JobInput) -> float:
    X = pd.DataFrame([job.model_dump()])[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    return float(production_model.predict(X)[0])


def _predict_energy(job: JobInput, production_time_min: float) -> float:
    row = job.model_dump()
    row["production_time_min"] = production_time_min
    X = pd.DataFrame([row])[CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["production_time_min"]]
    return float(energy_model.predict(X)[0])


# ---- Endpoints -------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "energy-consumption-optimization-api"}


@app.get("/metrics")
def get_metrics():
    """Real evaluation metrics for both trained models."""
    return {"production_time_model": TIME_METRICS, "energy_model": ENERGY_METRICS}


@app.post("/predict-time")
def predict_time(job: JobInput):
    return {"predicted_production_time_min": round(_predict_time(job), 1)}


@app.post("/predict-energy")
def predict_energy(job: JobInput):
    time_min = _predict_time(job)
    energy = _predict_energy(job, time_min)
    return {
        "predicted_production_time_min": round(time_min, 1),
        "predicted_energy_kwh": round(energy, 2),
    }


@app.post("/optimize-shift")
def optimize_shift(job: JobInput):
    """Simulates the job across all 3 shifts and recommends the cheapest one."""
    scenarios = {}
    for shift in TARIFF:
        scenario_job = job.model_copy(update={"shift": shift})
        time_min = _predict_time(scenario_job)
        energy = _predict_energy(scenario_job, time_min)
        cost = energy * TARIFF[shift]
        scenarios[shift] = {
            "predicted_time_min": round(time_min, 1),
            "predicted_energy_kwh": round(energy, 2),
            "cost_inr": round(cost, 2),
        }

    best_shift = min(scenarios, key=lambda s: scenarios[s]["cost_inr"])
    current_cost = scenarios[job.shift]["cost_inr"]
    best_cost = scenarios[best_shift]["cost_inr"]

    return {
        "original_shift": job.shift,
        "recommended_shift": best_shift,
        "scenarios": scenarios,
        "savings_inr": round(current_cost - best_cost, 2),
        "savings_percent": round((current_cost - best_cost) / current_cost * 100, 2) if current_cost > 0 else 0,
    }


@app.get("/optimization-log")
def optimization_log(limit: int = 20, only_savings: bool = True):
    """Returns real, pre-computed optimization results from the 500-job batch run."""
    df = OPT_LOG_DF
    if only_savings:
        df = df[df["savings_inr"] > 0]
    return {
        "summary": OPT_SUMMARY,
        "jobs": df.head(limit).to_dict(orient="records"),
    }


@app.get("/co2-tradeoff")
def co2_tradeoff():
    """
    Returns the pre-computed cost-vs-emissions trade-off analysis:
    naive vs cost-optimal vs energy/CO2-optimal scheduling strategies.
    """
    path = DATA_DIR / "co2_tradeoff.json"
    if not path.exists():
        raise HTTPException(404, "co2_tradeoff.json not found — run co2_tradeoff.py first")
    with open(path) as f:
        return json.load(f)


@app.post("/run-batch-optimization")
def run_batch_optimization(sample_size: int = 100):
    """
    Live re-run: samples real rows from the dataset and re-runs the optimizer
    fresh, using the actual loaded models (not the pre-computed CSV).
    """
    if sample_size > 1000:
        raise HTTPException(400, "sample_size too large, max 1000")

    full_df = pd.read_csv(DATA_DIR / "production_data.csv").sample(sample_size, random_state=None)

    total_original, total_optimized, improved = 0.0, 0.0, 0
    for _, row in full_df.iterrows():
        job = JobInput(**{k: row[k] for k in CATEGORICAL_FEATURES + NUMERIC_FEATURES})
        result = optimize_shift(job)
        original_cost = result["scenarios"][job.shift]["cost_inr"]
        best_cost = result["scenarios"][result["recommended_shift"]]["cost_inr"]
        total_original += original_cost
        total_optimized += best_cost
        if best_cost < original_cost:
            improved += 1

    savings = total_original - total_optimized
    pct = (savings / total_original * 100) if total_original > 0 else 0
    return {
        "jobs_simulated": sample_size,
        "jobs_with_cost_reduction": improved,
        "total_original_cost_inr": round(total_original, 2),
        "total_optimized_cost_inr": round(total_optimized, 2),
        "total_savings_inr": round(savings, 2),
        "percent_cost_reduction": round(pct, 2),
    }
