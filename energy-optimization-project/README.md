# Energy Consumption Optimization Through Predicted Production Time

An ML pipeline that predicts how long a manufacturing job will take, predicts
the energy it will consume, and recommends the cheapest shift to run it on —
cutting energy costs without changing production capacity.

## Problem

Factories schedule jobs without considering time-of-day electricity tariffs.
If we can accurately *predict* a job's production time, we can predict its
energy footprint and shift it into a cheaper tariff window — saving cost with
zero change to what's actually produced.

## Pipeline

```
Job details (machine, product, quantity, material, shift, etc.)
        │
        ▼
[Model 1] Production Time Predictor (RandomForestRegressor)
        │  → predicted production_time_min
        ▼
[Model 2] Energy Consumption Predictor (RandomForestRegressor)
        │  → predicted energy_kwh
        ▼
[Optimizer] Simulates Morning / Evening / Night shifts,
            applies time-of-day tariff, picks the cheapest option
        │
        ▼
Recommended shift + cost savings vs. original schedule
```

## Dataset

Synthetic dataset of 4,000 production jobs (`data/production_data.csv`),
generated with domain-realistic formulas — not random noise. Features:
machine type, product type, material, quantity, operator experience,
setup complexity, machine age, ambient temperature, shift, day of week.
Targets: production time (minutes), energy consumed (kWh).

*(Synthetic data is standard practice for a project like this when real
factory logs aren't available — the pipeline works identically on real data.)*

## Results

**Production Time Model**
| Metric | Value |
|---|---|
| R² | 0.979 |
| MAE | 36.1 minutes |
| MAPE | 7.7% |

**Energy Consumption Model**
| Metric | Value |
|---|---|
| R² | 0.996 |
| MAE | 2.67 kWh |

**Schedule Optimization** (simulated on 500 jobs)
| Metric | Value |
|---|---|
| Jobs with cost reduction | 414 / 500 (82.8%) |
| Total cost reduction | **29.4%** |
| Total savings | ₹82,225 (on ₹2,79,689 baseline) |

## Files

- `generate_dataset.py` — synthetic data generator
- `train_model.py` — production time model + evaluation plots
- `train_energy_model.py` — energy consumption model
- `optimize_schedule.py` — shift optimization engine
- `data/production_data.csv` — dataset
- `data/optimization_results.csv` — per-job optimization recommendations
- `models/` — trained model files (.pkl) + metrics
- `plots/` — actual vs predicted, feature importance

## Tech Stack

Python, pandas, scikit-learn (RandomForestRegressor), matplotlib

## Project Structure

```
energy-optimization-project/
├── generate_dataset.py        # synthetic data generator
├── train_model.py             # production time model + plots
├── train_energy_model.py      # energy consumption model
├── optimize_schedule.py       # batch optimization script
├── data/                      # dataset + optimization results
├── models/                    # trained .pkl files + metrics
├── plots/                     # evaluation plots
├── backend/                   # FastAPI app serving the real models
│   ├── main.py
│   ├── models/, data/         # copies the API loads at runtime
│   └── requirements.txt
└── frontend/                  # React app, calls the real backend
    ├── src/App.jsx
    └── package.json
```

## Running the full stack

```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — every number you see comes from a real
HTTP call to the real trained models, not a formula or mock data.

## Status

| Layer | Status |
|---|---|
| Dataset | Real synthetic data, domain-realistic formulas |
| Production time model | Trained, R²=0.979 |
| Energy model | Trained, R²=0.996 |
| Optimizer | Tested, 29.4% cost reduction on 500 jobs |
| Backend API | Built, tested, serving real predictions |
| Frontend | Built, calls the real backend (no mock data) |

## Next Steps (optional polish)

- Deploy backend to Render/Railway, frontend to Vercel — fully live demo
- PostgreSQL to log real job history over time instead of static CSVs
- Auth — not needed for this internal tool, skip it
