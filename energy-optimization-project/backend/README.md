# Energy Optimization API (Backend)

Real FastAPI server that loads the actual trained models
(`production_time_model.pkl`, `energy_model.pkl`) and serves live
predictions + a shift-cost optimizer.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000/docs for interactive Swagger UI — great for
showing your supervisor the live, real API directly.

## Endpoints

- `GET /metrics` — real R²/MAE for both trained models
- `POST /predict-time` — predicts production time for a job
- `POST /predict-energy` — predicts time + energy for a job
- `POST /optimize-shift` — simulates a job across all 3 shifts, recommends cheapest
- `GET /optimization-log` — pre-computed results from the 500-job batch run
- `POST /run-batch-optimization?sample_size=N` — live re-run of the optimizer
  on a fresh random sample, using the real loaded models

## Deploy (free options)

- Render.com (Web Service, free tier) — point it at this folder, build
  command `pip install -r requirements.txt`, start command
  `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Railway.app — similar, auto-detects Python + uvicorn
