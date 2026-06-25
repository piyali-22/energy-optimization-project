# Energy Optimization Dashboard (Frontend)

Real React app that calls the FastAPI backend's live endpoints — no
formulas, no mock data. Every number on screen comes from an actual
HTTP request to the trained models.

## Run locally

1. Make sure the backend is running first (see `../backend/README.md`):
   ```bash
   cd ../backend
   uvicorn main:app --reload --port 8000
   ```
2. In a new terminal, run the frontend:
   ```bash
   npm install
   npm run dev
   ```
3. Open http://localhost:3000

## How it talks to the backend

Set in `src/App.jsx`:
```js
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
```
To point at a deployed backend instead of localhost, create a `.env` file:
```
VITE_API_BASE=https://your-backend.onrender.com
```
