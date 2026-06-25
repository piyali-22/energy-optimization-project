import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList } from "recharts";
import { Zap, TrendingDown, Cpu, Activity, ArrowRight, CheckCircle2, RefreshCw, AlertTriangle } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// ---- Design tokens ----------------------------------------------------
const COLORS = {
  bg: "#F6F4EF", surface: "#FFFFFF", ink: "#16212B", inkSoft: "#5B6470",
  line: "#E7E3DA", amber: "#C9711F", amberSoft: "#F4E3D2", teal: "#1F7A6C", tealSoft: "#DCEEE9",
};

// Display-only labels mapped onto the model's real trained categories.
// The backend was trained on generic machine_type / product_type values —
// these labels just present them in plant-relevant language. The actual
// value sent to the API is always the real category the model knows.
const STATION_LABELS = {
  Welding: "Body Welding", Press: "Stamping", CNC: "Chassis Mount",
  Lathe: "Trim Assembly", Assembly: "Final Assembly",
};
const VEHICLE_LABELS = {
  A: "Tata Nexon", B: "Nexon EV", C: "Tata Punch", D: "Tata Tiago", E: "Tata Harrier",
};
const SHIFTS = ["Morning", "Evening", "Night"];
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MATERIALS = ["Steel", "Aluminum", "Plastic", "Composite"];

export default function App() {
  const [form, setForm] = useState({
    machine_type: "Welding", product_type: "B", material_type: "Steel",
    quantity: 220, operator_experience_years: 4, setup_complexity: 3,
    machine_age_years: 5, ambient_temperature_c: 31, shift: "Evening", day_of_week: "Mon",
  });
  const [metrics, setMetrics] = useState(null);
  const [summary, setSummary] = useState(null);
  const [log, setLog] = useState([]);
  const [co2, setCo2] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [rerunning, setRerunning] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/metrics`).then((r) => r.json()).then(setMetrics).catch(() => setError("offline"));
    fetch(`${API_BASE}/optimization-log?limit=8`).then((r) => r.json()).then((d) => {
      setSummary(d.summary);
      setLog(d.jobs);
    }).catch(() => setError("offline"));
    fetch(`${API_BASE}/co2-tradeoff`).then((r) => r.json()).then(setCo2).catch(() => {});
  }, []);

  function update(key, value) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  async function runOptimization() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/optimize-shift`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Can't reach the backend. Make sure it's running at " + API_BASE);
    } finally {
      setLoading(false);
    }
  }

  async function rerunBatch() {
    setRerunning(true);
    try {
      const res = await fetch(`${API_BASE}/run-batch-optimization?sample_size=200`, { method: "POST" });
      const data = await res.json();
      setSummary(data);
    } catch {
      setError("Can't reach the backend. Make sure it's running at " + API_BASE);
    } finally {
      setRerunning(false);
    }
  }

  const chartData = result
    ? SHIFTS.map((s) => ({ shift: s, cost: result.scenarios[s].cost_inr }))
    : [];

  return (
    <div style={{ background: COLORS.bg, color: COLORS.ink, minHeight: "100vh", fontFamily: "'Inter', sans-serif" }}>
      {/* Header */}
      <div style={{ background: COLORS.ink, padding: "28px 32px" }}>
        <div style={{ fontSize: 11, letterSpacing: "0.12em", color: "#9AA7B2", fontFamily: "'IBM Plex Mono', monospace" }}>
          TATA MOTORS PASSENGER VEHICLES · SANAND, AHMEDABAD
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 6, flexWrap: "wrap", gap: 10 }}>
          <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 28, fontWeight: 600, color: "#fff", margin: 0 }}>
            Energy Consumption Optimization
          </h1>
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            background: error ? "rgba(180,69,46,0.18)" : "rgba(31,122,108,0.18)",
            color: error ? "#E08F7A" : "#7FD9C4",
            padding: "5px 12px", borderRadius: 20, fontSize: 12, fontFamily: "'IBM Plex Mono', monospace",
          }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: error ? "#E08F7A" : "#7FD9C4", display: "inline-block" }} />
            {error ? "BACKEND OFFLINE" : "LIVE · CONNECTED TO REAL MODELS"}
          </div>
        </div>
        <div style={{ color: "#7C8794", fontSize: 13, marginTop: 4 }}>
          Predicting production time and energy use, then recommending the lowest-cost shift to run each job
        </div>
      </div>

      {error && (
        <div style={{ margin: "16px 32px 0", background: "#FBEAE5", border: "1px solid #E0B6A8", borderRadius: 8, padding: 12, display: "flex", gap: 8, alignItems: "center", color: "#8A3A24", fontSize: 13 }}>
          <AlertTriangle size={16} /> {error === "offline" ? `Can't reach the backend at ${API_BASE}. Run "uvicorn main:app --reload --port 8000" in the backend folder, then refresh.` : error}
        </div>
      )}

      {/* Stat bar */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, padding: "24px 32px 0" }}>
        <StatCard icon={TrendingDown} label="Simulated cost reduction" value={summary ? `${summary.percent_cost_reduction}%` : "—"} sub={summary ? `₹${summary.total_savings_inr.toLocaleString("en-IN")} saved / ${summary.jobs_simulated} jobs` : "loading…"} />
        <StatCard icon={CheckCircle2} label="Jobs optimized" value={summary ? `${summary.jobs_with_cost_reduction} / ${summary.jobs_simulated}` : "—"} sub={summary ? `${((summary.jobs_with_cost_reduction / summary.jobs_simulated) * 100).toFixed(1)}% found a cheaper shift` : "loading…"} />
        <StatCard icon={Cpu} label="Time model accuracy" value={metrics ? `R² ${metrics.production_time_model.R2_score}` : "—"} sub={metrics ? `MAPE ${metrics.production_time_model.MAPE_percent}%` : "loading…"} />
        <StatCard icon={Activity} label="Energy model accuracy" value={metrics ? `R² ${metrics.energy_model.R2_score}` : "—"} sub={metrics ? `MAE ${metrics.energy_model.MAE_kwh} kWh` : "loading…"} />
      </div>

      {/* Main grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.3fr", gap: 16, padding: "20px 32px" }}>
        {/* Form */}
        <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}`, borderRadius: 10, padding: 20 }}>
          <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: 15, marginBottom: 14 }}>
            Simulate a job
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Field label="Station">
              <select value={form.machine_type} onChange={(e) => update("machine_type", e.target.value)} style={selectStyle}>
                {Object.entries(STATION_LABELS).map(([k, label]) => <option key={k} value={k}>{label}</option>)}
              </select>
            </Field>
            <Field label="Vehicle model">
              <select value={form.product_type} onChange={(e) => update("product_type", e.target.value)} style={selectStyle}>
                {Object.entries(VEHICLE_LABELS).map(([k, label]) => <option key={k} value={k}>{label}</option>)}
              </select>
            </Field>
            <Field label="Material">
              <select value={form.material_type} onChange={(e) => update("material_type", e.target.value)} style={selectStyle}>
                {MATERIALS.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </Field>
            <Field label="Current shift">
              <select value={form.shift} onChange={(e) => update("shift", e.target.value)} style={selectStyle}>
                {SHIFTS.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Batch size (vehicles)">
              <input type="number" value={form.quantity} onChange={(e) => update("quantity", +e.target.value)} style={inputStyle} />
            </Field>
            <Field label="Day of week">
              <select value={form.day_of_week} onChange={(e) => update("day_of_week", e.target.value)} style={selectStyle}>
                {DAYS.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </Field>
            <Field label="Operator experience (yrs)">
              <input type="number" value={form.operator_experience_years} onChange={(e) => update("operator_experience_years", +e.target.value)} style={inputStyle} />
            </Field>
            <Field label="Setup complexity (1-5)">
              <input type="number" min={1} max={5} value={form.setup_complexity} onChange={(e) => update("setup_complexity", +e.target.value)} style={inputStyle} />
            </Field>
            <Field label="Machine age (yrs)">
              <input type="number" value={form.machine_age_years} onChange={(e) => update("machine_age_years", +e.target.value)} style={inputStyle} />
            </Field>
            <Field label="Ambient temp (°C)">
              <input type="number" value={form.ambient_temperature_c} onChange={(e) => update("ambient_temperature_c", +e.target.value)} style={inputStyle} />
            </Field>
          </div>

          <button onClick={runOptimization} disabled={loading} style={buttonStyle}>
            {loading ? "Calling the real model…" : "Run prediction + optimization"} <ArrowRight size={15} />
          </button>

          {result && (
            <div style={{ marginTop: 16, background: COLORS.amberSoft, borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 11, color: COLORS.inkSoft, fontFamily: "'IBM Plex Mono', monospace" }}>
                PREDICTED · {form.shift.toUpperCase()} SHIFT
              </div>
              <div style={{ display: "flex", gap: 24, marginTop: 6 }}>
                <Metric value={result.scenarios[form.shift].predicted_time_min} unit="min" label="production time" />
                <Metric value={result.scenarios[form.shift].predicted_energy_kwh} unit="kWh" label="energy" />
                <Metric value={result.scenarios[form.shift].cost_inr} unit="₹" label="cost" prefix />
              </div>
            </div>
          )}
        </div>

        {/* Optimization comparison */}
        <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}`, borderRadius: 10, padding: 20 }}>
          <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: 15, marginBottom: 4 }}>
            Shift-cost optimizer
          </div>
          <div style={{ fontSize: 12, color: COLORS.inkSoft, marginBottom: 14 }}>
            Same job, predicted live by the real model across all 3 shifts at time-of-day tariffs.
          </div>

          {!result ? (
            <div style={{ height: 220, display: "flex", alignItems: "center", justifyContent: "center", color: COLORS.inkSoft, fontSize: 13, border: `1px dashed ${COLORS.line}`, borderRadius: 8 }}>
              Run a prediction to see the optimizer compare shifts
            </div>
          ) : (
            <>
              <div style={{ height: 180 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.line} vertical={false} />
                    <XAxis dataKey="shift" tick={{ fontSize: 12, fill: COLORS.inkSoft }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: COLORS.inkSoft }} axisLine={false} tickLine={false} width={40} />
                    <Tooltip formatter={(v) => `₹${v.toFixed(0)}`} contentStyle={{ fontSize: 12, borderRadius: 8, border: `1px solid ${COLORS.line}` }} />
                    <Bar dataKey="cost" radius={[6, 6, 0, 0]}>
                      {chartData.map((r) => (
                        <Cell key={r.shift} fill={r.shift === result.recommended_shift ? COLORS.teal : COLORS.amber} opacity={r.shift === form.shift ? 1 : 0.55} />
                      ))}
                      <LabelList dataKey="cost" position="top" formatter={(v) => `₹${v.toFixed(0)}`} style={{ fontSize: 11, fill: COLORS.ink }} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={{ marginTop: 8, background: COLORS.tealSoft, borderRadius: 8, padding: 14, display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
                <div>
                  <div style={{ fontSize: 12, color: COLORS.teal, fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}>
                    <Zap size={14} /> RECOMMENDED: {result.recommended_shift.toUpperCase()} SHIFT
                  </div>
                  <div style={{ fontSize: 12, color: COLORS.inkSoft, marginTop: 2 }}>vs. running it on {form.shift}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 22, fontWeight: 700, color: COLORS.teal }}>
                    -₹{result.savings_inr.toFixed(0)}
                  </div>
                  <div style={{ fontSize: 11, color: COLORS.teal }}>{result.savings_percent.toFixed(1)}% cheaper</div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* CO2 vs cost trade-off */}
      {co2 && (
        <div style={{ padding: "0 32px 20px" }}>
          <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}`, borderRadius: 10, padding: 20 }}>
            <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: 15, marginBottom: 4 }}>
              Cost vs. sustainability trade-off
            </div>
            <div style={{ fontSize: 12, color: COLORS.inkSoft, marginBottom: 14 }}>
              Minimizing electricity cost and minimizing CO2 emissions are not the same goal, here's the real gap, computed on 500 jobs.
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 14 }}>
              {Object.entries(co2.strategy_comparison).map(([name, s]) => {
                const isCostOptimal = name.includes("cost_optimal") || name.includes("cost-optimal") || name.includes("deployed");
                const isEnergyOptimal = name.includes("energy_optimal") || name.includes("energy-optimal") || name.includes("alternative");
                const accent = isCostOptimal ? COLORS.amber : isEnergyOptimal ? COLORS.teal : COLORS.inkSoft;
                const label = name.includes("naive") ? "Naive (no optimization)"
                  : isCostOptimal ? "Cost-optimal (deployed)"
                  : "Energy-optimal (alternative)";
                return (
                  <div key={name} style={{ border: `1px solid ${COLORS.line}`, borderRadius: 8, padding: 12 }}>
                    <div style={{ fontSize: 11, color: accent, fontWeight: 600, marginBottom: 8 }}>{label}</div>
                    <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 18, fontWeight: 600 }}>
                      ₹{s.total_cost_inr.toLocaleString("en-IN")}
                    </div>
                    <div style={{ fontSize: 11, color: COLORS.inkSoft, marginBottom: 6 }}>total cost</div>
                    <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 18, fontWeight: 600 }}>
                      {s.total_co2_kg.toLocaleString("en-IN")} kg
                    </div>
                    <div style={{ fontSize: 11, color: COLORS.inkSoft }}>CO2 emitted</div>
                  </div>
                );
              })}
            </div>

            <div style={{ background: COLORS.amberSoft, borderRadius: 8, padding: 14, fontSize: 13, lineHeight: 1.5, color: COLORS.ink }}>
              <strong style={{ color: COLORS.amber }}>Key finding: </strong>{co2.key_finding}
            </div>
          </div>
        </div>
      )}

      {/* Optimization log */}
      <div style={{ padding: "0 32px 28px" }}>
        <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}`, borderRadius: 10, padding: 20 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
            <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: 15 }}>
              Optimization log — real jobs re-scheduled by the model
            </div>
            <button onClick={rerunBatch} disabled={rerunning} style={rerunButtonStyle}>
              <RefreshCw size={13} className={rerunning ? "spin" : ""} />
              {rerunning ? "Re-running on backend…" : "Re-run batch optimization"}
            </button>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ color: COLORS.inkSoft, textAlign: "left", fontSize: 11, fontFamily: "'IBM Plex Mono', monospace" }}>
                <th style={thStyle}>JOB</th>
                <th style={thStyle}>STATION</th>
                <th style={thStyle}>SHIFT CHANGE</th>
                <th style={thStyle}>ORIGINAL COST</th>
                <th style={thStyle}>NEW COST</th>
                <th style={thStyle}>SAVINGS</th>
              </tr>
            </thead>
            <tbody>
              {log.map((row) => (
                <tr key={row.job_id} style={{ borderTop: `1px solid ${COLORS.line}` }}>
                  <td style={{ ...tdStyle, fontFamily: "'IBM Plex Mono', monospace" }}>{row.job_id}</td>
                  <td style={tdStyle}>{STATION_LABELS[row.machine_type] || row.machine_type}</td>
                  <td style={tdStyle}>
                    <span style={{ color: COLORS.inkSoft }}>{row.original_shift}</span>{" "}
                    <ArrowRight size={11} style={{ display: "inline", verticalAlign: "middle" }} />{" "}
                    <span style={{ color: COLORS.teal, fontWeight: 600 }}>{row.recommended_shift}</span>
                  </td>
                  <td style={{ ...tdStyle, fontFamily: "'IBM Plex Mono', monospace" }}>₹{row.original_cost_inr.toFixed(2)}</td>
                  <td style={{ ...tdStyle, fontFamily: "'IBM Plex Mono', monospace" }}>₹{row.recommended_cost_inr.toFixed(2)}</td>
                  <td style={{ ...tdStyle, fontFamily: "'IBM Plex Mono', monospace", color: COLORS.teal, fontWeight: 600 }}>-₹{row.savings_inr.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, sub }) {
  return (
    <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}`, borderRadius: 10, padding: "16px 18px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, color: COLORS.inkSoft, fontSize: 12 }}>
        <Icon size={14} /> {label}
      </div>
      <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 24, fontWeight: 600, marginTop: 6 }}>{value}</div>
      <div style={{ fontSize: 11, color: COLORS.inkSoft, marginTop: 2 }}>{sub}</div>
    </div>
  );
}

function Metric({ value, unit, label, prefix }) {
  return (
    <div>
      <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 22, fontWeight: 600 }}>
        {prefix ? unit : ""}{typeof value === "number" ? value.toFixed(value < 10 ? 1 : 0) : value} {!prefix && <span style={{ fontSize: 12, fontWeight: 400 }}>{unit}</span>}
      </div>
      <div style={{ fontSize: 11, color: COLORS.inkSoft }}>{label}</div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label style={{ display: "block" }}>
      <div style={{ fontSize: 11, color: COLORS.inkSoft, marginBottom: 4 }}>{label}</div>
      {children}
    </label>
  );
}

const selectStyle = {
  width: "100%", padding: "8px 10px", borderRadius: 6, border: `1px solid ${COLORS.line}`,
  fontSize: 13, background: "#fff", color: COLORS.ink, fontFamily: "'Inter', sans-serif",
};
const inputStyle = { ...selectStyle };
const thStyle = { padding: "6px 8px", fontWeight: 500 };
const tdStyle = { padding: "10px 8px" };
const buttonStyle = {
  marginTop: 16, width: "100%", background: COLORS.ink, color: "#fff", border: "none",
  borderRadius: 8, padding: "12px 0", fontSize: 14, fontWeight: 600, cursor: "pointer",
  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
  fontFamily: "'Space Grotesk', sans-serif", opacity: 1,
};
const rerunButtonStyle = {
  display: "flex", alignItems: "center", gap: 6, background: COLORS.bg, border: `1px solid ${COLORS.line}`,
  borderRadius: 6, padding: "6px 12px", fontSize: 12, cursor: "pointer", color: COLORS.ink,
};
