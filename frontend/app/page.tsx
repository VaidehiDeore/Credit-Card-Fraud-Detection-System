"use client";

import { useState } from "react";
import axios from "axios";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
} from "chart.js";
import { Pie, Bar, Line } from "react-chartjs-2";

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement
);

export default function Home() {
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const sampleData = [
    {
      amount: 25000,
      merchant_cat: "electronics",
      merchant_id_hash: "M_1234",
      card_id_hash: "C_56789",
      city: "Unknown",
      country: "Foreign",
      device_type: "unknown",
      channel: "online",
      hour: 2,
      dayofweek: 6,
      prev_24h_tx_count_card: 8,
      prev_24h_amt_card: 50000,
      prev_1h_tx_count_card: 5,
      velocity_amt_1h: 25000,
      is_international: true,
      is_night: true,
    },
    {
      amount: 850,
      merchant_cat: "grocery",
      merchant_id_hash: "M_2222",
      card_id_hash: "C_33333",
      city: "Pune",
      country: "India",
      device_type: "mobile",
      channel: "offline",
      hour: 14,
      dayofweek: 2,
      prev_24h_tx_count_card: 2,
      prev_24h_amt_card: 1500,
      prev_1h_tx_count_card: 0,
      velocity_amt_1h: 0,
      is_international: false,
      is_night: false,
    },
  ];

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/score", sampleData);
      setResults(res.data.results);
    } catch {
      alert("Error connecting to backend");
    }
    setLoading(false);
  };

  const fraud = results.filter((r) => r.prediction === "Fraud").length;
  const nonFraud = results.filter((r) => r.prediction === "Non-Fraud").length;
  const high = results.filter((r) => r.risk_level === "HIGH").length;
  const medium = results.filter((r) => r.risk_level === "MEDIUM").length;
  const low = results.filter((r) => r.risk_level === "LOW").length;

  const lineData = {
    labels: ["Jan", "Mar", "May", "Jul", "Sep", "Nov"],
    datasets: [
      {
        label: "Fraud Risk Trend",
        data: [18, 32, 28, 61, 42, 76],
        borderColor: "#8b5cf6",
        backgroundColor: "rgba(139, 92, 246, 0.25)",
        tension: 0.45,
      },
    ],
  };

  const pieData = {
    labels: ["Fraud", "Non-Fraud"],
    datasets: [
      {
        data: [fraud, nonFraud],
        backgroundColor: ["#ef4444", "#22c55e"],
        borderColor: "#0f172a",
      },
    ],
  };

  const barData = {
    labels: ["High", "Medium", "Low"],
    datasets: [
      {
        label: "Risk Count",
        data: [high, medium, low],
        backgroundColor: ["#ef4444", "#f59e0b", "#22c55e"],
      },
    ],
  };

  return (
    <main className="dashboard">
      <section className="topbar fade-in">
        <div>
          <div className="title">💳 Fraud Detection Ops</div>
          <p className="subtitle">
            AI-powered credit card fraud monitoring dashboard
          </p>
        </div>

        <button className="run-btn" onClick={fetchData}>
          {loading ? "Analyzing..." : "Run Fraud Detection"}
        </button>
      </section>

      <section className="container">
        <div className="grid-main fade-in">
          <div className="stack">
            <div className="glass-card">
              <p className="metric-label">High Risk</p>
              <div className="metric-value red">{high}</div>
            </div>

            <div className="glass-card">
              <p className="metric-label">Medium Risk</p>
              <div className="metric-value yellow">{medium}</div>
            </div>

            <div className="glass-card">
              <p className="metric-label">Low Risk</p>
              <div className="metric-value green">{low}</div>
            </div>
          </div>

          <div className="glass-card virtual-card">
            <p className="metric-label">Virtual Fraud Shield Card</p>
            <h2>Credit Shield AI</h2>

            <div className="card-chip"></div>

            <div className="card-number">**** 8821</div>
            <p className="metric-label">ML Fraud Monitoring Active</p>
            <span className="status-pill">Virtual card active</span>
          </div>

          <div className="glass-card feed">
            <h2>Activity Feed</h2>
            <p>🔴 High risk transaction detected</p>
            <p>🟢 Safe transaction approved</p>
            <p>🟣 ML threshold applied</p>
            <p>⚡ FastAPI scoring active</p>
          </div>
        </div>

        <div className="grid-finance fade-in">
          <div className="glass-card">
            <p className="metric-label">Balance</p>
            <div className="metric-value blue">$5,312.45</div>
          </div>

          <div className="glass-card">
            <p className="metric-label">Available Credit</p>
            <div className="metric-value green">$13,500</div>
          </div>

          <div className="glass-card">
            <p className="metric-label">Utilization</p>
            <div className="metric-value purple">15.7%</div>
          </div>
        </div>

        <div className="grid-charts fade-in">
          <div className="glass-card">
            <h2>Fraud Risk Trend</h2>
            <Line data={lineData} />
          </div>

          <div className="glass-card">
            <h2>Fraud Distribution</h2>
            <Pie data={pieData} />
          </div>
        </div>

        <div className="grid-bottom fade-in">
          <div className="glass-card">
            <h2>Risk Count</h2>
            <Bar data={barData} />
          </div>

          <div className="glass-card">
            <h2>Transactions</h2>

            <table className="table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Probability</th>
                  <th>Prediction</th>
                  <th>Decision</th>
                  <th>Risk</th>
                </tr>
              </thead>

              <tbody>
                {results.map((r, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{r.fraud_probability_percent}%</td>
                    <td>{r.prediction}</td>
                    <td>{r.decision}</td>
                    <td>
                      <span
                        className={
                          r.risk_level === "HIGH"
                            ? "badge badge-red"
                            : r.risk_level === "MEDIUM"
                            ? "badge badge-yellow"
                            : "badge badge-green"
                        }
                      >
                        {r.risk_level}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {results.length === 0 && (
              <p className="empty">
                Click “Run Fraud Detection” to analyze transactions.
              </p>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}