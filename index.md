---
layout: default
title: Home
---

# ⚡ Quantum Felix — Scenario & Strategy Simulation Engine

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg?logo=python)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Research--Preview-orange)]()
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](./LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](./CONTRIBUTING.md)

**Quantum Felix** is an **experimental simulation and orchestration engine** designed for **multi-scenario testing, large-scale backtesting, and strategy optimization**. It is currently in a **Research Preview** stage — functional, but still being structured and systematized.

<div class="hero-section">
  <div class="hero-content">
    <h2>🚀 Get Started</h2>
    <p>Explore the power of quantum-inspired simulation for your strategy optimization needs.</p>
    <div class="cta-buttons">
      <a href="{{ '/documentation/' | relative_url }}" class="btn btn-primary">📖 Documentation</a>
      <a href="{{ '/features/' | relative_url }}" class="btn btn-secondary">✨ Features</a>
      <a href="https://github.com/{{ site.repository }}" class="btn btn-github">💻 View on GitHub</a>
    </div>
  </div>
</div>

## 🧩 What Makes Quantum Felix Special?

<div class="features-grid">
  <div class="feature-card">
    <h3>🔁 Scenario Factory</h3>
    <p>Generate realistic and synthetic trajectories with regime switching, shocks, and drift injection for comprehensive testing.</p>
  </div>
  
  <div class="feature-card">
    <h3>🧪 Advanced Backtesting</h3>
    <p>Perform walk-forward analysis, multi-seed Monte Carlo sweeps, and τ-sweep regime exploration.</p>
  </div>
  
  <div class="feature-card">
    <h3>🐈 Quantum Early Stopping</h3>
    <p>Revolutionary probabilistic approach inspired by Schrödinger's cat for robust training management.</p>
  </div>
  
  <div class="feature-card">
    <h3>🧭 Strategy Orchestration</h3>
    <p>Plug-in Strategy API supporting rule-based, ML, or hybrid policies for maximum flexibility.</p>
  </div>
</div>

## 🌍 Use Cases

**Quantum Felix** is designed for diverse applications where strategies must be validated under uncertainty:

- **📈 Finance & Trading** — Backtest strategies with realistic cost models and stress tests
- **⚡ Energy & IoT** — Demand/load simulations with drift and anomaly injection  
- **🏭 Industrial Control** — Predictive maintenance with multi-scenario simulations
- **🤖 Robotics** — What-if testing of policies under uncertainty and latency constraints
- **🛡️ Cybersecurity** — Anomaly simulation and robust response evaluation

## 🐱 The Quantum Approach

Our unique **Quantum Early Stopping** mechanism uses probability amplitudes inspired by quantum mechanics:

- `psi_alive` and `psi_dead` represent the system's state superposition
- Dynamic fidelity measurements guide training decisions
- Probabilistic decision-making prevents premature optimization halt

This innovative approach provides **softer, more intelligent stopping criteria** compared to traditional deterministic thresholds.

## ⚙️ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simple scenario backtest (WIP)
python scripts/run_backtest.py --config configs/example.yaml
```

## 🚧 Current Status

- ✅ Core concepts implemented (scenarios, sweeps, cost models)
- 🛠️ Modularization in progress (Strategy API, reporting)  
- 🔮 Planned integrations (AstroMind-4D bridge, HTML reports)

---

<div class="footer-cta">
  <h2>Ready to Explore?</h2>
  <p>Dive into the documentation or check out our examples to get started with Quantum Felix.</p>
  <a href="{{ '/documentation/' | relative_url }}" class="btn btn-primary">Get Started Now</a>
</div>

<style>
.hero-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 3rem 2rem;
  margin: 2rem -2rem;
  border-radius: 10px;
  text-align: center;
}

.hero-content h2 {
  margin-bottom: 1rem;
  font-size: 2.5rem;
}

.cta-buttons {
  margin-top: 2rem;
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.btn {
  padding: 12px 24px;
  border-radius: 6px;
  text-decoration: none;
  font-weight: 600;
  transition: all 0.3s ease;
  display: inline-block;
}

.btn-primary {
  background-color: #28a745;
  color: white;
}

.btn-secondary {
  background-color: #6c757d;
  color: white;
}

.btn-github {
  background-color: #333;
  color: white;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
}

.feature-card {
  background: #f8f9fa;
  padding: 2rem;
  border-radius: 10px;
  border-left: 4px solid #667eea;
  transition: transform 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.feature-card h3 {
  margin-bottom: 1rem;
  color: #333;
}

.footer-cta {
  background-color: #f8f9fa;
  padding: 3rem 2rem;
  margin: 3rem -2rem 0;
  text-align: center;
  border-radius: 10px;
  border-top: 4px solid #667eea;
}

@media (max-width: 768px) {
  .cta-buttons {
    flex-direction: column;
    align-items: center;
  }
  
  .hero-content h2 {
    font-size: 2rem;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
}
</style>