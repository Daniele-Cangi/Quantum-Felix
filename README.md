
# Quantum Felix — Quantum-Inspired Simulation & Strategy Engine

**Quantum Felix** is a research prototype that introduces a *quantum-inspired early stopping* mechanism for simulations, backtesting, and stress-testing under uncertainty.  
Although entirely classical (Python, NumPy, SciPy), it borrows concepts from **quantum mechanics** to manage uncertainty in model training and evaluation.

---

## ⚙️ Technology Core

Traditional early stopping monitors a validation metric and stops once it flattens.  
This is simple but **too deterministic**: noise or temporary stagnation often trigger *premature stopping*.

Quantum Felix replaces the binary rule with a **probabilistic model**:

- `psi_alive` → amplitude that training should continue  
- `psi_dead` → amplitude that training should stop  
- **Update rules**:
  - `psi_alive` increases if *fidelity* (consistency with the best past result) and *improvement* are high  
  - `psi_dead` increases otherwise  
- Amplitudes are **normalized** so that |psi_alive|² + |psi_dead|² = 1  
- At each epoch the system performs a **collapse**:
  - If `psi_dead` dominates → STOP  
  - Else → CONTINUE  

This approach introduces **stochastic buffering**: the model can explore deeper even if short-term signals are ambiguous.

---

## 🧩 Architecture (Conceptual)

The core workflow of **Quantum Felix** can be seen as a pipeline:

```
[Data Loader] → [Scenario Engine] → [Runtime Executor] → [Evaluator & Metrics] → [Optimizer] → [Reporter]
```

- **Data Loader** → imports synthetic or realistic time series  
- **Scenario Engine** → injects shocks, drift, regime switching  
- **Runtime Executor** → runs strategies (rule-based, ML, or hybrid)  
- **Evaluator & Metrics** → computes fidelity, improvement, Sharpe ratios, drawdowns  
- **Optimizer** → sweeps hyperparameters and seeds  
- **Reporter** → exports results (CSV, JSON, HTML)  

---

## 🔬 Implementation

- **Language**: Python 3.11+  
- **Core libraries**: NumPy, SciPy, scikit-learn, Optuna  
- **Scenario engine**: synthetic + realistic datasets, multi-seed sweeps  
- **Cost models**: fees, slippage, latency (useful in finance but generalizable)  
- **Metrics**: fidelity, improvement, PnL-like returns, Sharpe ratios, drawdowns  
- **CLI**: configuration via YAML/JSON, reproducible experiments  

---

## 🚀 Why It’s Different

- **Deterministic vs Probabilistic**  
  - Standard: stop at first plateau.  
  - Felix: maintain a superposition until evidence is decisive.  

- **Noise Robustness**  
  - Small fluctuations don’t trigger false stops.  

- **Generalizable**  
  - While inspired by trading research, the same logic applies to IoT, robotics, energy, anomaly detection.  

---

## Schrödinger’s Cat Analogy

The inspiration comes from Schrödinger’s famous thought experiment:  
- A cat inside a box is **both alive and dead** until observed.  
- In Quantum Felix, training is likewise in a **superposed state**: both *continue* and *stop* remain possible until evidence makes one prevail.  

👉 This analogy illustrates why Felix delays premature stopping while still ensuring convergence when real plateaus emerge.

---

## 🌍 Example Use Cases

- **Finance** — avoid discarding potentially profitable strategies due to noisy plateaus.  
- **Energy & IoT** — resilient to signal drift in demand/load forecasts.  
- **Industrial Control** — predictive maintenance with probabilistic stopping instead of rigid thresholds.  
- **Robotics** — policies continue learning under uncertain feedback.  
- **Cybersecurity** — simulate attacks and test response without premature halts.  

---

## 📂 Repository Structure

```
quantum-felix/
├─ src/felix/
│  ├─ quantum_cat.py       # main quantum-inspired early stopping engine
│  ├─ __main__.py          # entry point
│  └─ __init__.py
├─ scripts/
│  └─ run_quantum_cat.py   # CLI wrapper
├─ README.md
├─ requirements.txt
└─ LICENSE
```

---

## 🚀 Quick Start

```bash
# Install in editable mode
py -3.13 -m pip install -e .

# Run help
quantum-cat --help

# Run with test config
quantum-cat --complex-test --realistic
```

---

## 📜 License
Apache License 2.0
