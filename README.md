
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

🔬 *Technical note*: Fidelity is computed as a correlation-like measure against the best historical state, while improvement is tracked as relative delta on chosen metrics. Normalization ensures numerical stability and prevents drift in long runs.

This approach introduces **stochastic buffering**: the model can explore deeper even if short-term signals are ambiguous.

---

## 🧩 Architecture (Conceptual)

The **Quantum Felix pipeline** can be summarized as:

```
+-------------+     +-----------------+     +------------------+
| Data Loader | --> | Scenario Engine | --> | Runtime Executor |
+-------------+     +-----------------+     +------------------+
                                                 |
                                                 v
                                          +---------------------+
                                          | Evaluator & Metrics |
                                          +---------------------+
                                                 |
                                                 v
                                          +-----------+
                                          | Optimizer |
                                          +-----------+
                                                 |
                                                 v
                                          +----------+
                                          | Reporter |
                                          +----------+
```

- **Data Loader** → imports synthetic or realistic time series, ensuring consistent formatting and handling of edge cases (missing data, irregular intervals).  
- **Scenario Engine** → injects shocks, drift, regime switching to test robustness under stress conditions.  
- **Runtime Executor** → executes strategies in controlled loops, supporting rule-based logic, ML-driven approaches, or hybrid models.  
- **Evaluator & Metrics** → computes fidelity, improvement, Sharpe ratios, drawdowns, and aggregates multi-run statistics.  
- **Optimizer** → manages sweeps across hyperparameters, seeds, and scenarios to search for robust configurations.  
- **Reporter** → consolidates outcomes into CSV, JSON, or HTML, providing transparency and reproducibility.

---

## 🔬 Implementation

- **Language**: Python 3.11+  
- **Core libraries**: NumPy, SciPy, scikit-learn, Optuna  
- **Scenario engine**: supports synthetic + realistic datasets, with random seed control for reproducibility.  
- **Cost models**: fees, slippage, and latency included for financial use cases, but designed as pluggable modules for other domains.  
- **Metrics**: fidelity, improvement, risk-adjusted returns, Sharpe ratios, max drawdowns.  
- **CLI**: YAML/JSON-driven configuration allows experiment replication and automation.  

🔬 *Technical note*: Each module is designed to be **independent and testable**, making the framework easy to extend or embed into larger workflows.

---

## 🚀 Why It’s Different

- **Deterministic vs Probabilistic**  
  - Standard early stopping (e.g. Keras, PyTorch callbacks) halts training at the first plateau.  
  - Felix maintains a *superposed state* until evidence is decisive, avoiding premature exits.  

- **Noise Robustness**  
  - Small fluctuations do not force training to stop, reducing the risk of false convergence.  

- **Generalizable**  
  - Though inspired by trading strategies, the approach applies to robotics, IoT, energy, and anomaly detection.  

🔬 *Technical note*: The probabilistic stopping mechanism can be tuned with parameters controlling sensitivity to fidelity vs. improvement, effectively letting users balance **exploration vs. convergence**.

---

## 🐈 Why the Name "Quantum Felix"?

The name **Quantum Felix** is directly inspired by *Schrödinger’s Cat*, one of the most famous thought experiments in quantum mechanics.  

- *Felix* (Latin for “cat”) reflects the metaphor of a system existing in a **superposition of states** until observed.  
- Just like the cat is simultaneously *alive* and *dead* before the box is opened, training in Quantum Felix is simultaneously in *continue* and *stop* states.  
- The “collapse” of the state in Felix corresponds to the **probabilistic decision** to either halt or extend training.  

This naming highlights the project’s **quantum-inspired philosophy**: managing uncertainty not by rigid thresholds, but through **probabilistic reasoning and dynamic adaptation**.

---

## 🌍 Example Use Cases

- **Finance** — prevents discarding profitable strategies too early in noisy markets.  
- **Energy & IoT** — robust to drift in demand/load forecasts and irregular signals.  
- **Industrial Control** — predictive maintenance benefits from delay in premature halts when variance is high.  
- **Robotics** — ensures policies continue improving even under inconsistent feedback.  
- **Cybersecurity** — supports anomaly simulations and resilience testing under adversarial or rare conditions.  

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
