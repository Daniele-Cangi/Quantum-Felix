
# 🐱 Quantum Felix — Simulation & Strategy Engine inspired by Schrödinger’s Cat  

Quantum Felix is a research-preview engine for multi-scenario simulation, strategy backtesting, and stress-testing under uncertainty.  
Its unique feature is a *Schrödinger’s Cat–inspired probabilistic early stopping mechanism*, where training continues or halts based on dynamic probability amplitudes (`psi_alive`, `psi_dead`, `cat_fidelity`).  

---

## ✨ Features  

- **Scenario diversity** → synthetic + realistic data generation  
- **Systematic evaluation** → sweeps, multi-seeds, robust metrics  
- **Auditability** → configs, seeds, and results stored for reproducibility  
- **Quantum-inspired early stopping** → avoids premature convergence, captures non-linear dynamics  

---

### 🐱 Schrödinger’s Cat Analogy  

The idea of *Quantum Early Stopping* is inspired by Schrödinger’s famous thought experiment:  
- A cat inside a box is **both alive and dead** until observed.  
- Its state is represented by a **superposition** of two probability amplitudes.  

In Quantum Felix:  
- `psi_alive` = amplitude of the model being in the *continue training* state.  
- `psi_dead` = amplitude of the model being in the *stop training* state.  
- As training progresses, these amplitudes are updated based on **fidelity** (how consistent the current run is with the best past) and **improvement** (measured progress).  
- At each step, the algorithm performs a **“collapse”** — sampling or thresholding to decide whether training continues or halts.  

👉 This probabilistic framing allows the system to **delay premature stopping**, while still converging when evidence accumulates that improvement has plateaued.  

---

## 📖 Why It Matters  

Simulation frameworks often trade off between **flexibility** and **realism**.  
Quantum Felix aims to provide:  
- **Scenario diversity** → synthetic + realistic mixing  
- **Systematic evaluation** → sweeps, multi-seeds, robust metrics  
- **Auditability** → configs, seeds, and results stored for reproducibility  

This makes it useful for **finance, IoT, robotics, energy, and anomaly detection**, where strategies must be validated under uncertainty and stress.  

---

## 🌍 Example Use Cases  

- 📈 **Finance & Trading** — backtest strategies with realistic cost models and stress tests.  
- ⚡ **Energy & IoT** — demand/load simulations with drift and anomaly injection.  
- 🏭 **Industrial Control** — predictive maintenance with multi-scenario simulations.  
- 🤖 **Robotics** — what-if testing of policies under uncertainty and latency constraints.  
- 🛡️ **Cybersecurity** — anomaly simulation and robust response evaluation.  

---

## 📂 Repository Structure (planned)  

```
quantum-felix/
├─ src/felix/
│  ├─ quantum_cat.py       # main quantum-inspired early stopping engine
│  ├─ __main__.py          # entry point for `python -m felix`
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
