---
layout: page
title: Examples
permalink: /examples/
---

# 🚀 Quantum Felix Examples

Explore practical examples and tutorials to get started with Quantum Felix. Each example includes complete code, explanations, and expected outcomes.

## 🎯 Basic Examples

### 1. Simple Mean Reversion Strategy

This example demonstrates a basic mean reversion strategy using synthetic market data.

```python
import numpy as np
from felix import QuantumFelix
from felix.scenarios import SyntheticScenario
from felix.strategies import MeanReversionStrategy

# Create a synthetic market scenario
scenario = SyntheticScenario(
    duration=252,  # 1 year of trading days
    base_return=0.0005,  # Daily return: 0.05%
    volatility=0.02,     # Daily volatility: 2%
    trending_periods=[50, 100, 150],  # Trending periods
    shock_probability=0.01            # 1% chance of market shock
)

# Define mean reversion strategy
strategy = MeanReversionStrategy(
    lookback_window=20,     # 20-day moving average
    entry_threshold=2.0,    # Enter when 2 std devs from mean
    exit_threshold=0.5,     # Exit when 0.5 std devs from mean
    max_position_size=0.8   # Maximum 80% of capital
)

# Initialize simulation engine
engine = QuantumFelix()

# Run simulation
results = engine.run_simulation(
    scenario=scenario,
    strategy=strategy,
    initial_capital=100000,
    seeds=range(5)  # Run 5 different random seeds
)

# Print results
print("=== Mean Reversion Strategy Results ===")
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
```

**Expected Output:**
```
=== Mean Reversion Strategy Results ===
Total Return: 12.34%
Sharpe Ratio: 1.23
Max Drawdown: -8.45%
Win Rate: 67.32%
```

### 2. Momentum Strategy with Quantum Early Stopping

```python
from felix.strategies import MomentumStrategy
from felix.runtime.quantum_cat import QuantumEarlyStopping

# Create momentum strategy with quantum early stopping
strategy = MomentumStrategy(
    fast_window=12,         # Fast moving average
    slow_window=26,         # Slow moving average
    signal_window=9,        # Signal line
    entry_threshold=0.02,   # 2% momentum threshold
    quantum_stopping=QuantumEarlyStopping(
        patience=15,
        min_delta=0.001,
        fidelity_weight=0.7
    )
)

# Multi-asset scenario
scenario = SyntheticScenario(
    duration=500,
    assets=['BTC', 'ETH', 'ADA'],
    correlation_matrix=np.array([
        [1.0, 0.7, 0.5],
        [0.7, 1.0, 0.6],
        [0.5, 0.6, 1.0]
    ]),
    regime_switching=True
)

results = engine.run_simulation(scenario, strategy)

# Quantum stopping analysis
print("=== Quantum Early Stopping Analysis ===")
print(f"Training stopped at iteration: {results.quantum_stats.stop_iteration}")
print(f"Final psi_alive: {results.quantum_stats.final_psi_alive:.3f}")
print(f"Final psi_dead: {results.quantum_stats.final_psi_dead:.3f}")
print(f"Collapse probability: {results.quantum_stats.collapse_probability:.3f}")
```

## 📊 Advanced Examples

### 3. Multi-Factor Strategy with Risk Management

```python
from felix.strategies import MultiFactorStrategy
from felix.risk import RiskManager
from felix.scenarios import RealisticScenario

# Set up realistic market scenario using historical data
scenario = RealisticScenario(
    data_source='yahoo',
    symbols=['SPY', 'QQQ', 'IWM'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    benchmark='SPY'
)

# Create multi-factor strategy
strategy = MultiFactorStrategy(
    factors={
        'momentum': {
            'lookback': 252,
            'weight': 0.3
        },
        'mean_reversion': {
            'lookback': 20,
            'weight': 0.3
        },
        'volatility': {
            'lookback': 30,
            'weight': 0.2
        },
        'value': {
            'pe_ratio_threshold': 15,
            'weight': 0.2
        }
    },
    rebalance_frequency='monthly'
)

# Add risk management
risk_manager = RiskManager(
    max_portfolio_risk=0.15,    # 15% portfolio volatility limit
    max_single_position=0.1,    # 10% max position size
    stop_loss_threshold=-0.05,  # 5% stop loss
    var_limit=0.02,            # 2% daily VaR limit
    correlation_threshold=0.8   # Avoid highly correlated positions
)

strategy.add_risk_manager(risk_manager)

# Run simulation with transaction costs
results = engine.run_simulation(
    scenario=scenario,
    strategy=strategy,
    transaction_costs={
        'commission': 0.001,    # 0.1% commission
        'bid_ask_spread': 0.0005,  # 0.05% spread
        'market_impact': 0.0001    # 0.01% market impact
    }
)

print("=== Multi-Factor Strategy Results ===")
print(f"Total Return: {results.total_return:.2%}")
print(f"Benchmark Return: {results.benchmark_return:.2%}")
print(f"Alpha: {results.alpha:.2%}")
print(f"Information Ratio: {results.information_ratio:.2f}")
print(f"Maximum Drawdown: {results.max_drawdown:.2%}")

# Risk analysis
print("\n=== Risk Analysis ===")
print(f"Portfolio Volatility: {results.portfolio_volatility:.2%}")
print(f"VaR (95%): {results.var_95:.2%}")
print(f"Expected Shortfall: {results.expected_shortfall:.2%}")
print(f"Risk-Adjusted Return: {results.risk_adjusted_return:.2%}")
```

### 4. Machine Learning Strategy

```python
from felix.strategies.ml import MLStrategy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class CustomMLStrategy(MLStrategy):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.lookback = 60
        
    def create_features(self, data):
        """Create features for ML model"""
        features = []
        prices = data['close']
        
        # Technical indicators
        for window in [5, 10, 20, 50]:
            # Moving averages
            ma = prices.rolling(window).mean()
            features.append((prices / ma - 1).fillna(0))
            
            # Volatility
            volatility = prices.pct_change().rolling(window).std()
            features.append(volatility.fillna(0))
            
        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        features.append(rsi.fillna(50) / 100)
        
        # MACD
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        features.append(macd.fillna(0))
        
        return np.column_stack(features)
    
    def create_targets(self, data):
        """Create target variable (future returns)"""
        returns = data['close'].pct_change(5).shift(-5)  # 5-day forward return
        return returns.fillna(0)
    
    def train_model(self, features, targets):
        """Train the ML model"""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled, targets)
        
    def generate_signal(self, features):
        """Generate trading signal"""
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        
        # Convert prediction to position signal
        if prediction > 0.01:  # Expect > 1% return
            return 1.0  # Long position
        elif prediction < -0.01:  # Expect < -1% return
            return -1.0  # Short position
        else:
            return 0.0  # No position

# Use ML strategy
ml_strategy = CustomMLStrategy()

results = engine.run_simulation(
    scenario=scenario,
    strategy=ml_strategy,
    train_test_split=0.7,  # 70% for training, 30% for testing
    walk_forward=True      # Use walk-forward analysis
)

print("=== ML Strategy Results ===")
print(f"Out-of-sample Return: {results.oos_return:.2%}")
print(f"Feature Importance:")
for i, importance in enumerate(results.feature_importance):
    print(f"  Feature {i+1}: {importance:.3f}")
```

## 🧪 Specialized Examples

### 5. Regime Detection and Switching Strategy

```python
from felix.scenarios import RegimeSwitchingScenario
from felix.strategies import RegimeAwareStrategy

# Create regime-switching market
scenario = RegimeSwitchingScenario(
    regimes={
        'bull_market': {
            'duration_range': (60, 180),
            'return_mean': 0.001,
            'volatility': 0.015,
            'transition_prob': 0.02
        },
        'bear_market': {
            'duration_range': (30, 120),
            'return_mean': -0.0005,
            'volatility': 0.025,
            'transition_prob': 0.03
        },
        'sideways': {
            'duration_range': (90, 200),
            'return_mean': 0.0001,
            'volatility': 0.012,
            'transition_prob': 0.015
        }
    },
    initial_regime='bull_market'
)

# Regime-aware strategy
strategy = RegimeAwareStrategy(
    regime_detection_window=50,
    strategies_by_regime={
        'bull_market': MomentumStrategy(fast=12, slow=26),
        'bear_market': MeanReversionStrategy(lookback=20),
        'sideways': CoveredCallStrategy(delta_threshold=0.3)
    },
    regime_confidence_threshold=0.7
)

results = engine.run_simulation(scenario, strategy)

print("=== Regime-Aware Strategy Results ===")
print(f"Regime Detection Accuracy: {results.regime_accuracy:.2%}")
for regime, performance in results.performance_by_regime.items():
    print(f"{regime}: {performance.return:.2%} return, "
          f"{performance.sharpe:.2f} Sharpe")
```

### 6. Portfolio Optimization with Monte Carlo

```python
from felix.optimize import MonteCarloOptimizer
from felix.strategies import PortfolioOptimizationStrategy

# Define universe of strategies
strategy_universe = {
    'momentum': MomentumStrategy(fast=12, slow=26),
    'mean_revert': MeanReversionStrategy(lookback=20),
    'breakout': BreakoutStrategy(lookback=20, threshold=2.0),
    'pairs_trading': PairsTradingStrategy(correlation_window=60)
}

# Portfolio optimization strategy
portfolio_strategy = PortfolioOptimizationStrategy(
    strategy_universe=strategy_universe,
    optimization_method='max_sharpe',
    rebalance_frequency='quarterly',
    min_weight=0.05,  # Minimum 5% allocation
    max_weight=0.4    # Maximum 40% allocation
)

# Monte Carlo optimization
optimizer = MonteCarloOptimizer(
    n_simulations=1000,
    parameter_ranges={
        'momentum.fast': (8, 16),
        'momentum.slow': (20, 35),
        'mean_revert.lookback': (10, 30),
        'breakout.threshold': (1.5, 2.5)
    }
)

# Run optimization
optimal_results = optimizer.optimize(
    scenario=scenario,
    strategy=portfolio_strategy,
    objective='sharpe_ratio'
)

print("=== Portfolio Optimization Results ===")
print(f"Optimal Sharpe Ratio: {optimal_results.best_sharpe:.2f}")
print("Optimal Weights:")
for strategy, weight in optimal_results.optimal_weights.items():
    print(f"  {strategy}: {weight:.1%}")
print("Optimal Parameters:")
for param, value in optimal_results.optimal_params.items():
    print(f"  {param}: {value}")
```

## 📈 Real-World Case Studies

### 7. Cryptocurrency Trading Strategy

```python
from felix.data.crypto import CryptoDataFeed
from felix.strategies.crypto import CryptoArbitrageStrategy

# Real crypto data
crypto_scenario = RealisticScenario(
    data_source=CryptoDataFeed(
        exchanges=['binance', 'coinbase', 'kraken'],
        symbols=['BTC/USD', 'ETH/USD', 'ADA/USD'],
        timeframe='1h'
    ),
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Crypto arbitrage strategy
crypto_strategy = CryptoArbitrageStrategy(
    price_threshold=0.005,      # 0.5% price difference
    volume_threshold=10000,     # Minimum $10k volume
    execution_delay=0.1,        # 100ms execution delay
    transaction_fees={
        'binance': 0.001,
        'coinbase': 0.005,
        'kraken': 0.0026
    }
)

results = engine.run_simulation(
    scenario=crypto_scenario,
    strategy=crypto_strategy,
    slippage_model='crypto_enhanced'
)

print("=== Crypto Arbitrage Results ===")
print(f"Total Trades: {results.total_trades}")
print(f"Profitable Trades: {results.profitable_trades} ({results.win_rate:.1%})")
print(f"Average Trade Return: {results.avg_trade_return:.2%}")
print(f"Maximum Concurrent Positions: {results.max_positions}")
```

### 8. Energy Trading Strategy

```python
from felix.scenarios.energy import EnergyMarketScenario
from felix.strategies.energy import EnergyStorageStrategy

# Energy market with demand patterns
energy_scenario = EnergyMarketScenario(
    market_type='electricity',
    location='ERCOT',  # Texas electricity market
    include_weather=True,
    demand_seasonality=True,
    renewable_penetration=0.3
)

# Energy storage arbitrage
energy_strategy = EnergyStorageStrategy(
    storage_capacity=100,    # MWh
    max_charge_rate=25,      # MW
    max_discharge_rate=25,   # MW
    round_trip_efficiency=0.85,
    degradation_rate=0.0001,
    price_forecast_horizon=24  # hours
)

results = engine.run_simulation(
    scenario=energy_scenario,
    strategy=energy_strategy,
    simulation_frequency='hourly'
)

print("=== Energy Trading Results ===")
print(f"Total Revenue: ${results.total_revenue:,.2f}")
print(f"Storage Utilization: {results.storage_utilization:.1%}")
print(f"Peak Shaving Value: ${results.peak_shaving_value:,.2f}")
print(f"Grid Arbitrage Value: ${results.arbitrage_value:,.2f}")
```

## 🔧 Debugging and Analysis Examples

### 9. Performance Attribution Analysis

```python
from felix.analyze import PerformanceAttributor

# Run detailed performance attribution
attributor = PerformanceAttributor(
    benchmark='SPY',
    factor_models=['fama_french_3', 'carhart_4'],
    attribution_frequency='monthly'
)

# Analyze results
attribution = attributor.analyze(results)

print("=== Performance Attribution ===")
print(f"Alpha: {attribution.alpha:.2%}")
print(f"Beta: {attribution.beta:.2f}")
print("\nFactor Exposures:")
for factor, exposure in attribution.factor_exposures.items():
    print(f"  {factor}: {exposure:.3f}")

print("\nReturn Attribution:")
for source, contribution in attribution.return_attribution.items():
    print(f"  {source}: {contribution:.2%}")

# Plot attribution
attribution.plot_attribution_tree()
attribution.plot_rolling_attribution()
```

### 10. Stress Testing Framework

```python
from felix.testing import StressTester

# Define stress scenarios
stress_scenarios = {
    'market_crash': {
        'type': 'shock',
        'magnitude': -0.20,  # 20% market drop
        'duration': 5,       # 5 days
        'recovery_time': 30  # 30 days to recover
    },
    'volatility_spike': {
        'type': 'volatility_regime',
        'vol_multiplier': 3.0,
        'duration': 60
    },
    'liquidity_crisis': {
        'type': 'liquidity_shock',
        'bid_ask_multiplier': 5.0,
        'market_impact_multiplier': 3.0,
        'duration': 20
    },
    'black_swan': {
        'type': 'tail_event',
        'probability': 0.001,
        'magnitude': -0.35,
        'correlation_breakdown': True
    }
}

# Run stress tests
stress_tester = StressTester(base_scenario=scenario)

for stress_name, stress_config in stress_scenarios.items():
    stress_results = stress_tester.run_stress_test(
        strategy=strategy,
        stress_scenario=stress_config,
        n_simulations=100
    )
    
    print(f"\n=== {stress_name.title()} Stress Test ===")
    print(f"Worst Case Return: {stress_results.worst_case:.2%}")
    print(f"Expected Return: {stress_results.expected_return:.2%}")
    print(f"Recovery Time: {stress_results.avg_recovery_time:.1f} days")
    print(f"Probability of Ruin: {stress_results.ruin_probability:.2%}")
```

## 📊 Visualization Examples

### 11. Interactive Results Dashboard

```python
from felix.visualization import InteractiveDashboard

# Create interactive dashboard
dashboard = InteractiveDashboard(results)

# Add various plots
dashboard.add_plot('cumulative_returns', title='Cumulative Returns')
dashboard.add_plot('drawdown_periods', title='Drawdown Analysis')
dashboard.add_plot('rolling_sharpe', window=60, title='Rolling 60-Day Sharpe')
dashboard.add_plot('position_heatmap', title='Position Size Over Time')
dashboard.add_plot('trade_analysis', title='Trade Statistics')

# Generate and save dashboard
dashboard.generate(output_file='quantum_felix_results.html')
print("Interactive dashboard saved to quantum_felix_results.html")

# Or serve it locally
dashboard.serve(port=8080)
print("Dashboard available at http://localhost:8080")
```

## 🎓 Learning Path

### Beginner → Intermediate → Advanced

1. **Start Here (Beginner)**:
   - Example 1: Simple Mean Reversion Strategy
   - Example 2: Momentum Strategy with Quantum Early Stopping

2. **Build Skills (Intermediate)**:
   - Example 3: Multi-Factor Strategy with Risk Management
   - Example 5: Regime Detection and Switching Strategy

3. **Master Techniques (Advanced)**:
   - Example 4: Machine Learning Strategy
   - Example 6: Portfolio Optimization with Monte Carlo
   - Examples 7-8: Real-world case studies

4. **Expert Level**:
   - Examples 9-11: Analysis and debugging frameworks
   - Create your own custom strategies and scenarios

## 📝 Next Steps

- Explore the [documentation]({{ '/documentation/' | relative_url }}) for detailed API reference
- Check out the [features]({{ '/features/' | relative_url }}) page for comprehensive capabilities
- Visit our [GitHub repository](https://github.com/{{ site.repository }}) for the latest code
- Join our community discussions and contribute your own examples!

---

**Need help?** Open an issue on [GitHub](https://github.com/{{ site.repository }}/issues) or check our documentation for more detailed explanations.