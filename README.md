 Q-Learning Stock Trading Agent - Feature Prototype

**Final Year Project - CM3070 Artificial Intelligence**  
**University of London**

A tabular Q-learning agent that learns stock trading strategies through reinforcement learning. This prototype demonstrates the feasibility of applying RL to portfolio management and serves as the foundation for a Deep Q-Network implementation.

## 🎯 Project Overview

This prototype implements the Watkins & Dayan (1992) Q-learning algorithm for stock trading. The agent learns to make buy, hold, and sell decisions by interacting with historical market data and optimizing portfolio value over time.

**Key Features:**
- Tabular Q-learning implementation from scratch
- 40-state discrete representation (RSI × Trend × Position)
- Realistic trading environment with transaction costs
- Comprehensive evaluation using financial metrics
- Professional visualizations and analysis

## 📊 Results Summary

**Training:**
- 100 episodes with epsilon-greedy exploration
- Converged successfully (rewards: 3.79 → 77.78)
- Final epsilon: 0.01 (exploration → exploitation)

**Testing Performance:**
| Metric | Q-Learning | Buy-and-Hold |
|--------|------------|--------------|
| Total Return | -5.19% | +165.59% |
| Sharpe Ratio | -0.28 | 2.35 |
| Max Drawdown | -12.17% | -25.10% |
| Trades | 32 | 2 |

**Key Finding:** The agent underperformed buy-and-hold in absolute returns, which aligns with academic consensus (Fischer 2018, Sun et al. 2023) that basic RL methods typically match rather than beat baselines. However, lower maximum drawdown demonstrates risk management capability.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rl-stock-trading-prototype.git
cd rl-stock-trading-prototype

# Install dependencies
pip install -r requirements.txt
```

### Running the Prototype

**Quick Demo (30 seconds):**
```bash
python demo_script.py
```

**Full Experiment (2 minutes):**
```bash
python visualize_results.py
```

## 📁 Project Structure

```
rl-stock-trading-prototype/
├── tabular_q_trading.py      # Main prototype implementation
├── demo_script.py             # Quick demo (20 episodes)
├── visualize_results.py       # Full experiment + visualizations
├── requirements.txt           # Python dependencies
├── results/                   # Pre-generated results
│   ├── learning_curve.png
│   ├── portfolio_comparison.png
│   ├── q_table_heatmap.png
│   ├── action_distribution.png
│   ├── trades_on_price.png
│   └── prototype_analysis.txt
└── README.md
```

## 🧠 Technical Details

### State Space (40 states)
- **RSI bins:** [0-30 (oversold), 30-50, 50-70, 70-100 (overbought)]
- **Trend bins:** [strong down, down, neutral, up, strong up]
- **Position:** [0 (no stock), 1 (holding stock)]

### Action Space (3 actions)
- **0:** Hold (maintain current position)
- **1:** Buy (if not holding)
- **2:** Sell (if holding)

### Reward Function
- Percentage change in portfolio value
- Includes 0.1% transaction costs

### Q-Learning Parameters
- Learning rate (α): 0.1
- Discount factor (γ): 0.95
- Epsilon decay: 0.995
- Episodes: 100

## 📈 Visualizations

The prototype generates five professional visualizations:

1. **Learning Curve** - Training progress over 100 episodes
2. **Portfolio Comparison** - RL agent vs buy-and-hold baseline
3. **Q-Table Heatmap** - Learned state-action values
4. **Action Distribution** - Trading behavior analysis
5. **Trades on Price** - Buy/sell decisions overlaid on price chart

See `results/` folder for examples.

## 🔬 Evaluation

Performance is evaluated using standard financial metrics:

- **Total Return**: Overall profitability
- **Sharpe Ratio**: Risk-adjusted performance
- **Maximum Drawdown**: Worst peak-to-trough decline

Compared against buy-and-hold baseline following industry best practices.

## 💡 Key Insights

### Strengths
✅ Successfully implements Q-learning from scratch  
✅ Agent converges and learns stable policy  
✅ Demonstrates active trading behavior (32 trades)  
✅ Lower drawdown shows risk management  
✅ Code is modular and extensible to DQN  

### Limitations
⚠️ Limited state representation (only 40 states)  
⚠️ Simple features (RSI and trend only)  
⚠️ Single stock (no diversification)  
⚠️ Underperforms in strong uptrend markets  

### Next Steps
🚀 Implement Deep Q-Network with continuous states  
🚀 Expand to multi-stock portfolio management  
🚀 Add risk-adjusted reward function (Sharpe ratio)  
🚀 Incorporate additional features (volume, volatility, sentiment)  

## 📚 Academic Foundation

This prototype is based on:

- **Watkins & Dayan (1992).** "Q-learning." *Machine Learning*, 8(3-4), 279-292.
- **Fischer (2018).** "Reinforcement learning in financial markets - a survey." *FAU Discussion Papers*.
- **Sun et al. (2023).** "Reinforcement Learning for Quantitative Trading." *ACM Transactions on Intelligent Systems and Technology*.

## 🛠️ Dependencies

- Python 3.8+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- yfinance >= 0.2.0

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

**Jonathan**  
Final Year Project - CM3070 Artificial Intelligence  
University of London (UOL-SIM)  
November 2025

## 🎓 Academic Context

This prototype is submitted as part of the preliminary report for the Final Year Project (CM3070). It demonstrates:

1. **Technical feasibility** of RL-based trading systems
2. **Implementation capability** from academic literature to working code
3. **Critical evaluation** using appropriate financial metrics
4. **Foundation** for Deep Q-Network implementation in final system

## 📞 Contact

For questions about this prototype, please contact through GitHub issues or university channels.

---

**Note:** This is a research prototype for educational purposes. Not intended as financial advice or real trading system.
