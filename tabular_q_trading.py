"""
Tabular Q-Learning Stock Trading Agent
Feature Prototype for Final Year Project
Author: Jonathan
Date: November 2025

This implements a tabular Q-learning agent for stock trading to demonstrate
the feasibility of using reinforcement learning for portfolio management.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingEnvironment:
    """
    Stock trading environment with discrete state space.
    
    State representation:
    - RSI bins: [0-30 (oversold), 30-50, 50-70, 70-100 (overbought)]
    - Trend bins: [strong down, down, neutral, up, strong up]
    - Position: [0 (no stock), 1 (holding stock)]
    
    Total states: 4 * 5 * 2 = 40 states
    Actions: 0=Hold, 1=Buy, 2=Sell
    """
    
    def __init__(self, data, initial_capital=10000, transaction_cost=0.001):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 14  # Start after indicators are available
        self.capital = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.trades = []
        return self._get_state()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50  # Neutral RSI if not enough data
        # Ensure prices are a 1-D numeric array to avoid object/array elements
        prices = np.asarray(prices, dtype=float).ravel()
        deltas = np.diff(prices)
        # Use the first `period` deltas to seed Wilder's smoothing (no off-by-one)
        seed = deltas[:period]
        # Separate positive and negative moves (strict > / <) for correct averages
        up = seed[seed > 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / (down + 1e-10)
        # If there was no movement in the seed window, RSI is neutral (50)
        if up == 0 and down == 0:
            rsi = 50.0
        else:
            rsi = 100 - (100 / (1 + rs))

        # Calculate rolling RSI
        rsi_values = [rsi]
        # Iterate directly over remaining deltas and coerce to scalar floats
        for delta in deltas[period:]:
            # Coerce potential 0-d / 1-d numpy types to Python float safely
            delta = float(np.asarray(delta).ravel()[0])
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = float(-delta)
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / (down + 1e-10)
            # If no movement, keep RSI neutral
            if up == 0 and down == 0:
                rsi_values.append(50.0)
            else:
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values[-1] if rsi_values else 50
    
    def _calculate_trend(self, prices, short_window=5, long_window=10):
        """Calculate trend using moving averages"""
        if len(prices) < long_window:
            return 0  # Neutral
        
        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices[-long_window:])
        
        diff_pct = (short_ma - long_ma) / long_ma * 100
        
        if diff_pct < -2:
            return 0  # Strong down
        elif diff_pct < -0.5:
            return 1  # Down
        elif diff_pct < 0.5:
            return 2  # Neutral
        elif diff_pct < 2:
            return 3  # Up
        else:
            return 4  # Strong up
    
    def _get_state(self):
        """Get current discrete state"""
        prices = self.data['Close'].iloc[:self.current_step+1].values
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices)
        if rsi < 30:
            rsi_bin = 0  # Oversold
        elif rsi < 50:
            rsi_bin = 1
        elif rsi < 70:
            rsi_bin = 2
        else:
            rsi_bin = 3  # Overbought
        
        # Calculate trend
        trend_bin = self._calculate_trend(prices)
        
        # Position (0 = no stock, 1 = holding)
        position = 1 if self.shares > 0 else 0
        
        # State is combination of all features
        state = rsi_bin * 10 + trend_bin * 2 + position
        return state
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done)
        Actions: 0=Hold, 1=Buy, 2=Sell
        """
        current_price = self.data['Close'].iloc[self.current_step]
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.shares == 0 and self.capital > 0:
                # Buy as many shares as possible
                cost = current_price * (1 + self.transaction_cost)
                shares_to_buy = int(self.capital / cost)
                if shares_to_buy > 0:
                    self.shares = shares_to_buy
                    self.capital -= shares_to_buy * cost
                    self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))
        
        elif action == 2:  # Sell
            if self.shares > 0:
                # Sell all shares
                proceeds = self.shares * current_price * (1 - self.transaction_cost)
                self.capital += proceeds
                self.trades.append(('SELL', self.current_step, current_price, self.shares))
                self.shares = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward (change in portfolio value)
        new_portfolio_value = self.capital + self.shares * self.data['Close'].iloc[self.current_step]
        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value * 100  # Percentage change
        self.portfolio_value = new_portfolio_value
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done

class QLearningAgent:
    """
    Tabular Q-Learning agent for stock trading.
    
    Uses standard Q-learning update:
    Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    """
    
    def __init__(self, n_states=40, n_actions=3, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
    
    def decay_epsilon(self):
        """Decay exploration rate once per episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train(self, env, episodes=100):
        """Train the agent"""
        episode_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state, training=True)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
            
            # Decay epsilon once per episode to control exploration rate
            self.decay_epsilon()
            
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode+1}/{episodes}, Avg Reward: {np.mean(episode_rewards[-20:]):.2f}, Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards
    
    def test(self, env):
        """Test the agent (no exploration)"""
        state = env.reset()
        done = False
        actions_taken = []
        
        while not done:
            action = self.get_action(state, training=False)
            actions_taken.append(action)
            next_state, reward, done = env.step(action)
            state = next_state
        
        return env.portfolio_value, env.trades, actions_taken

def download_data(ticker='AAPL', start_date='2020-01-01', end_date='2024-11-01', use_synthetic=False):
    """
    Download stock data from Yahoo Finance or generate synthetic data.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        use_synthetic: If True, skip download and use synthetic data
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    
    if not use_synthetic:
        print(f"Attempting to download {ticker} data from Yahoo Finance...")
        print(f"Period: {start_date} to {end_date}")
        
        try:
            # Try to download with yfinance
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False,
                repair=True  # Automatically repair missing data
            )
            
            if len(data) > 0:
                # Flatten yfinance MultiIndex columns (Price x Ticker) for single-ticker downloads
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        data = data.droplevel('Ticker', axis=1)
                    except (KeyError, ValueError):
                        data.columns = data.columns.get_level_values(0)
                # Drop helper columns we do not need in the trading env
                for col in ['Repaired?', 'Dividends', 'Stock Splits']:
                    if col in data.columns:
                        data = data.drop(columns=col)
                # Keep consistent column ordering for downstream consumers
                ordered_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in data.columns]
                if ordered_cols:
                    data = data[ordered_cols]

                print(f"✓ Successfully downloaded {len(data)} days of {ticker} data")
                print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
                min_price = float(data['Close'].min())
                max_price = float(data['Close'].max())
                print(f"  Price range: ${min_price:.2f} to ${max_price:.2f}")
                return data
            else:
                print("⚠ Download returned empty dataset")
                
        except Exception as e:
            print(f"⚠ Download failed: {str(e)[:100]}")
    
    # Fallback to synthetic data
    print("\n→ Using synthetic data (reproducible for testing)")
    print("  Note: Synthetic data is perfectly acceptable for prototype demonstration")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic stock price with trend and volatility
    # Uses geometric Brownian motion (standard in finance)
    np.random.seed(42)  # Fixed seed for reproducibility
    n_days = len(dates)
    
    # Parameters calibrated to match real stock behavior
    mu = 0.0005       # Daily drift (0.05% per day ≈ 13% annual)
    sigma = 0.02      # Daily volatility (2% ≈ 32% annual)
    
    # Generate price path
    returns = np.random.normal(mu, sigma, n_days)
    price = 150 * np.exp(np.cumsum(returns))  # Start at $150
    
    # Add seasonal pattern (realistic for some stocks)
    seasonal = 10 * np.sin(np.arange(n_days) / 50)
    price += seasonal
    
    # Generate OHLC data (Open, High, Low, Close)
    data = pd.DataFrame({
        'Open': price * (1 + np.random.normal(0, 0.003, n_days)),
        'High': price * (1 + np.abs(np.random.normal(0.008, 0.003, n_days))),
        'Low': price * (1 - np.abs(np.random.normal(0.008, 0.003, n_days))),
        'Close': price,
        'Volume': np.random.randint(20000000, 60000000, n_days)
    }, index=dates)
    
    # Ensure High >= Open/Close and Low <= Open/Close (realistic constraint)
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    print(f"✓ Generated {len(data)} days of synthetic {ticker}-like data")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
    print(f"  Avg daily return: {(data['Close'].pct_change().mean() * 100):.3f}%")
    print(f"  Volatility: {(data['Close'].pct_change().std() * 100):.3f}%")
    
    return data

def calculate_metrics(returns, benchmark_returns=None):
    """Calculate performance metrics"""
    total_return = (returns[-1] / returns[0] - 1) * 100
    
    # Sharpe ratio (annualized)
    daily_returns = np.diff(returns) / returns[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
    
    # Max drawdown
    cummax = np.maximum.accumulate(returns)
    drawdown = (returns - cummax) / cummax * 100
    max_drawdown = np.min(drawdown)
    
    metrics = {
        'Total Return (%)': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown
    }
    
    if benchmark_returns is not None:
        benchmark_total = (benchmark_returns[-1] / benchmark_returns[0] - 1) * 100
        metrics['vs Benchmark (%)'] = total_return - benchmark_total
    
    return metrics

def run_experiment():
    """Run complete training and testing experiment"""
    
    # Download data
    data = download_data('AAPL', '2020-01-01', '2024-11-01')
    
    # Split into train/test
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    test_data = data.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\nTraining period: {len(train_data)} days")
    print(f"Testing period: {len(test_data)} days")
    
    # Create environments
    train_env = TradingEnvironment(train_data, initial_capital=10000)
    test_env = TradingEnvironment(test_data, initial_capital=10000)
    
    # Create agent
    agent = QLearningAgent(
        n_states=40,
        n_actions=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train agent
    print("\n" + "="*50)
    print("TRAINING Q-LEARNING AGENT")
    print("="*50)
    episode_rewards = agent.train(train_env, episodes=100)
    
    # Test agent
    print("\n" + "="*50)
    print("TESTING Q-LEARNING AGENT")
    print("="*50)
    final_value, trades, actions = agent.test(test_env)
    
    # Calculate baseline (buy and hold)
    baseline_env = TradingEnvironment(test_data, initial_capital=10000)
    baseline_env.step(1)  # Buy at start
    while baseline_env.current_step < len(test_data) - 2:  # Stop before last step
        baseline_env.step(0)  # Hold
    baseline_value = baseline_env.portfolio_value
    
    # Print results
    print(f"\nQ-Learning Final Value: ${final_value:,.2f}")
    print(f"Buy-and-Hold Final Value: ${baseline_value:,.2f}")
    print(f"Profit: ${final_value - 10000:,.2f} ({(final_value/10000 - 1)*100:.2f}%)")
    print(f"Number of trades: {len(trades)}")
    
    # Calculate metrics
    rl_returns = [10000]
    for i in range(len(test_data)):
        test_env_temp = TradingEnvironment(test_data, initial_capital=10000)
        for j in range(i+1):
            if j < len(actions):
                test_env_temp.step(actions[j])
        rl_returns.append(test_env_temp.portfolio_value)
    
    baseline_returns = [10000]
    for i in range(len(test_data)):
        price_return = test_data['Close'].iloc[i] / test_data['Close'].iloc[0]
        baseline_returns.append(10000 * price_return)
    
    rl_metrics = calculate_metrics(np.array(rl_returns), np.array(baseline_returns))
    baseline_metrics = calculate_metrics(np.array(baseline_returns))
    
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    print(f"{'Metric':<25} {'Q-Learning':<15} {'Buy-and-Hold':<15}")
    print("-"*55)
    for key in ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']:
        print(f"{key:<25} {rl_metrics[key]:<15.2f} {baseline_metrics[key]:<15.2f}")
    
    # Save results
    results = {
        'agent': agent,
        'train_env': train_env,
        'test_env': test_env,
        'episode_rewards': episode_rewards,
        'trades': trades,
        'actions': actions,
        'rl_metrics': rl_metrics,
        'baseline_metrics': baseline_metrics,
        'rl_returns': rl_returns,
        'baseline_returns': baseline_returns,
        'test_data': test_data
    }
    
    return results

if __name__ == "__main__":
    np.random.seed(42)
    results = run_experiment()
    print("\n✓ Experiment complete!")
    print("✓ Results saved to 'results' variable")
