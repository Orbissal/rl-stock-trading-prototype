"""
Deep Q-Network (DQN) Stock Trading Agent
CM3070 Final Year Project - Tier 2 Implementation
Author: Jonathan
Date: 2026

This builds directly on the tabular Q-learning prototype (Tier 1).
The key upgrade: instead of a 40-state lookup table, we use a neural
network that accepts 11 continuous market features and outputs Q-values
for all 3 actions (Hold, Buy, Sell).

Architecture follows Mnih et al. (2015) with two key stabilisation
mechanisms:
  1. Experience Replay Buffer - breaks correlation between consecutive
     training samples
  2. Target Network - provides stable Q-value targets during training

References:
  Mnih et al. (2015). Human-level control through deep reinforcement
    learning. Nature, 518(7540), 529-533.
  Liu et al. (2020). FinRL: A deep reinforcement learning library for
    automated stock trading. NeurIPS 2020 Workshop.
  Moody & Saffell (1998). Reinforcement learning for trading systems
    and portfolios. IEEE/IAFE Conference.
"""

import numpy as np
import pandas as pd
import os
import warnings
import random
from collections import deque
from pathlib import Path

# Suppress TensorFlow informational messages (oneDNN warnings etc.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Suppress additional TF logging
tf.get_logger().setLevel('ERROR')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (same as Tier 1)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING
# Loads from local CSV files saved by save_data.py
# Falls back to yfinance if CSV not found (requires internet)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(ticker='AAPL', start_date='2018-01-01', end_date='2024-11-01'):
    """
    Load stock data. Tries local CSV first (no internet needed),
    falls back to yfinance download if CSV not found.

    Args:
        ticker: Stock symbol e.g. 'AAPL', 'MSFT', 'GOOGL'
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'

    Returns:
        DataFrame with OHLCV columns, DatetimeIndex
    """
    csv_path = f'data_{ticker}.csv'

    if os.path.exists(csv_path):
        print(f"Loading {ticker} from local file: {csv_path}")
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # Flatten MultiIndex columns if present (yfinance sometimes saves these)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Filter to requested date range
        data = data.loc[start_date:end_date]
        print(f"  Loaded {len(data)} days ({data.index[0].date()} to {data.index[-1].date()})")
        return data

    else:
        print(f"No local file found for {ticker}, downloading from Yahoo Finance...")
        try:
            import yfinance as yf
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            print(f"  Downloaded {len(data)} days")
            return data
        except Exception as e:
            raise RuntimeError(
                f"Could not load data for {ticker}. "
                f"Run save_data.py while online first. Error: {e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DQN TRADING ENVIRONMENT
#
# This is an upgraded version of TradingEnvironment from tabular_q_trading.py.
# The only meaningful change is _get_state(), which now returns an 11-dimensional
# numpy array of continuous features instead of a single integer (0-39).
#
# Everything else — step(), reset(), reward calculation, transaction costs —
# is identical to Tier 1. This is deliberate: it keeps the comparison clean.
# Any performance difference between Tier 1 and Tier 2 is due to the richer
# state representation, not changes in environment dynamics.
# ─────────────────────────────────────────────────────────────────────────────

class DQNTradingEnvironment:
    """
    Stock trading environment for DQN agent.

    State: 11-dimensional continuous vector (see _get_state for details)
    Actions: 0=Hold, 1=Buy, 2=Sell
    Reward: Percentage change in portfolio value with transaction cost penalty

    Identical reward and action logic to TradingEnvironment in tabular_q_trading.py.
    Only _get_state() differs — returns continuous vector vs discrete integer.
    """

    # Number of features in the state vector
    STATE_SIZE = 11

    def __init__(self, data, initial_capital=10000, transaction_cost=0.001):
        """
        Args:
            data: DataFrame with at least 'Close' and 'Volume' columns
            initial_capital: Starting portfolio value in dollars
            transaction_cost: Fraction of trade value charged per trade (0.1% = 0.001)
        """
        self.data = data.copy().reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        # Pre-compute Close prices as a plain numpy array once
        # This avoids repeated pandas indexing inside the loop (faster training)
        self.closes = self.data['Close'].values.astype(float)
        self.volumes = self.data['Volume'].values.astype(float)

        # We need at least 20 days of history before we can compute all features
        # (Bollinger Bands use a 20-day window)
        self.START_STEP = 20

        self.reset()

    def reset(self):
        """Reset environment to initial state. Called at the start of each episode."""
        self.current_step = self.START_STEP
        self.capital = float(self.initial_capital)
        self.shares = 0
        self.portfolio_value = float(self.initial_capital)
        self.trades = []
        self.portfolio_history = [self.portfolio_value]
        return self._get_state()

    # ── Feature Engineering ──────────────────────────────────────────────────

    def _rsi(self, prices, period=14):
        """
        Relative Strength Index — momentum indicator.
        Same implementation as Tier 1 (Wilder smoothing method).
        Returns value in [0, 100], normalised to [0, 1] for the state vector.
        """
        if len(prices) < period + 1:
            return 0.5  # Neutral if insufficient history

        prices = np.asarray(prices, dtype=float).ravel()
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed > 0].sum() / period
        down = -seed[seed < 0].sum() / period

        if up == 0 and down == 0:
            return 0.5

        rs = up / (down + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        for delta in deltas[period:]:
            delta = float(np.asarray(delta).ravel()[0])
            upval = max(delta, 0.0)
            downval = max(-delta, 0.0)
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / (down + 1e-10)
            rsi = 100 - (100 / (1 + rs))

        return rsi / 100.0  # Normalise to [0, 1]

    def _macd(self, prices):
        """
        MACD (Moving Average Convergence Divergence).
        MACD line = EMA(12) - EMA(26)
        Signal line = EMA(9) of MACD line

        Returns (macd_normalised, signal_normalised) both in approximately [-1, 1].
        Cited in Sun et al. (2023) as effective feature for RL trading agents.
        """
        prices = np.asarray(prices, dtype=float)

        def ema(data, span):
            alpha = 2.0 / (span + 1)
            result = np.zeros(len(data))
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result

        if len(prices) < 26:
            return 0.0, 0.0

        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        macd_line = ema12 - ema26

        if len(macd_line) < 9:
            return 0.0, 0.0

        signal_line = ema(macd_line, 9)

        # Normalise by recent price standard deviation to make scale-invariant
        std = np.std(prices[-20:]) + 1e-10
        return float(macd_line[-1] / std), float(signal_line[-1] / std)

    def _bollinger_position(self, prices, period=20):
        """
        Bollinger Band position: where is current price within the bands?
        Returns value in [0, 1]:
          0 = at or below lower band (oversold)
          0.5 = at middle band (neutral)
          1 = at or above upper band (overbought)

        Bands: middle ± 2 standard deviations (standard definition).
        """
        if len(prices) < period:
            return 0.5

        window = prices[-period:]
        middle = np.mean(window)
        std = np.std(window) + 1e-10
        upper = middle + 2 * std
        lower = middle - 2 * std

        current = prices[-1]
        band_range = upper - lower + 1e-10
        position = (current - lower) / band_range
        return float(np.clip(position, 0.0, 1.0))

    def _volume_ratio(self, volumes, period=20):
        """
        Current volume relative to 20-day average.
        Normalised: 1.0 means exactly average volume.
        Clipped to [0, 3] then scaled to [0, 1].
        High volume often confirms price moves (Sun et al. 2023).
        """
        if len(volumes) < period:
            return 0.5
        avg_volume = np.mean(volumes[-period:]) + 1e-10
        ratio = volumes[-1] / avg_volume
        return float(np.clip(ratio / 3.0, 0.0, 1.0))

    def _ma_ratio(self, prices, short=5, long=20):
        """
        Short moving average / long moving average ratio.
        Captures trend direction (same concept as Tier 1's trend_bin but continuous).
        Normalised: 0.5 = neutral, >0.5 = uptrend, <0.5 = downtrend.
        """
        if len(prices) < long:
            return 0.5
        short_ma = np.mean(prices[-short:])
        long_ma = np.mean(prices[-long:]) + 1e-10
        ratio = short_ma / long_ma
        # Map ratio around 1.0 to [0, 1]: clip to [0.9, 1.1] then scale
        normalised = (ratio - 0.9) / 0.2
        return float(np.clip(normalised, 0.0, 1.0))

    def _momentum(self, prices, period=5):
        """
        5-day price rate of change: (current - 5 days ago) / 5 days ago.
        Clipped to [-10%, +10%] then scaled to [0, 1].
        """
        if len(prices) < period + 1:
            return 0.5
        roc = (prices[-1] - prices[-period - 1]) / (prices[-period - 1] + 1e-10)
        # Scale: -10% maps to 0, +10% maps to 1
        normalised = (roc + 0.10) / 0.20
        return float(np.clip(normalised, 0.0, 1.0))

    def _volatility(self, prices, period=20):
        """
        20-day rolling volatility (standard deviation of daily returns).
        Annualised and normalised: 0% vol → 0, 60% vol → 1.
        High volatility signals risk; used for position-sizing awareness.
        """
        if len(prices) < period + 1:
            return 0.3  # Assume moderate volatility if insufficient history
        returns = np.diff(prices[-period - 1:]) / (prices[-period - 1:-1] + 1e-10)
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        return float(np.clip(annual_vol / 0.60, 0.0, 1.0))

    def _daily_return(self, prices):
        """
        Today's return vs yesterday.
        Clipped to [-5%, +5%] then scaled to [0, 1].
        """
        if len(prices) < 2:
            return 0.5
        ret = (prices[-1] - prices[-2]) / (prices[-2] + 1e-10)
        normalised = (ret + 0.05) / 0.10
        return float(np.clip(normalised, 0.0, 1.0))

    def _get_state(self):
        """
        Construct the 11-dimensional state vector for the current timestep.

        All features are normalised to [0, 1] or approximately [-1, 1]
        to ensure stable neural network training (prevents any single
        feature from dominating gradient updates).

        State vector layout:
          [0]  RSI (14-period) — momentum
          [1]  MACD line — trend momentum
          [2]  MACD signal — trend signal
          [3]  Bollinger Band position — mean reversion signal
          [4]  Volume ratio — market participation
          [5]  MA ratio (5/20 day) — trend direction
          [6]  5-day momentum — short-term price change
          [7]  20-day volatility — risk level
          [8]  Daily return — immediate price signal
          [9]  Position (0=cash, 1=holding) — portfolio state
          [10] Capital ratio — how much capital remains
        """
        # Get price and volume history up to current step
        prices = self.closes[:self.current_step + 1]
        volumes = self.volumes[:self.current_step + 1]

        macd, signal = self._macd(prices)

        state = np.array([
            self._rsi(prices),                    # [0] RSI
            np.clip(macd, -1.0, 1.0),             # [1] MACD line
            np.clip(signal, -1.0, 1.0),           # [2] MACD signal
            self._bollinger_position(prices),      # [3] Bollinger position
            self._volume_ratio(volumes),           # [4] Volume ratio
            self._ma_ratio(prices),               # [5] MA ratio
            self._momentum(prices),               # [6] 5-day momentum
            self._volatility(prices),             # [7] Volatility
            self._daily_return(prices),           # [8] Daily return
            float(1 if self.shares > 0 else 0),  # [9] Position
            float(self.capital / self.initial_capital),  # [10] Capital ratio
        ], dtype=np.float32)

        return state

    def step(self, action):
        """
        Execute one trading action.
        Identical logic to Tier 1 TradingEnvironment.step().

        Args:
            action: int, 0=Hold, 1=Buy, 2=Sell

        Returns:
            (next_state, reward, done)
            next_state: numpy array of shape (11,), or None if done
            reward: float, percentage portfolio change
            done: bool
        """
        current_price = float(self.closes[self.current_step])

        # ── Execute Action ────────────────────────────────────────────────
        if action == 1:  # Buy
            if self.shares == 0 and self.capital > 0:
                cost_per_share = current_price * (1 + self.transaction_cost)
                shares_to_buy = int(self.capital / cost_per_share)
                if shares_to_buy > 0:
                    self.shares = shares_to_buy
                    self.capital -= shares_to_buy * cost_per_share
                    self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))

        elif action == 2:  # Sell
            if self.shares > 0:
                proceeds = self.shares * current_price * (1 - self.transaction_cost)
                self.capital += proceeds
                self.trades.append(('SELL', self.current_step, current_price, self.shares))
                self.shares = 0

        # ── Advance Time ──────────────────────────────────────────────────
        self.current_step += 1

        # ── Calculate Reward ──────────────────────────────────────────────
        # Percentage change in total portfolio value (same formula as Tier 1)
        new_price = float(self.closes[self.current_step])
        new_portfolio_value = self.capital + self.shares * new_price
        reward = (new_portfolio_value - self.portfolio_value) / (self.portfolio_value + 1e-10) * 100
        self.portfolio_value = new_portfolio_value
        self.portfolio_history.append(self.portfolio_value)

        # ── Check Termination ─────────────────────────────────────────────
        done = self.current_step >= len(self.data) - 1

        next_state = self._get_state() if not done else None

        return next_state, reward, done


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: EXPERIENCE REPLAY BUFFER
#
# Problem: If we train on consecutive (day1, day2, day3...) experiences,
# each sample is heavily correlated with the last — neural networks train
# poorly on correlated data, they overfit to recent patterns.
#
# Solution (Mnih et al. 2015): Store all experiences in a buffer, then
# sample RANDOM mini-batches. This breaks the correlation and stabilises
# training dramatically.
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer storing (state, action, reward, next_state, done) tuples.

    When full, oldest experiences are overwritten (circular/deque behaviour).
    Random sampling during training breaks temporal correlations.

    Buffer size of 10,000 balances memory usage against diversity of experiences.
    (Mnih et al. used 1,000,000 for Atari — we use less given CPU constraints
    and shorter financial time series. Liu et al. (2020) FinRL uses 10,000-50,000.)
    """

    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        """
        Store one experience tuple.
        next_state is None when episode ends — handled in sample().
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a mini-batch of experiences.

        Returns separate numpy arrays for each component — ready to feed
        directly into TensorFlow operations.
        """
        batch = random.sample(self.buffer, batch_size)

        states      = np.array([e[0] for e in batch], dtype=np.float32)
        actions     = np.array([e[1] for e in batch], dtype=np.int32)
        rewards     = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([
            e[3] if e[3] is not None else np.zeros(DQNTradingEnvironment.STATE_SIZE)
            for e in batch
        ], dtype=np.float32)
        dones       = np.array([e[4] for e in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DQN AGENT
#
# The main agent. Contains:
#   - online_network: the network we train every step
#   - target_network: a frozen copy updated every TARGET_UPDATE_FREQ steps
#   - replay_buffer: the experience store
#
# Training loop (per step):
#   1. Select action using epsilon-greedy on online_network
#   2. Execute action, get (next_state, reward, done)
#   3. Store experience in replay_buffer
#   4. Sample random mini-batch from replay_buffer
#   5. Compute TD targets using target_network (stable targets)
#   6. Update online_network weights via gradient descent on TD error
#   7. Every TARGET_UPDATE_FREQ steps: copy online → target network
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent for stock trading.

    Neural network architecture (following FinRL conventions for financial tasks):
      Input:  11 features (continuous state vector)
      Dense:  64 neurons, ReLU activation
      Dense:  64 neurons, ReLU activation
      Output: 3 neurons (Q-values for Hold, Buy, Sell), linear activation

    Two networks (Mnih et al. 2015):
      online_network — trained every TRAIN_FREQ steps
      target_network — frozen copy, updated every TARGET_UPDATE_FREQ steps
    """

    # ── Hyperparameters ───────────────────────────────────────────────────────
    # These follow Liu et al. (2020) FinRL defaults adjusted for CPU training
    LEARNING_RATE      = 0.001   # Adam optimiser learning rate
    GAMMA              = 0.95    # Discount factor (same as Tier 1 for comparability)
    EPSILON_START      = 1.0     # Initial exploration rate
    EPSILON_MIN        = 0.01    # Minimum exploration rate (same as Tier 1)
    EPSILON_DECAY      = 0.990   # Per-episode decay (same as Tier 1)
    BATCH_SIZE         = 32      # Mini-batch size (Mnih et al. 2015 original value)
    BUFFER_SIZE        = 10000   # Replay buffer capacity
    TARGET_UPDATE_FREQ = 100     # Steps between target network updates (Mnih et al.)
    TRAIN_FREQ         = 4       # Train every N environment steps

    def __init__(self, state_size=11, n_actions=3):
        self.state_size = state_size
        self.n_actions  = n_actions
        self.epsilon    = self.EPSILON_START
        self.step_count = 0  # Total steps taken across all episodes

        # Build both networks with identical architecture
        self.online_network = self._build_network()
        self.target_network = self._build_network()

        # Initialise target network with same weights as online network
        self.target_network.set_weights(self.online_network.get_weights())

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.BUFFER_SIZE)

    def _build_network(self):
        """
        Build the Q-network.

        Architecture rationale:
        - 2 hidden layers of 64 neurons: sufficient capacity for 11-input
          financial state without overfitting risk (Liu et al. 2020 FinRL uses
          64-128 neurons for similar financial tasks)
        - ReLU activation: avoids vanishing gradient, standard for DQN
          (Mnih et al. 2015)
        - Linear output: Q-values are unbounded, so no activation on output
        - Adam optimiser: adaptive learning rates, handles sparse gradients
          well in financial data
        - MSE loss: standard for Q-learning regression target
        """
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(64, activation='relu',
                        kernel_initializer='he_uniform'),  # He init for ReLU
            layers.Dense(64, activation='relu',
                        kernel_initializer='he_uniform'),
            layers.Dense(self.n_actions, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss='mse'
        )

        return model

    def get_action(self, state, training=True):
        """
        Epsilon-greedy action selection.

        During training: with probability epsilon, explore randomly.
        During testing: always exploit (use network's best action).

        Identical behaviour to Tier 1 QLearningAgent.get_action().
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Feed state through network, pick action with highest Q-value
        state_tensor = np.array([state], dtype=np.float32)
        q_values = self.online_network(state_tensor, training=False).numpy()[0]
        return int(np.argmax(q_values))

    def get_q_values(self, state):
        """
        Return Q-values for all actions for a given state.
        Used by the explainability layer to generate trade justifications.
        """
        state_tensor = np.array([state], dtype=np.float32)
        return self.online_network(state_tensor, training=False).numpy()[0]

    def store_experience(self, state, action, reward, next_state, done):
        """Store one (s, a, r, s', done) tuple in the replay buffer."""
        self.replay_buffer.store(state, action, reward, next_state, done)

    def train_step(self):
        """
        One training update using a random mini-batch from the replay buffer.

        TD Target (Bellman equation with target network):
          If done:     y = r
          If not done: y = r + γ * max_a' Q_target(s', a')

        We use the TARGET network (not online network) to compute the max
        Q-value for the next state. This is the key Mnih et al. (2015)
        stabilisation technique — targets don't change every step.

        Returns: training loss (float), for logging
        """
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return None  # Wait until buffer has enough experiences

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.BATCH_SIZE)

        # ── Compute TD Targets ────────────────────────────────────────────
        # Get Q-values for next states from TARGET network (stable)
        next_q_values = self.target_network(next_states, training=False).numpy()
        max_next_q    = np.max(next_q_values, axis=1)

        # Bellman targets: r + γ * max Q(s', a') * (1 - done)
        # The (1 - done) term zeros out the future reward for terminal states
        targets = rewards + self.GAMMA * max_next_q * (1.0 - dones)

        # ── Update Only The Taken Action's Q-value ────────────────────────
        # We don't want to change Q-values for actions we didn't take.
        # Strategy: get current Q-values, only modify the position for the
        # action that was actually taken, then train on the full vector.
        current_q = self.online_network(states, training=False).numpy()
        for i, action in enumerate(actions):
            current_q[i][action] = targets[i]

        # ── Gradient Update ───────────────────────────────────────────────
        history = self.online_network.fit(
            states, current_q,
            epochs=1,
            verbose=0,
            batch_size=self.BATCH_SIZE
        )

        return history.history['loss'][0]

    def update_target_network(self):
        """
        Copy weights from online network to target network.
        Called every TARGET_UPDATE_FREQ steps.
        This is the 'hard update' from Mnih et al. (2015).
        """
        self.target_network.set_weights(self.online_network.get_weights())

    def decay_epsilon(self):
        """Decay exploration rate at end of each episode. Same as Tier 1."""
        self.epsilon = max(self.epsilon * self.EPSILON_DECAY, self.EPSILON_MIN)

    def save(self, filepath='dqn_model.keras'):
        """Save trained model weights to disk."""
        self.online_network.save(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath='dqn_model.keras'):
        """Load trained model weights from disk."""
        self.online_network = keras.models.load_model(filepath)
        self.target_network.set_weights(self.online_network.get_weights())
        print(f"Model loaded from {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: RULE-BASED EXPLAINABILITY
#
# Addresses the "black box" criticism of neural network trading systems.
# For each trade decision, we inspect the 11-dimensional state vector and
# generate a human-readable justification.
#
# This is the rule-based alternative to LLM-generated explanations —
# simpler, deterministic, and appropriate for the retail investor audience.
# Cited motivation: Puiutta & Veith (2020) on explainable RL.
# ─────────────────────────────────────────────────────────────────────────────

def explain_action(state, action, q_values):
    """
    Generate a human-readable explanation for a trading decision.

    Inspects the state vector features and produces natural language
    justification. This makes the system interpretable for retail investors
    without financial expertise (addresses the accessibility gap from
    the literature review).

    Args:
        state: numpy array of shape (11,) — current market state
        action: int, 0=Hold, 1=Buy, 2=Sell
        q_values: numpy array of shape (3,) — network's Q-value estimates

    Returns:
        dict with 'action', 'confidence', 'reasons', 'summary'
    """
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

    # Unpack state features (match order in _get_state)
    rsi           = state[0]   # [0, 1], normalised RSI
    macd          = state[1]   # approx [-1, 1]
    macd_signal   = state[2]   # approx [-1, 1]
    boll_position = state[3]   # [0, 1], Bollinger position
    volume_ratio  = state[4]   # [0, 1], normalised volume
    ma_ratio      = state[5]   # [0, 1], MA trend
    momentum      = state[6]   # [0, 1], 5-day momentum
    volatility    = state[7]   # [0, 1], normalised volatility
    daily_return  = state[8]   # [0, 1], today's return
    position      = state[9]   # 0 or 1
    capital_ratio = state[10]  # [0, 1]

    reasons = []

    # ── RSI interpretation ─────────────────────────────────────────────────
    rsi_raw = rsi * 100  # Convert back to [0, 100] for readability
    if rsi_raw < 30:
        reasons.append(f"RSI is {rsi_raw:.0f} — oversold territory, suggesting potential upward reversal")
    elif rsi_raw > 70:
        reasons.append(f"RSI is {rsi_raw:.0f} — overbought territory, suggesting potential downward correction")
    else:
        reasons.append(f"RSI is {rsi_raw:.0f} — neutral momentum")

    # ── MACD interpretation ────────────────────────────────────────────────
    if macd > macd_signal and macd > 0:
        reasons.append("MACD is above signal line and positive — bullish trend momentum")
    elif macd < macd_signal and macd < 0:
        reasons.append("MACD is below signal line and negative — bearish trend momentum")
    elif macd > macd_signal:
        reasons.append("MACD crossed above signal line — potential bullish crossover")
    else:
        reasons.append("MACD crossed below signal line — potential bearish crossover")

    # ── Bollinger Band interpretation ──────────────────────────────────────
    if boll_position < 0.2:
        reasons.append("Price is near the lower Bollinger Band — potential oversold condition")
    elif boll_position > 0.8:
        reasons.append("Price is near the upper Bollinger Band — potential overbought condition")
    else:
        reasons.append("Price is within normal Bollinger Band range")

    # ── Volume interpretation ──────────────────────────────────────────────
    volume_pct = volume_ratio * 3 * 100  # Approximate % of average
    if volume_ratio > 0.67:  # > 2x average
        reasons.append(f"Volume is elevated ({volume_pct:.0f}% of average) — confirming price move")
    elif volume_ratio < 0.2:
        reasons.append("Volume is low — move may lack conviction")

    # ── Volatility interpretation ──────────────────────────────────────────
    vol_pct = volatility * 60  # Approximate annualised %
    if volatility > 0.67:
        reasons.append(f"High volatility ({vol_pct:.0f}% annualised) — elevated market risk")
    elif volatility < 0.25:
        reasons.append(f"Low volatility ({vol_pct:.0f}% annualised) — calm market conditions")

    # ── Confidence from Q-values ───────────────────────────────────────────
    # Confidence = difference between best and second-best Q-value
    # Higher difference = more decisive recommendation
    sorted_q = np.sort(q_values)[::-1]
    confidence = float(sorted_q[0] - sorted_q[1])
    confidence_pct = min(abs(confidence) / (abs(sorted_q[0]) + 1e-10) * 100, 100)

    # ── Build Summary ──────────────────────────────────────────────────────
    action_name = action_names[action]

    if action == 1:  # BUY
        summary = (f"Recommendation: {action_name}. "
                   f"The agent identifies a potential entry opportunity "
                   f"based on current momentum and technical indicators. "
                   f"Confidence: {confidence_pct:.0f}%.")
    elif action == 2:  # SELL
        summary = (f"Recommendation: {action_name}. "
                   f"The agent recommends exiting the position based on "
                   f"current market signals. "
                   f"Confidence: {confidence_pct:.0f}%.")
    else:  # HOLD
        summary = (f"Recommendation: {action_name}. "
                   f"Current signals do not strongly favour entering or "
                   f"exiting a position. "
                   f"Confidence: {confidence_pct:.0f}%.")

    return {
        'action': action_name,
        'confidence': confidence_pct,
        'reasons': reasons,
        'summary': summary,
        'q_values': {
            'HOLD': float(q_values[0]),
            'BUY':  float(q_values[1]),
            'SELL': float(q_values[2])
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PERFORMANCE METRICS
# Identical to Tier 1 calculate_metrics() for clean comparison.
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(portfolio_values, label='Agent'):
    """
    Calculate standard financial performance metrics.

    Args:
        portfolio_values: list or array of portfolio values over time
        label: string label for printing

    Returns:
        dict with Total Return, Sharpe Ratio, Max Drawdown, Win Rate
    """
    values = np.array(portfolio_values, dtype=float)

    # Total return
    total_return = (values[-1] / values[0] - 1) * 100

    # Daily returns
    daily_returns = np.diff(values) / (values[:-1] + 1e-10)

    # Sharpe ratio (annualised, risk-free rate = 0 for simplicity)
    # Consistent with Tier 1 methodology
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)) * np.sqrt(252)

    # Maximum drawdown
    cummax = np.maximum.accumulate(values)
    drawdown = (values - cummax) / (cummax + 1e-10) * 100
    max_drawdown = float(np.min(drawdown))

    return {
        'Total Return (%)': float(total_return),
        'Sharpe Ratio':     float(sharpe),
        'Max Drawdown (%)': float(max_drawdown),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_agent(agent, train_data, episodes=100, verbose=True):
    """
    Full training loop for the DQN agent.

    Each episode is one complete pass through the training data.
    Within each episode, the agent interacts with the environment
    step-by-step, storing experiences and periodically training.

    Args:
        agent: DQNAgent instance
        train_data: DataFrame of training period price data
        episodes: Number of training episodes
        verbose: Whether to print progress

    Returns:
        dict with training history (rewards, losses, epsilons)
    """
    env = DQNTradingEnvironment(train_data)

    episode_rewards = []
    episode_losses  = []
    epsilons        = []

    print(f"\nTraining DQN Agent for {episodes} episodes...")
    print(f"State size: {DQNTradingEnvironment.STATE_SIZE} features")
    print(f"Training days per episode: {len(train_data)}")
    print(f"Replay buffer size: {agent.BUFFER_SIZE}")
    print(f"Batch size: {agent.BATCH_SIZE}\n")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        losses = []
        done = False

        while not done:
            # 1. Select action
            action = agent.get_action(state, training=True)

            # 2. Execute action
            next_state, reward, done = env.step(action)

            # 3. Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # 4. Train every TRAIN_FREQ steps (once buffer is large enough)
            agent.step_count += 1
            if agent.step_count % agent.TRAIN_FREQ == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            # 5. Update target network periodically
            if agent.step_count % agent.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            total_reward += reward
            state = next_state

        # End of episode
        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(losses) if losses else 0.0)
        epsilons.append(agent.epsilon)

        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss   = np.mean(episode_losses[-10:])
            print(f"Episode {episode+1:3d}/{episodes} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer)}")

    print(f"\nTraining complete.")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Total environment steps: {agent.step_count}")

    return {
        'episode_rewards': episode_rewards,
        'episode_losses':  episode_losses,
        'epsilons':        epsilons
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent, test_data, label='DQN Agent'):
    """
    Evaluate trained agent on test data (no exploration, ε=0).

    Also generates trade-by-trade explanations for the last 5 trades
    using the explainability layer.

    Returns:
        dict with portfolio values, trades, actions, metrics, explanations
    """
    env = DQNTradingEnvironment(test_data)
    state = env.reset()
    done  = False

    actions_taken = []
    explanations  = []

    while not done:
        action   = agent.get_action(state, training=False)
        q_values = agent.get_q_values(state)
        actions_taken.append(action)

        # Generate explanation for non-hold actions (trades)
        if action != 0:
            explanation = explain_action(state, action, q_values)
            explanations.append(explanation)

        next_state, reward, done = env.step(action)
        state = next_state

    portfolio_values = env.portfolio_history
    metrics = calculate_metrics(portfolio_values, label)

    # Print results
    print(f"\n{'='*55}")
    print(f"EVALUATION: {label}")
    print(f"{'='*55}")
    print(f"Total Return:  {metrics['Total Return (%)']:>8.2f}%")
    print(f"Sharpe Ratio:  {metrics['Sharpe Ratio']:>8.2f}")
    print(f"Max Drawdown:  {metrics['Max Drawdown (%)']:>8.2f}%")
    print(f"Trades:        {len(env.trades):>8d}")
    hold_pct = actions_taken.count(0) / len(actions_taken) * 100
    buy_pct  = actions_taken.count(1) / len(actions_taken) * 100
    sell_pct = actions_taken.count(2) / len(actions_taken) * 100
    print(f"Hold/Buy/Sell: {hold_pct:.1f}% / {buy_pct:.1f}% / {sell_pct:.1f}%")

    # Print last few trade explanations
    if explanations:
        print(f"\nSample Trade Explanations (last {min(3, len(explanations))}):")
        for exp in explanations[-3:]:
            print(f"\n  Action: {exp['action']} (Confidence: {exp['confidence']:.0f}%)")
            print(f"  {exp['summary']}")
            for reason in exp['reasons'][:3]:
                print(f"    • {reason}")

    return {
        'portfolio_values': portfolio_values,
        'trades':           env.trades,
        'actions':          actions_taken,
        'metrics':          metrics,
        'explanations':     explanations
    }


def evaluate_buyandhold(test_data):
    """
    Buy-and-hold baseline: buy on day 1, hold until last day.
    Identical to Tier 1 baseline for fair comparison.
    """
    closes = test_data['Close'].values.astype(float)
    initial = 10000.0
    start_price = closes[0]
    portfolio_values = [initial * (p / start_price) for p in closes]

    metrics = calculate_metrics(portfolio_values, 'Buy-and-Hold')

    print(f"\n{'='*55}")
    print(f"EVALUATION: Buy-and-Hold Baseline")
    print(f"{'='*55}")
    print(f"Total Return:  {metrics['Total Return (%)']:>8.2f}%")
    print(f"Sharpe Ratio:  {metrics['Sharpe Ratio']:>8.2f}")
    print(f"Max Drawdown:  {metrics['Max Drawdown (%)']:>8.2f}%")
    print(f"Trades:        {'2':>8}")

    return {
        'portfolio_values': portfolio_values,
        'metrics': metrics
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: VISUALISATIONS
# All charts saved to results/ folder — directly usable in the report.
# ─────────────────────────────────────────────────────────────────────────────

def create_visualisations(train_history, dqn_results, baseline_results,
                           test_data, output_dir='results'):
    """
    Generate all charts for Chapter 4 (Implementation) and
    Chapter 5 (Evaluation) of the final report.

    Charts produced:
      1. dqn_learning_curve.png  — training reward and loss over episodes
      2. dqn_portfolio_comparison.png — DQN vs buy-and-hold equity curve
      3. dqn_action_distribution.png — action frequency during testing
      4. dqn_trades_on_price.png — buy/sell points on price chart
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    # ── Chart 1: Learning Curve ───────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    rewards = train_history['episode_rewards']
    losses  = train_history['episode_losses']
    eps     = range(1, len(rewards) + 1)

    ax1.plot(eps, rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    window = 10
    smoothed = pd.Series(rewards).rolling(window=window).mean()
    ax1.plot(eps, smoothed, linewidth=2, color='steelblue',
             label=f'{window}-Episode Moving Average')
    ax1.set_ylabel('Total Reward', fontsize=11)
    ax1.set_title('DQN Training Progress', fontsize=13, fontweight='bold')
    ax1.legend()

    ax2.plot(eps, losses, alpha=0.3, color='tomato', label='Training Loss')
    smoothed_loss = pd.Series(losses).rolling(window=window).mean()
    ax2.plot(eps, smoothed_loss, linewidth=2, color='tomato',
             label=f'{window}-Episode Moving Average')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('MSE Loss', fontsize=11)
    ax2.set_title('Training Loss (TD Error)', fontsize=13, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dqn_learning_curve.png', dpi=300, bbox_inches='tight')
    print(f"Saved dqn_learning_curve.png")
    plt.close()

    # ── Chart 2: Portfolio Comparison ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    dqn_vals      = dqn_results['portfolio_values']
    baseline_vals = baseline_results['portfolio_values']

    # Align lengths (DQN starts 20 steps in due to indicator warm-up)
    min_len = min(len(dqn_vals), len(baseline_vals))
    days = range(min_len)

    ax.plot(days, dqn_vals[:min_len], label='DQN Agent',
            linewidth=2, color='steelblue')
    ax.plot(days, baseline_vals[:min_len], label='Buy-and-Hold',
            linewidth=2, linestyle='--', color='tomato')
    ax.axhline(y=10000, color='gray', linestyle=':', alpha=0.5,
               label='Initial Capital ($10,000)')

    dqn_ret  = dqn_results['metrics']['Total Return (%)']
    base_ret = baseline_results['metrics']['Total Return (%)']
    ax.set_title(
        f'Portfolio Performance: DQN ({dqn_ret:+.1f}%) vs '
        f'Buy-and-Hold ({base_ret:+.1f}%)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('Trading Days', fontsize=11)
    ax.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
    )

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dqn_portfolio_comparison.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved dqn_portfolio_comparison.png")
    plt.close()

    # ── Chart 3: Action Distribution ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    actions = dqn_results['actions']
    counts  = [actions.count(i) for i in range(3)]
    labels  = ['Hold', 'Buy', 'Sell']
    colours = ['#3498db', '#2ecc71', '#e74c3c']

    ax1.bar(labels, counts, color=colours)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Action Distribution (DQN Testing)', fontsize=12, fontweight='bold')

    ax2.pie(counts, labels=labels, autopct='%1.1f%%',
            colors=colours, startangle=90)
    ax2.set_title('Action Proportions', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dqn_action_distribution.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved dqn_action_distribution.png")
    plt.close()

    # ── Chart 4: Trades on Price Chart ────────────────────────────────────
    trades = dqn_results['trades']
    if trades and len(test_data) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        price_series = test_data['Close'].values.astype(float)
        ax.plot(range(len(price_series)), price_series,
                label='Stock Price', alpha=0.7, linewidth=1.5, color='gray')

        buy_trades  = [t for t in trades if t[0] == 'BUY']
        sell_trades = [t for t in trades if t[0] == 'SELL']

        # Offset step index by START_STEP (environment warm-up)
        start = 20
        for i, trade in enumerate(buy_trades):
            idx = trade[1] - start
            if 0 <= idx < len(price_series):
                ax.scatter(idx, trade[2], color='green', marker='^',
                           s=80, zorder=5,
                           label='Buy' if i == 0 else '')
        for i, trade in enumerate(sell_trades):
            idx = trade[1] - start
            if 0 <= idx < len(price_series):
                ax.scatter(idx, trade[2], color='red', marker='v',
                           s=80, zorder=5,
                           label='Sell' if i == 0 else '')

        ax.set_xlabel('Trading Day', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.set_title('DQN Trade Decisions on Price Chart', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/dqn_trades_on_price.png',
                    dpi=300, bbox_inches='tight')
        print(f"Saved dqn_trades_on_price.png")
        plt.close()

    print(f"\nAll visualisations saved to '{output_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: MAIN EXPERIMENT
# Ties everything together. Run this file directly to train and evaluate.
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(ticker='AAPL', episodes=100):
    """
    Complete DQN experiment: load data, train, evaluate, visualise.

    Data split (same ratio as Tier 1 for comparability):
      Train: first 80%  (~1,100 days for 2018-2024 AAPL)
      Test:  last  20%  (~270 days)

    Args:
        ticker: Stock to train/test on
        episodes: Training episodes (100 matches Tier 1)

    Returns:
        dict containing all results
    """
    print("="*60)
    print("DQN STOCK TRADING AGENT — CM3070 FYP TIER 2")
    print("="*60)

    # ── Load Data ─────────────────────────────────────────────────────────
    data = load_data(ticker, '2018-01-01', '2024-11-01')

    # ── Split ─────────────────────────────────────────────────────────────
    split_idx  = int(len(data) * 0.80)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    test_data  = data.iloc[split_idx:].reset_index(drop=True)

    print(f"\nTicker: {ticker}")
    print(f"Training period: {len(train_data)} days")
    print(f"Testing period:  {len(test_data)} days")

    # ── Initialise Agent ──────────────────────────────────────────────────
    np.random.seed(42)
    tf.random.set_seed(42)
    agent = DQNAgent(state_size=DQNTradingEnvironment.STATE_SIZE, n_actions=3)

    print("\nNetwork architecture:")
    agent.online_network.summary()

    # ── Train ─────────────────────────────────────────────────────────────
    train_history = train_agent(agent, train_data, episodes=episodes)

    # ── Save Model ────────────────────────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    agent.save(f'results/dqn_{ticker}_model.keras')

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)

    dqn_results      = evaluate_agent(agent, test_data, label='DQN Agent')
    baseline_results = evaluate_buyandhold(test_data)

    # ── Comparison Table ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'METRIC':<25} {'DQN':>12} {'BUY & HOLD':>12}")
    print(f"{'─'*55}")
    for metric in ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']:
        dqn_val  = dqn_results['metrics'][metric]
        base_val = baseline_results['metrics'][metric]
        print(f"{metric:<25} {dqn_val:>11.2f}  {base_val:>11.2f}")
    print(f"{'─'*55}")
    print(f"{'Trades':<25} {len(dqn_results['trades']):>11}  {'2':>11}")

    # ── Visualisations ────────────────────────────────────────────────────
    print("\nGenerating visualisations...")
    create_visualisations(train_history, dqn_results, baseline_results, test_data)

    return {
        'agent':            agent,
        'train_history':    train_history,
        'dqn_results':      dqn_results,
        'baseline_results': baseline_results,
        'train_data':       train_data,
        'test_data':        test_data,
    }


if __name__ == '__main__':
    results = run_experiment(ticker='AAPL', episodes=200)
    print("\n✓ Experiment complete. Results saved to results/ folder.")
